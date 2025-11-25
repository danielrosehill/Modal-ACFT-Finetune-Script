"""
Modal ACFT (Adapter-based Continuous Fine-Tuning) for Whisper.

ACFT trains models to work with dynamic audio context (audio_ctx) for variable-length
audio processing. This is needed for FUTO Voice Input and other apps that process
audio clips shorter than 30 seconds without padding.

The approach uses contrastive learning:
- Reference model processes full 30-second context (1500 frames)
- Target model processes variable-length context (scaled to actual audio)
- MSE loss aligns hidden states between the two

Based on: https://github.com/futo-org/whisper-acft
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import modal

# Hugging Face dataset for training
# TODO: Replace with your own dataset (must have 'audio' and 'text'/'sentence' columns)
DATASET_NAME = "your-username/your-whisper-dataset"

# ACFT training defaults
DEFAULT_EPOCHS = 8
DEFAULT_LR = 1e-6
DEFAULT_MAX_AUDIO_LENGTH = 29.0  # seconds
FULL_AUDIO_CTX = 1500  # frames for 30 seconds
CTX_VARIANCE = 64  # random variance in context frames


def build_image() -> modal.Image:
    """Build the base image with all dependencies for ACFT."""
    img = (
        modal.Image.debian_slim(python_version="3.12")
        .apt_install("git", "ffmpeg")
        .pip_install(
            "torch",
            "torchaudio",
            "torchcodec",
            "transformers",
            "datasets",
            "accelerate",
            "soundfile",
            "librosa",
            "huggingface_hub",
            "tqdm",
        )
    )
    return img


def build_hf_secret() -> list[modal.Secret]:
    """Use the Modal-stored Hugging Face secret."""
    return [modal.Secret.from_name("huggingface-token")]


@dataclass(frozen=True)
class ACFTModelConfig:
    app_name: str
    cache_volume: str
    model_id: str
    default_repo: str


# TODO: Replace default_repo values with your own Hugging Face repo paths
MEDIUM_ACFT = ACFTModelConfig(
    app_name="whisper-medium-acft",
    cache_volume="whisper-medium-acft-cache",
    model_id="openai/whisper-medium",
    default_repo="your-username/whisper-acft-medium",
)

TINY_ACFT = ACFTModelConfig(
    app_name="whisper-tiny-acft",
    cache_volume="whisper-tiny-acft-cache",
    model_id="openai/whisper-tiny",
    default_repo="your-username/whisper-acft-tiny",
)

SMALL_ACFT = ACFTModelConfig(
    app_name="whisper-small-acft",
    cache_volume="whisper-small-acft-cache",
    model_id="openai/whisper-small",
    default_repo="your-username/whisper-acft-small",
)

BASE_ACFT = ACFTModelConfig(
    app_name="whisper-base-acft",
    cache_volume="whisper-base-acft-cache",
    model_id="openai/whisper-base",
    default_repo="your-username/whisper-acft-base",
)

# Build image once at module level
BASE_IMAGE_BUILT = build_image()


def _train_acft(
    cfg: ACFTModelConfig,
    repo_name: Optional[str] = None,
    num_epochs: int = DEFAULT_EPOCHS,
    learning_rate: float = DEFAULT_LR,
    max_audio_length: float = DEFAULT_MAX_AUDIO_LENGTH,
):
    """
    Launch ACFT training for a Whisper model.

    ACFT teaches the model to handle variable-length audio context by aligning
    hidden states between a full-context reference model and a dynamic-context
    target model.

    Args:
        cfg: Model configuration.
        repo_name: Hugging Face model repo to push to (private). Defaults to cfg.default_repo.
        num_epochs: Number of training epochs.
        learning_rate: Optimizer learning rate (recommended: 1e-6).
        max_audio_length: Maximum audio length in seconds to process (skip longer).
    """
    import os
    import torch
    from datasets import load_dataset, Audio
    from transformers import WhisperModel, WhisperProcessor, WhisperForConditionalGeneration
    from huggingface_hub import HfApi
    from tqdm import tqdm

    hf_repo = repo_name or cfg.default_repo

    print(f"ACFT Training for {cfg.model_id}")
    print(f"Will push to: {hf_repo}")
    print(f"Epochs: {num_epochs}, LR: {learning_rate}")

    # Load dataset
    print(f"Loading dataset from Hugging Face: {DATASET_NAME}...")
    dataset = load_dataset(DATASET_NAME, split="train", trust_remote_code=True)

    if "audio" in dataset.column_names:
        dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

    print(f"Loaded {len(dataset)} samples")

    # Load models - one for training, one as reference (frozen)
    print("Loading models...")
    model_train = WhisperModel.from_pretrained(cfg.model_id).cuda().train()
    model_base = WhisperModel.from_pretrained(cfg.model_id).cuda().eval()

    # Freeze reference model
    for param in model_base.parameters():
        param.requires_grad = False

    processor = WhisperProcessor.from_pretrained(cfg.model_id)

    # Loss and optimizer
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model_train.parameters(), lr=learning_rate)

    def get_sample(example):
        """Prepare a single sample for ACFT training."""
        audio = example["audio"]
        waveform = audio["array"]
        sampling_rate = audio["sampling_rate"]

        input_features = processor(
            waveform, sampling_rate=sampling_rate, return_tensors="pt"
        ).input_features

        # Get text from appropriate column
        text_key = "text" if "text" in example else "sentence"
        text = example.get(text_key, "")

        return {
            "length": len(waveform) / sampling_rate,
            "input_features": input_features,
            "input_ids": processor.tokenizer.encode(text.lower()),
        }

    def compute_partial_encoder(model, input_features, n_ctx):
        """
        Compute encoder hidden states with partial audio context.

        This simulates whisper.cpp's audio_ctx parameter by only using
        the first n_ctx frames of the encoder output.
        """
        # Get full encoder output
        encoder_outputs = model.encoder(input_features)
        hidden_states = encoder_outputs.last_hidden_state

        # Truncate to n_ctx frames (simulate audio_ctx)
        # The encoder output has shape [batch, seq_len, hidden_dim]
        # seq_len is 1500 for 30 seconds of audio at 50 frames/second
        if n_ctx < hidden_states.shape[1]:
            hidden_states = hidden_states[:, :n_ctx, :]

        return hidden_states

    def compute_hidden_state_loss(example):
        """Compute ACFT loss for a single sample."""
        optimizer.zero_grad()

        # Compute dynamic context based on audio length
        # 1500 frames = 30 seconds, so frames_per_second = 50
        n_ctx = int(round((FULL_AUDIO_CTX / 30.0) * example["length"]))

        # Add random variance for robustness
        extra_ctx = torch.randint(
            -min(CTX_VARIANCE, n_ctx // 3),
            min(CTX_VARIANCE, n_ctx // 3) + 1,
            (1,)
        ).item()
        n_ctx = max(1, n_ctx + extra_ctx)  # Ensure at least 1 frame

        input_features = example["input_features"].cuda()
        input_ids = torch.tensor([example["input_ids"]], dtype=torch.long).cuda()

        # Target model with partial context
        encoder_hidden_partial = compute_partial_encoder(model_train, input_features, n_ctx)
        output_partial = model_train.decoder(
            input_ids=input_ids,
            encoder_hidden_states=encoder_hidden_partial,
            output_hidden_states=True,
        )

        # Reference model with full context (no grad)
        with torch.no_grad():
            encoder_hidden_full = compute_partial_encoder(model_base, input_features, FULL_AUDIO_CTX)
            output_full = model_base.decoder(
                input_ids=input_ids,
                encoder_hidden_states=encoder_hidden_full,
                output_hidden_states=True,
            )

        # MSE loss between hidden states
        loss = criterion(
            torch.cat(output_partial.hidden_states, dim=0),
            torch.cat(output_full.hidden_states, dim=0),
        )

        loss.backward()
        optimizer.step()

        return loss.item()

    # Training loop
    total_losses = []
    for epoch in range(num_epochs):
        print(f"\n=== Epoch {epoch + 1}/{num_epochs} ===")
        epoch_losses = []

        # Shuffle dataset each epoch
        shuffled_ds = dataset.shuffle(seed=epoch)

        pbar = tqdm(shuffled_ds, desc=f"Epoch {epoch + 1}")
        for example in pbar:
            sample = get_sample(example)

            # Skip audio longer than max length
            if sample["length"] > max_audio_length:
                continue

            # Skip if no transcription
            if len(sample["input_ids"]) <= 1:
                continue

            try:
                loss = compute_hidden_state_loss(sample)
                epoch_losses.append(loss)
                pbar.set_postfix({"loss": f"{loss:.6f}"})
            except Exception as e:
                print(f"Warning: Skipped sample due to error: {e}")
                continue

        if epoch_losses:
            avg_loss = sum(epoch_losses) / len(epoch_losses)
            print(f"Epoch {epoch + 1} average loss: {avg_loss:.6f}")
            total_losses.append(avg_loss)

    # Save and push model
    print("\n=== Saving and pushing model ===")

    # Create the full model for inference
    full_model = WhisperForConditionalGeneration.from_pretrained(cfg.model_id)
    full_model.model = model_train.cpu().eval()

    # Save locally first
    save_dir = f"/cache/acft-{cfg.model_id.split('/')[-1]}"
    full_model.save_pretrained(save_dir)
    processor.save_pretrained(save_dir)

    # Push to Hugging Face
    api = HfApi()
    api.create_repo(repo_id=hf_repo, private=True, exist_ok=True)
    api.upload_folder(
        folder_path=save_dir,
        repo_id=hf_repo,
        commit_message=f"ACFT trained model - {num_epochs} epochs, LR {learning_rate}",
    )

    print(f"\nACFT training complete!")
    print(f"Model pushed to: https://huggingface.co/{hf_repo}")
    print(f"Final average losses per epoch: {total_losses}")


# Medium ACFT app
medium_acft_app = modal.App(MEDIUM_ACFT.app_name)
medium_acft_cache = modal.Volume.from_name(MEDIUM_ACFT.cache_volume, create_if_missing=True)


@medium_acft_app.function(
    image=BASE_IMAGE_BUILT,
    gpu="A100-40GB",
    timeout=60 * 60 * 24,
    volumes={"/cache": medium_acft_cache},
    secrets=build_hf_secret(),
)
def train_medium_acft(
    repo_name: Optional[str] = None,
    num_epochs: int = DEFAULT_EPOCHS,
    learning_rate: float = DEFAULT_LR,
    max_audio_length: float = DEFAULT_MAX_AUDIO_LENGTH,
):
    _train_acft(
        MEDIUM_ACFT,
        repo_name=repo_name,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        max_audio_length=max_audio_length,
    )


@medium_acft_app.local_entrypoint()
def main_medium_acft():
    """Local entrypoint for medium ACFT fine-tuning."""
    train_medium_acft.remote()


# Tiny ACFT app
tiny_acft_app = modal.App(TINY_ACFT.app_name)
tiny_acft_cache = modal.Volume.from_name(TINY_ACFT.cache_volume, create_if_missing=True)


@tiny_acft_app.function(
    image=BASE_IMAGE_BUILT,
    gpu="A100-40GB",
    timeout=60 * 60 * 24,
    volumes={"/cache": tiny_acft_cache},
    secrets=build_hf_secret(),
)
def train_tiny_acft(
    repo_name: Optional[str] = None,
    num_epochs: int = DEFAULT_EPOCHS,
    learning_rate: float = DEFAULT_LR,
    max_audio_length: float = DEFAULT_MAX_AUDIO_LENGTH,
):
    _train_acft(
        TINY_ACFT,
        repo_name=repo_name,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        max_audio_length=max_audio_length,
    )


@tiny_acft_app.local_entrypoint()
def main_tiny_acft():
    """Local entrypoint for tiny ACFT fine-tuning."""
    train_tiny_acft.remote()


# Small ACFT app
small_acft_app = modal.App(SMALL_ACFT.app_name)
small_acft_cache = modal.Volume.from_name(SMALL_ACFT.cache_volume, create_if_missing=True)


@small_acft_app.function(
    image=BASE_IMAGE_BUILT,
    gpu="A100-40GB",
    timeout=60 * 60 * 24,
    volumes={"/cache": small_acft_cache},
    secrets=build_hf_secret(),
)
def train_small_acft(
    repo_name: Optional[str] = None,
    num_epochs: int = DEFAULT_EPOCHS,
    learning_rate: float = DEFAULT_LR,
    max_audio_length: float = DEFAULT_MAX_AUDIO_LENGTH,
):
    _train_acft(
        SMALL_ACFT,
        repo_name=repo_name,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        max_audio_length=max_audio_length,
    )


@small_acft_app.local_entrypoint()
def main_small_acft():
    """Local entrypoint for small ACFT fine-tuning."""
    train_small_acft.remote()


# Base ACFT app
base_acft_app = modal.App(BASE_ACFT.app_name)
base_acft_cache = modal.Volume.from_name(BASE_ACFT.cache_volume, create_if_missing=True)


@base_acft_app.function(
    image=BASE_IMAGE_BUILT,
    gpu="A100-40GB",
    timeout=60 * 60 * 24,
    volumes={"/cache": base_acft_cache},
    secrets=build_hf_secret(),
)
def train_base_acft(
    repo_name: Optional[str] = None,
    num_epochs: int = DEFAULT_EPOCHS,
    learning_rate: float = DEFAULT_LR,
    max_audio_length: float = DEFAULT_MAX_AUDIO_LENGTH,
):
    _train_acft(
        BASE_ACFT,
        repo_name=repo_name,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        max_audio_length=max_audio_length,
    )


@base_acft_app.local_entrypoint()
def main_base_acft():
    """Local entrypoint for base ACFT fine-tuning."""
    train_base_acft.remote()


__all__ = [
    "train_medium_acft",
    "train_tiny_acft",
    "train_small_acft",
    "train_base_acft",
    "medium_acft_app",
    "tiny_acft_app",
    "small_acft_app",
    "base_acft_app",
]

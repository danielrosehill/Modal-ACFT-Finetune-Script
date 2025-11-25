# Modal Whisper ACFT Fine-Tuning

Run Whisper ACFT (Adapter-based Continuous Fine-Tuning) on [Modal](https://modal.com) with A100 GPUs.

ACFT trains Whisper models to handle variable-length audio context, enabling better performance in apps like FUTO Voice Input that process audio clips shorter than 30 seconds.

Based on: https://github.com/futo-org/whisper-acft

## Setup

1. Install Modal CLI and authenticate:
   ```bash
   pip install modal
   modal token new
   ```

2. Create a Modal secret named `huggingface-token` with your HF token (needs write access)

3. Update `modal_acft.py`:
   - Set `DATASET_NAME` to your Hugging Face dataset (needs `audio` and `text`/`sentence` columns)
   - Update the `default_repo` values in the model configs to your HF repos

## Usage

Train any Whisper variant:

```bash
# Tiny (fastest)
modal run modal_acft.py::tiny_acft_app

# Base
modal run modal_acft.py::base_acft_app

# Small
modal run modal_acft.py::small_acft_app

# Medium
modal run modal_acft.py::medium_acft_app
```

Models are automatically pushed to your Hugging Face repo as private models.

## Configuration

Edit defaults in `modal_acft.py`:
- `DEFAULT_EPOCHS`: Training epochs (default: 8)
- `DEFAULT_LR`: Learning rate (default: 1e-6)
- `DEFAULT_MAX_AUDIO_LENGTH`: Max audio length in seconds (default: 29.0)

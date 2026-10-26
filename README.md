# 🌟 [EMNLP26 Main, Oral] Rigel: Self-Distilled Score Adaptation for Image and Video Captioning Evaluation
[![Conference](https://img.shields.io/badge/EMNLP-2026-bbeaff.svg)](https://arxiv.org/abs/2606.29997)
[![arXiv](https://img.shields.io/badge/arXiv-2606.29997-b31b1b.svg)](https://arxiv.org/abs/2606.29997)
[![Model](https://img.shields.io/badge/%F0%9F%A4%97%20Model-Rigel-yellow)](https://huggingface.co/ShuitsuKoyama/Rigel)

- Accepted at EMNLP 2026 Main (Oral)
- 🌐 [project page](https://rigel-mnghv.kinsta.page/)
- 📄 [arXiv](https://arxiv.org/abs/2606.29997)
- 🤗 [Model](https://huggingface.co/ShuitsuKoyama/Rigel)

## Quick Installation

To install the Rigel library, run the following command in your terminal:
```bash
pip install git+https://github.com/SHUonkei/Rigel.git
```

To use the Rigel library, follow the steps outlined below.

### Step 1: Importing Required Libraries
Import the necessary libraries from Rigel.
```python
from rigel import from_pretrained
```

### Step 2: Downloading and Loading the Model
Download and load the model using the Rigel library functions. The checkpoint is downloaded from the [Hugging Face Hub](https://huggingface.co/ShuitsuKoyama/Rigel), and the base model ([Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct)) is downloaded automatically.
```python
model = from_pretrained()
```

### Step 3: Preparing the Data
Prepare your data in the specified format. Both videos and images are supported; the modality is determined from the file extension.
```python
media_path = ["test.mp4"]
cand = ["a man is playing guitar on stage"]
refs = [["a musician performs a guitar solo on stage", "a man plays an electric guitar in front of a crowd"]]
```

### Step 4: Making Scores
Use the model to make predictions on your data.
```python
scores = model(media_path=media_path, cand=cand, refs=refs)
print(scores)
```

### Note
- Ensure that `test.mp4` is correctly located in your working directory.
- For reference-free evaluation, pass an empty list for each sample (e.g., `refs=[[]]`).
- A CUDA-compatible NVIDIA GPU is required.

```python
from rigel import from_pretrained

model = from_pretrained()

# Data must be in the following format:
media_path = ["test.mp4"]
cand = ["a man is playing guitar on stage"]
refs = [["a musician performs a guitar solo on stage", "a man plays an electric guitar in front of a crowd"]]

scores = model(media_path=media_path, cand=cand, refs=refs)
print(scores)
```

A command-line interface is also available:
```bash
rigel-score --media test.mp4 --cand "a man is playing guitar on stage" --ref "a musician performs a guitar solo on stage"
```

## Instructions (for researchers)

We assume the following environment for our experiments:

- Python 3.10
- [uv](https://github.com/astral-sh/uv) for dependency management
- PyTorch version 2.5.1 with CUDA 12.1 support

### Clone & Install

```bash
git clone git@github.com:SHUonkei/Rigel.git
cd Rigel
```

```bash
uv sync
```

### Checkpoint

The checkpoint is hosted on [Hugging Face](https://huggingface.co/ShuitsuKoyama/Rigel) and is downloaded automatically by the evaluation scripts.

### Evaluation

```bash
# VATEX-EVAL
uv run python src/evaluate/vatex_eval.py \
    --json_path data/vatex_eval/vid_cans_score_dict_available.json \
    --video_dir data/vatex_eval/videos \
    --output_dir results/vatex_eval --use_n_refs 1

# ActivityNet-FOIL
uv run python src/evaluate/activitynet_foil.py \
    --storage_path data/anet_foil --output_dir results/activitynet_foil --use_references

# Composite / Flickr8k / Nebula / FOIL
uv run python src/evaluate/composite.py --data_root data/composite --output_dir results/composite
uv run python src/evaluate/flickr8k.py --data_root data/flickr8k --output_dir results/flickr8k --all
uv run python src/evaluate/nebula_test.py --data_root data/nebula --output_dir results/nebula
uv run python src/evaluate/foil.py --data_root data/foil --output_dir results/foil --all
```

Most scripts accept `--no_references` for reference-free evaluation.

## License

This work is licensed under the BSD-3-Clause license. To view a copy of this license, see [LICENSE](LICENSE).

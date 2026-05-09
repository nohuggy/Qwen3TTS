# 🎙️ Qwen3TTS

## 🚀 Installation & Launch

### 1. Google Colab
Run it in Google Colab: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nohuggy/Qwen3TTS/blob/main/colab.ipynb)

## 📦 Local Setup (Modular Version)
This repository follows the **OmniVoice** structure.

### Requirements
```bash
pip install -r requirements_colab.txt
```

### Run
```bash
python -m omnivoice.app_colab
```

## ✨ Features
- **Voice Cloning**: Clone voices with 3+ seconds of audio.
- **Custom Voice**: 9 preset character voices with style control.
- **Voice Design**: Create unique voices from text descriptions.
- **Optimized**: Uses FP16, SDPA, and TF32 for maximum performance on T4 GPUs.

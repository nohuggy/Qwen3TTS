# Qwen3-TTS: Multi-Speaker Role Engine

A professional-grade Text-to-Speech engine built on the Qwen3-TTS architecture, optimized for complex script production, multi-character dialogue, and high-fidelity voice cloning.

## Quick Start

### 1. Google Colab
The easiest way to run Qwen3-TTS is via Google Colab:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nohuggy/Qwen3TTS/blob/main/colab.ipynb)

### 2. Local Setup
```bash
pip install -r requirements_colab.txt
python -m omnivoice.app_colab
```

## Key Features

### 1. Multi-Speaker Voice Cloning & Design
*   **Concurrent Roles**: Manage up to 3 distinct speaker profiles simultaneously in Voice Cloning.
*   **Multi-Character Voice Design**: Synthesize entire multi-character dialogues purely from detailed text descriptions (Description-to-Voice).
*   **Speaker Switching**: Use `## Name ##` to seamlessly switch between voices in both Voice Cloning and Voice Design.
*   **Emotion & Pause Control**: Use `[emotion]` tags (e.g., `[shout]`, `[happy]`) and `[pause:X.X]` markers for granular direction.

**Example Script:**
```text
## Alex ## [shout] Hey! [pause:0.5] Over here!
## Sara ## [happy] Oh! I see you now.
```

### 2. Role Bank & Voice Compilation
*   **Role Maker**: Instantly compile a reference audio clip and transcript into a highly portable `.qwen3tts` voice file.
*   **Auto-Compilation**: Automatically extract and compile unique `.qwen3tts` profiles for each newly synthesized character directly from the Voice Design tab.
*   **Flexible Loading**: Load `.qwen3tts` files directly into Role Bank panels without needing to re-process reference audio.

### 3. Professional Subtitle Engine (Pure SRT)
*   **Perfect Word-Level Alignment**: Integrates the Qwen3-ForcedAligner-0.6B model to generate frame-accurate, word-level SRT timestamps—even across highly emotional, multi-speaker generated audio.
*   **Smart Cleaning**: Automatically scrubs all technical metadata, structural tags, and speaker markers before forced alignment.
*   **Natural Splitting**: Intelligently balances subtitle blocks based on sentence structure and word count for maximum readability.

### 4. Advanced Production Features
*   **Custom Voice**: 9 preset studio-grade characters (`aiden`, `dylan`, `eric`, `ono_anna`, `ryan`, `serena`, `sohee`, `uncle_fu`, `vivian`) with instruction-driven style control.
*   **All-In-One ZIP Export**: Download your final `.wav` audio, `.srt` subtitle track, and all auto-compiled `.qwen3tts` voice profiles packaged perfectly into a single ZIP archive.
*   **Dynamic UI**: A responsive interface that lazily loads resources to save VRAM, auto-hides inactive preview panels to reduce clutter, and provides real-time MP3 previews the moment audio synthesis completes.

## Optimization & Performance
*   **Hardware Agnostic**: Automatic detection and optimization for CUDA (GPU) and CPU environments.
*   **Memory Management**: Aggressive garbage collection, sequential model loading/unloading, and `flash-attn` optimizations ensure complex multi-character designs render cleanly without VRAM spikes.
*   **Speed**: Leverages FP16 and TF32 optimizations for maximum throughput on T4/A100 GPUs.

## Models
*   **Synthesis**: Qwen3-TTS-12Hz-1.7B (Base, Custom, Design)
*   **ASR**: Qwen3-ASR-0.6B
*   **Alignment**: Qwen3-ForcedAligner-0.6B

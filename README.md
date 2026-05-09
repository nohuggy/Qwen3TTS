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

### 1. Multi-Speaker Role Bank
*   **Concurrent Roles**: Manage up to 3 distinct speaker profiles simultaneously.
*   **Flexible Cloning**: Define roles using Name, Reference Audio, and Transcripts.
*   **Reliable Import**: Advanced ZIP extraction with metadata validation to prevent duration errors.
*   **Role Management**: Individual "Clear" controls for each role to reset state and manage memory.

### 2. Advanced Script Parsing
Qwen3-TTS supports a specialized script format for professional dialogue production:
*   **Speaker Switching**: Use `## Name ##` to switch between voices in your Role Bank.
*   **Emotion Control**: Use `[emotion]` tags (e.g., `[shout]`, `[happy]`, `[whisper]`) to set the tone for specific segments.
*   **Precision Pauses**: Insert `[pause:X.X]` anywhere to inject specific durations of silence.

**Example Script:**
```text
## Alex ## [shout] Hey! [pause:0.5] Over here!
## Sara ## [happy] Oh! I see you now.
## Bob ## [angry] Get back here!
```

### 3. Professional Subtitle Engine (Pure SRT)
*   **Smart Cleaning**: Automatically filters out technical metadata (role names, emotion tags, and pause markers) from generated subtitles.
*   **Natural Splitting**: Preserves speaker boundaries and sentence structure for perfectly timed, readable SRT files.
*   **ASR Alignment**: Uses Qwen3-ASR and Forced Aligner for word-level precision.

### 4. Advanced Voice Modes
*   **Custom Voice**: 9 preset studio-grade characters with high-fidelity style control.
*   **Voice Design**: Synthesize unique voices from detailed text descriptions (Description-to-Voice).

## Optimization & Performance
*   **Hardware Agnostic**: Automatic detection and optimization for CUDA (GPU) and CPU environments.
*   **Speed**: Uses FP16, SDPA, and TF32 optimizations for maximum throughput on T4/A100 GPUs.
*   **UI Architecture**: Minimalist, emoji-free interface with threaded background processing for a responsive user experience.

## Models
*   **Synthesis**: Qwen3-TTS-12Hz-1.7B (Base, Custom, Design)
*   **ASR**: Qwen3-ASR-0.6B
*   **Alignment**: Qwen3-ForcedAligner-0.6B

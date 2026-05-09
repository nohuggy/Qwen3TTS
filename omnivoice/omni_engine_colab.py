import torch
import gc
import time
import soundfile as sf
import tempfile
from qwen_tts import Qwen3TTSModel

# Global variables
current_model = None
current_model_type = None

# Device detection
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

# Enable PyTorch optimizations if GPU is available
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.conv.fp32_precision = 'tf32'
    torch.backends.cuda.matmul.fp32_precision = 'tf32'
    print(f"🚀 Running on GPU: {torch.cuda.get_device_name(0)}")
else:
    print("⚠️ CUDA not available. Running on CPU (this will be slow).")

def load_model(model_type):
    """Load model with appropriate hardware optimization"""
    global current_model, current_model_type

    if current_model_type == model_type:
        print(f"✅ Using cached {model_type} model")
        return current_model

    if current_model is not None:
        print(f"Unloading {current_model_type} model...")
        del current_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"Loading {model_type} model (1.7B) on {DEVICE}...")
    start = time.time()

    try:
        if model_type == "base":
            model_name = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
        elif model_type == "custom":
            model_name = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
        elif model_type == "design":
            model_name = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"

        current_model = Qwen3TTSModel.from_pretrained(
            model_name,
            torch_dtype=DTYPE,
            device_map=DEVICE,
            attn_implementation="sdpa" if torch.cuda.is_available() else "eager"
        )

        current_model_type = model_type
        load_time = time.time() - start

        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0) / 1024**3
            print(f"✅ Loaded in {load_time:.1f}s | GPU: {allocated:.2f}GB")
        else:
            print(f"✅ Loaded in {load_time:.1f}s")

        return current_model

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def voice_clone(text, reference_audio, ref_transcript):
    """Generate speech by cloning a reference voice"""
    if not text or not reference_audio:
        return None

    try:
        total_start = time.time()
        model = load_model("base")
        if model is None:
            return None

        print(f"⏱️ Creating prompt...")
        prompt_start = time.time()

        # Logic: If transcript is provided, use high-quality mode.
        # If empty, fallback to x-vector only mode (Fast Mode).
        if ref_transcript and ref_transcript.strip():
            print("   Mode: High-Quality (using transcript)")
            prompt_items = model.create_voice_clone_prompt(
                ref_audio=reference_audio,
                ref_text=ref_transcript,
                x_vector_only_mode=False
            )
        else:
            print("   Mode: Standard (no transcript, using x-vector fallback)")
            prompt_items = model.create_voice_clone_prompt(
                ref_audio=reference_audio,
                x_vector_only_mode=True
            )

        prompt_time = time.time() - prompt_start
        print(f"   Prompt: {prompt_time:.1f}s")

        print(f"⏱️ Generating audio...")
        gen_start = time.time()

        with torch.inference_mode():
            wavs, sr = model.generate_voice_clone(
                text=text,
                voice_clone_prompt=prompt_items
            )

        gen_time = time.time() - gen_start

        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        sf.write(temp_file.name, wavs[0], sr)

        total_time = time.time() - total_start
        audio_duration = len(wavs[0]) / sr
        rtf = gen_time / audio_duration

        print(f"✅ Done! Total: {total_time:.1f}s | Gen: {gen_time:.1f}s | Audio: {audio_duration:.1f}s | RTF: {rtf:.2f}x")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        return temp_file.name

    except Exception as e:
        print(f"❌ Error in voice_clone: {str(e)}")
        return None

def custom_voice(text, voice_name, instruction):
    """Generate speech using preset voices"""
    if not text:
        return None

    try:
        total_start = time.time()
        model = load_model("custom")
        if model is None:
            return None

        print(f"⏱️ Generating with voice: {voice_name}...")
        gen_start = time.time()

        with torch.inference_mode():
            if instruction and instruction.strip():
                wavs, sr = model.generate_custom_voice(
                    text=text,
                    speaker=voice_name,
                    instruct=instruction
                )
            else:
                wavs, sr = model.generate_custom_voice(
                    text=text,
                    speaker=voice_name
                )

        gen_time = time.time() - gen_start

        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        sf.write(temp_file.name, wavs[0], sr)

        total_time = time.time() - total_start
        audio_duration = len(wavs[0]) / sr
        rtf = gen_time / audio_duration

        print(f"✅ Done! Total: {total_time:.1f}s | Gen: {gen_time:.1f}s | Audio: {audio_duration:.1f}s | RTF: {rtf:.2f}x")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        return temp_file.name

    except Exception as e:
        print(f"❌ Error in custom_voice: {str(e)}")
        return None

def voice_design(text, voice_description):
    """Generate speech from text description"""
    if not text or not voice_description:
        return None

    try:
        total_start = time.time()
        model = load_model("design")
        if model is None:
            return None

        print(f"⏱️ Generating...")
        gen_start = time.time()

        with torch.inference_mode():
            wavs, sr = model.generate_voice_design(
                text=text,
                instruct=voice_description
            )

        gen_time = time.time() - gen_start

        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        sf.write(temp_file.name, wavs[0], sr)

        total_time = time.time() - total_start
        audio_duration = len(wavs[0]) / sr
        rtf = gen_time / audio_duration

        print(f"✅ Done! Total: {total_time:.1f}s | Gen: {gen_time:.1f}s | Audio: {audio_duration:.1f}s | RTF: {rtf:.2f}x")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        return temp_file.name

    except Exception as e:
        print(f"❌ Error in voice_design: {str(e)}")
        return None

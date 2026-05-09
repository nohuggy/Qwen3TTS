import torch
import gc
import time
import soundfile as sf
import tempfile
from qwen_tts import Qwen3TTSModel

# Global variables
current_model = None
current_model_type = None
ASR_PIPE = None

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

def setup_asr():
    """Perform initial ASR setup: install package and pre-download weights"""
    print("🛠️ Initializing ASR Setup...")
    try:
        import qwen_asr
        print("   ✅ qwen-asr already installed.")
    except ImportError:
        print("   📥 Installing qwen-asr package...")
        try:
            import subprocess
            subprocess.run(["pip", "install", "qwen-asr"], check=True)
            print("   ✅ qwen-asr installed successfully.")
        except Exception as e:
            print(f"   ❌ Failed to install qwen-asr: {e}")

    try:
        print("   📥 Pre-warming Qwen3-ASR-0.6B cache (HuggingFace)...")
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id="Qwen/Qwen3-ASR-0.6B")
        print("   ✅ ASR weights cached and ready.")
    except Exception as e:
        print(f"   ⚠️ Could not pre-warm ASR cache: {e}")

# Run setup at boot
setup_asr()

def load_model(model_type):
    """Load model with appropriate hardware optimization"""
    global current_model, current_model_type

    if current_model_type == model_type:
        print(f"✅ Using cached {model_type} model")
        return current_model

    if current_model is not None:
        print(f"Unloading {current_model_type} model...")
        del current_model
        current_model = None
        current_model_type = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Ensure ASR is unloaded before loading TTS
    unload_asr()

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

        # Logic: Prioritize high-quality cloning.
        # If ref_transcript is provided, use it.
        # If empty, attempt to use the model's internal ASR (x_vector_only_mode=False).
        if ref_transcript and ref_transcript.strip():
            print("   Mode: High-Quality (using manual transcript)")
            prompt_items = model.create_voice_clone_prompt(
                ref_audio=reference_audio,
                ref_text=ref_transcript,
                x_vector_only_mode=False
            )
        else:
            print("   Mode: High-Quality (attempting Auto-ASR)...")
            try:
                # In the official workflow, omitting ref_text with x_vector_only_mode=False 
                # should trigger internal ASR for alignment.
                prompt_items = model.create_voice_clone_prompt(
                    ref_audio=reference_audio,
                    x_vector_only_mode=False
                )
            except Exception as e:
                print(f"   ⚠️ Auto-ASR failed: {str(e)}. Falling back to Standard Mode.")
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
        # Ensure audio is on CPU and in numpy format for soundfile
        audio_data = wavs[0].cpu().numpy() if torch.is_tensor(wavs[0]) else wavs[0]
        sf.write(temp_file.name, audio_data, sr)

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
        # Ensure audio is on CPU and in numpy format for soundfile
        audio_data = wavs[0].cpu().numpy() if torch.is_tensor(wavs[0]) else wavs[0]
        sf.write(temp_file.name, audio_data, sr)

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
        # Ensure audio is on CPU and in numpy format for soundfile
        audio_data = wavs[0].cpu().numpy() if torch.is_tensor(wavs[0]) else wavs[0]
        sf.write(temp_file.name, audio_data, sr)

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

def get_asr_pipe():
    """Load Qwen3-ASR-0.6B model"""
    global ASR_PIPE
    if ASR_PIPE is None:
        print("Loading Qwen3-ASR-0.6B model...")
        # Unload TTS model first to save VRAM
        global current_model, current_model_type
        if current_model is not None:
            print(f"Unloading {current_model_type} model before ASR...")
            del current_model
            current_model = None
            current_model_type = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        try:
            from qwen_asr import Qwen3ASRModel
            ASR_PIPE = Qwen3ASRModel.from_pretrained(
                "Qwen/Qwen3-ASR-0.6B",
                dtype=DTYPE,
                device_map=DEVICE,
                attn_implementation="sdpa" if torch.cuda.is_available() else "eager"
            )
            print("✅ ASR model loaded")
        except Exception as e:
            print(f"⚠️ Failed to load Qwen3ASRModel: {e}")
            print("Attempting to use transformers pipeline as final fallback...")
            try:
                from transformers import pipeline
                ASR_PIPE = pipeline(
                    "automatic-speech-recognition",
                    model="Qwen/Qwen3-ASR-0.6B",
                    device=DEVICE,
                    torch_dtype=DTYPE,
                    trust_remote_code=True
                )
                print("✅ ASR model loaded (Pipeline Fallback)")
            except Exception as e2:
                print(f"❌ Final fallback failed: {e2}")
                raise ImportError("qwen-asr is required for Qwen3-ASR. Please run 'pip install qwen-asr'")
    return ASR_PIPE

def unload_asr():
    """Unload ASR model to save VRAM"""
    global ASR_PIPE
    if ASR_PIPE is not None:
        print("🗑️ Unloading ASR Engine...")
        del ASR_PIPE
        ASR_PIPE = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def transcribe_ref(audio_path):
    """Transcribe reference audio using Qwen3-ASR"""
    if not audio_path:
        return ""
    try:
        model = get_asr_pipe()
        # Handle both Qwen3ASRModel and pipeline
        if hasattr(model, "transcribe"):
            results = model.transcribe(audio=audio_path)
            return results[0].text if results else ""
        else:
            result = model(audio_path)
            return result.get("text", "").strip()
    except Exception as e:
        print(f"❌ Transcription Error: {e}")
        return f"Error: {e}"

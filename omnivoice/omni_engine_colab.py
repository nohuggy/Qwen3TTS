import torch
import gc
import time
import soundfile as sf
import tempfile
import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
from qwen_tts import Qwen3TTSModel

# Global variables
current_model = None
current_model_type = None
ASR_PIPE = None
ALIGNER_PIPE = None

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
        snapshot_download(repo_id="Qwen/Qwen3-ForcedAligner-0.6B")
        print("   ✅ ASR & Aligner weights cached and ready.")
    except Exception as e:
        print(f"   ⚠️ Could not pre-warm ASR/Aligner cache: {e}")

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

    # Ensure ASR and Aligner are unloaded before loading TTS
    unload_asr()
    unload_aligner()

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

def format_timestamp(seconds):
    """Format seconds into SRT timestamp format (HH:MM:SS,mmm)"""
    td = float(seconds)
    hours = int(td // 3600)
    minutes = int((td % 3600) // 60)
    secs = int(td % 60)
    millis = int((td % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

def unify_punctuation(text):
    """Clean up and unify punctuation for better TTS processing (Original OmniVoice Rules)"""
    if not text: return ""
    import re
    # Ellipsis: Convert ..., ⋯⋯, 。。。 to standard ……
    text = re.sub(r'(\.\.\.+|…+|⋯+|。。。+)', '……', text)
    # Title Marks: Convert 〈 〉, 『 』, 「 」 to standard 《 》 or 〈 〉
    text = text.replace('『', '「').replace('』', '」') 
    # Quotation marks: Inner 『 』 -> ‘ ’ , Outer 「 」 -> “ ”
    text = text.replace('「', '“').replace('」', '”')
    text = text.replace('『', '‘').replace('』', '’')
    
    # Also include the basic full-width mappings for safety
    replacements = {
        '，': ',', '。': '.', '！': '!', '？': '?', 
        '；': ';', '：': ':', '（': '(', '）': ')'
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text

def voice_clone(text, reference_audio, ref_transcript, gen_srt=False, convert_punc=False):
    """Generate speech by cloning a reference voice"""
    if not text or not reference_audio:
        return None, ""
    
    if convert_punc:
        text = unify_punctuation(text)
        if ref_transcript:
            ref_transcript = unify_punctuation(ref_transcript)

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

        srt_content = ""
        if gen_srt:
            srt_content = generate_srt(text, temp_file.name)

        return temp_file.name, srt_content

    except Exception as e:
        print(f"❌ Error in voice_clone: {str(e)}")
        return None, ""

def custom_voice(text, voice_name, instruction, gen_srt=False, convert_punc=False):
    """Generate speech using preset voices"""
    if not text:
        return None, ""

    if convert_punc:
        text = unify_punctuation(text)

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

        srt_content = ""
        if gen_srt:
            srt_content = generate_srt(text, temp_file.name)

        return temp_file.name, srt_content

    except Exception as e:
        print(f"❌ Error in custom_voice: {str(e)}")
        return None, ""

def voice_design(text, voice_description, gen_srt=False, convert_punc=False):
    """Generate speech from text description"""
    if not text or not voice_description:
        return None, ""

    if convert_punc:
        text = unify_punctuation(text)

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

        srt_content = ""
        if gen_srt:
            srt_content = generate_srt(text, temp_file.name)

        return temp_file.name, srt_content

    except Exception as e:
        print(f"❌ Error in voice_design: {str(e)}")
        return None, ""

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

def get_aligner_pipe():
    """Load Qwen3-ForcedAligner-0.6B model"""
    global ALIGNER_PIPE
    if ALIGNER_PIPE is None:
        print("Loading Qwen3-ForcedAligner-0.6B model...")
        # Unload TTS and ASR first to save VRAM
        global current_model, current_model_type
        if current_model is not None:
            print(f"Unloading {current_model_type} model before Aligner...")
            del current_model
            current_model = None
            current_model_type = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        unload_asr()
        
        try:
            # The Forced Aligner is typically loaded via Qwen3ASRModel with specific config
            # Or if it's a standalone model, we use it directly.
            # According to Qwen3-ASR docs, we can use the ASR model with return_time_stamps=True
            # which internally uses the Forced Aligner.
            # To save VRAM, we'll load the ASR-1.7B or ASR-0.6B if we need alignment.
            # However, since the user asked for Qwen3-ForcedAligner-0.6B specifically:
            from qwen_asr import Qwen3ASRModel
            ALIGNER_PIPE = Qwen3ASRModel.from_pretrained(
                "Qwen/Qwen3-ASR-0.6B", # Use the ASR model as the frontend
                dtype=DTYPE,
                device_map=DEVICE,
                forced_aligner="Qwen/Qwen3-ForcedAligner-0.6B",
                attn_implementation="sdpa" if torch.cuda.is_available() else "eager"
            )
            print("✅ Forced Aligner loaded")
        except Exception as e:
            print(f"❌ Aligner Load Error: {e}")
            return None
    return ALIGNER_PIPE

def unload_aligner():
    """Unload Aligner model to save VRAM"""
    global ALIGNER_PIPE
    if ALIGNER_PIPE is not None:
        print("🗑️ Unloading Aligner Engine...")
        del ALIGNER_PIPE
        ALIGNER_PIPE = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def generate_srt(text, audio_path):
    """Generate SRT subtitles using Forced Aligner"""
    if not text or not audio_path:
        return ""
    try:
        model = get_aligner_pipe()
        if model is None: return ""
        
        print("🔍 Generating subtitles (Forced Aligner)...")
        # For alignment, we provide the reference text
        results = model.transcribe(
            audio=audio_path,
            context=text,
            return_time_stamps=True
        )
        
        if not results or not results[0].time_stamps:
            print("⚠️ No timestamps generated.")
            return ""
        
        srt_content = ""
        # Group word-level timestamps into readable segments
        # Qwen3-ASR returns a list of Timestamp objects
        ts = results[0].time_stamps
        
        # Simple grouping: 10 words per line
        words_per_line = 10
        for i in range(0, len(ts), words_per_line):
            chunk = ts[i:i+words_per_line]
            start = format_timestamp(chunk[0].start_time)
            end = format_timestamp(chunk[-1].end_time)
            line_text = "".join([c.text for c in chunk]).strip()
            srt_content += f"{(i//words_per_line)+1}\n{start} --> {end}\n{line_text}\n\n"
        
        print("✅ SRT generated successfully")
        # Auto offload after task
        unload_aligner()
        return srt_content.strip()
    except Exception as e:
        print(f"❌ SRT Generation Error: {e}")
        import traceback
        traceback.print_exc()
        unload_aligner()
        return f"SRT Error: {e}"

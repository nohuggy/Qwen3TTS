import torch
import gc
import time
import soundfile as sf
import tempfile
import os
import re
import difflib
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
        time.sleep(1) # Breathe

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

def get_slug(text, max_tokens=8):
    """Generate a clean filename slug from text (OmniVoice Rules)"""
    if not text: return "output"
    clean_text = re.sub(r"[^\w\s\u4e00-\u9fff]", "", text)
    tokens = re.findall(r"[\u4e00-\u9fff]|[a-zA-Z0-9]+", clean_text)
    selected = tokens[:max_tokens]
    if not selected:
        return "output"
    res = selected[0]
    for i in range(1, len(selected)):
        prev, curr = selected[i - 1], selected[i]
        if re.match(r"[a-zA-Z0-9]", prev) or re.match(r"[a-zA-Z0-9]", curr):
            res += " " + curr
        else:
            res += curr
    return res.strip()

def smart_balanced_split(text, target_words=14, max_words=22):
    """Split original text into readable segments for subtitles (Optimized for semantic units)"""
    if not text: return []
    # Split into paragraphs first
    paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
    all_segments = []
    
    for p_text in paragraphs:
        p_text = re.sub(r'\s+', ' ', p_text).strip()
        # Pattern captures tokens with surrounding punctuation
        pattern = re.compile(r'([^\w\s\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]*)([\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]|[a-zA-Z0-9-]+)([^\w\s\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af\(\[\{\u300c\u300e\u300a\u3008\u201c\u2018\uFF08]*)(\s*)')
        tokens = []
        for match in pattern.finditer(p_text):
            lead_punct, word, trail_punct, space = match.groups()
            tokens.append(lead_punct + word + trail_punct + space)
        
        if not tokens: continue

        def split_recursive(tkns):
            if len(tkns) <= max_words:
                return ["".join(tkns).strip()]
            n = max(2, round(len(tkns) / target_words))
            avg = len(tkns) / n
            ideal_end = int(avg)
            best_break = ideal_end
            min_p = 1000
            for offset in range(-6, 7):
                idx = ideal_end + offset
                if idx <= 0 or idx >= len(tkns): continue
                p_orphan = 100 if (idx < 3 or (len(tkns) - idx) < 3) else 0
                t = tkns[idx - 1]
                p = abs(offset) * 4 + p_orphan
                if any(x in t for x in "。！？.!?;；…"): p -= 80
                elif any(x in t for x in "，,、"): p -= 40
                elif any(x in t for x in "”’」』"): p -= 20
                else: p += 40
                if p < min_p:
                    min_p = p
                    best_break = idx
            return split_recursive(tkns[:best_break]) + split_recursive(tkns[best_break:])

        all_segments.extend(split_recursive(tokens))
    return all_segments

def align_robust(user_segments, aligner_tokens):
    """Align user-provided text segments to aligner timestamps"""
    user_clean = [re.sub(r'[^\w\u4e00-\u9fff]', '', s).lower() for s in user_segments]
    
    # Create mapping of clean aligner chars to timestamps
    char_times = []
    for c in aligner_tokens:
        txt = c.text
        s, e = c.start_time, c.end_time
        c_clean = re.sub(r'[^\w\u4e00-\u9fff]', '', txt).lower()
        if not c_clean: continue
        duration = e - s
        for i in range(len(c_clean)):
            char_times.append((s + (i / len(c_clean)) * duration, s + ((i + 1) / len(c_clean)) * duration))
            
    user_full_clean = "".join(user_clean)
    aligner_clean = "".join([re.sub(r'[^\w\u4e00-\u9fff]', '', c.text).lower() for c in aligner_tokens])
    matcher = difflib.SequenceMatcher(None, user_full_clean, aligner_clean)
    
    mapping = [None] * len(user_full_clean)
    for u_s, w_s, length in matcher.get_matching_blocks():
        for i in range(length):
            if w_s + i < len(char_times):
                mapping[u_s + i] = char_times[w_s + i]
                
    matched_indices = [i for i, x in enumerate(mapping) if x is not None]
    if not matched_indices:
        total_dur = char_times[-1][1] if char_times else 10.0
        for i in range(len(mapping)):
            mapping[i] = ((i / len(mapping)) * total_dur, ((i + 1) / len(mapping)) * total_dur)
    else:
        # Simple interpolation for gaps
        first_idx = matched_indices[0]
        first_s = mapping[first_idx][0]
        for i in range(first_idx):
            mapping[i] = ((i / first_idx) * first_s, ((i + 1) / first_idx) * first_s)
            
        for j in range(len(matched_indices) - 1):
            idx1, idx2 = matched_indices[j], matched_indices[j+1]
            t1, t2 = mapping[idx1][1], mapping[idx2][0]
            gap_len = idx2 - idx1 - 1
            if gap_len > 0:
                for k in range(1, gap_len + 1):
                    mapping[idx1 + k] = (t1 + ((k-1)/gap_len)*(t2-t1), t1 + (k/gap_len)*(t2-t1))
                    
        last_idx = matched_indices[-1]
        last_e = mapping[last_idx][1]
        total_end = char_times[-1][1] if char_times else last_e + 2.0
        rem_len = len(mapping) - 1 - last_idx
        if rem_len > 0:
            for k in range(1, rem_len + 1):
                mapping[last_idx + k] = (last_e + ((k-1)/rem_len)*(total_end-last_e), last_e + (k/rem_len)*(total_end-last_e))
        
    results = []
    curr = 0
    for s_clean in user_clean:
        if not s_clean:
            results.append((results[-1][1] if results else 0.0, (results[-1][1] if results else 0.0) + 1.0))
            continue
        start_t = mapping[curr][0]
        end_t = mapping[curr + len(s_clean) - 1][1]
        results.append((start_t, end_t))
        curr += len(s_clean)
    return results

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
        
        # Explicitly clear audio tensors to free VRAM before alignment
        del wavs
        gc.collect()
        time.sleep(1) # Breathe

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
        
        # Explicitly clear audio tensors to free VRAM before alignment
        del wavs
        gc.collect()
        time.sleep(1) # Breathe

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
        
        # Explicitly clear audio tensors to free VRAM before alignment
        del wavs
        gc.collect()
        time.sleep(1) # Breathe

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
            time.sleep(1) # Breathe
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
    """Generate SRT subtitles using Forced Aligner and Robust Splitter"""
    if not text or not audio_path:
        return ""
    try:
        model = get_aligner_pipe()
        if model is None: return ""
        
        print("🔍 Generating subtitles (Forced Aligner + Robust Mapping)...")
        # Step 1: Split original text into balanced segments
        user_segments = smart_balanced_split(text)
        
        # Step 2: Get word-level alignment from model
        results = model.transcribe(
            audio=audio_path,
            context=text,
            return_time_stamps=True
        )
        
        if not results or not results[0].time_stamps:
            print("⚠️ No timestamps generated.")
            return ""
        
        # Step 3: Align original segments to timestamps
        aligned = align_robust(user_segments, results[0].time_stamps)
        
        # Step 4: Format SRT
        srt_content = ""
        for i, ((start, end), segment_text) in enumerate(zip(aligned, user_segments)):
            srt_content += f"{i+1}\n{format_timestamp(start)} --> {format_timestamp(end)}\n{segment_text}\n\n"
        
        print("✅ SRT generated successfully")
        unload_aligner()
        return srt_content.strip()
    except Exception as e:
        print(f"❌ SRT Generation Error: {e}")
        unload_aligner()
        return f"SRT Error: {e}"

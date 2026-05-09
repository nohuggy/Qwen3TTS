import torch
import gc
import time
import soundfile as sf
import tempfile
import os
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["PYTHONWARNINGS"] = "ignore"
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
    """Load model with status yielding for UI"""
    global current_model, current_model_type
    
    if current_model_type == model_type:
        yield f"Using cached {model_type} model"
        yield current_model
        return

    if current_model is not None:
        yield f"Unloading {current_model_type} model..."
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
    
    yield f"Loading {model_type} model (1.7B) on {DEVICE}..."
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

        yield current_model
        return

    except Exception as e:
        msg = f"❌ Error loading model: {str(e)}"
        print(msg)
        import traceback
        traceback.print_exc()
        yield msg
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

def clean_script(text):
    """Purify script by reconstructing only the spoken dialogue, removing all technical tags"""
    if not text: return ""
    
    # 1. Parse into segments (same logic as voice_clone to ensure consistency)
    segments = re.findall(r"##\s*(.*?)\s*##\s*(.*?)(?=##|$)", text, re.DOTALL)
    
    if not segments:
        # Mono Mode: Use aggressive regex cleaning
        clean = re.sub(r"##.*?##", "", text, flags=re.DOTALL)
        clean = re.sub(r"\[pause:\d+\.?\d*\]", "", clean, flags=re.DOTALL)
        clean = re.sub(r"\[.*?\]", "", clean, flags=re.DOTALL)
        # Preserve line breaks for smart_balanced_split
        lines = [l.strip() for l in clean.split('\n') if l.strip()]
        return "\n".join(lines)

    # Multi Mode: Reconstruct from segments, preserving each line separately
    clean_parts = []
    for _, raw_content in segments:
        # A. Remove emotion tag from the start: [tag]
        content = re.sub(r"^\s*\[.*?\]", "", raw_content, flags=re.DOTALL)
        # B. Remove any internal pauses: [pause:X]
        content = re.sub(r"\[pause:\d+\.?\d*\]", "", content, flags=re.DOTALL)
        # C. Remove any other bracketed acting instructions
        content = re.sub(r"\[.*?\]", "", content, flags=re.DOTALL)
        
        if content.strip():
            # Keep each speaker's dialogue as its own line
            clean_parts.append(content.strip())
            
    # Join with newline to preserve sentence boundaries for the SRT splitter
    return "\n".join(clean_parts)

def voice_clone(text, role_bank_data, gen_srt=False, convert_punc=False, status_callback=None):
    """Generate speech using a Role Bank and script tags (## Name ##, [pause], [emotion])"""
    if not text or not role_bank_data:
        return None, ""
    
    if convert_punc:
        text = unify_punctuation(text)

    try:
        total_start = time.time()
        # Phase 0: Load Model
        model = None
        for status in load_model("base"):
            if isinstance(status, str):
                if status_callback: status_callback(status)
                yield status
            else:
                model = status
        
        if model is None:
            yield "❌ Model loading failed. Check console for details."
            return

        # Phase 1: Build Role Prompts
        role_prompts = {}
        first_prompt = None
        
        for role in role_bank_data:
            name = role.get('name', '').strip()
            audio = role.get('audio')
            ref_text = role.get('text', '').strip()
            
            if not audio: continue
            
            msg = f"Creating prompt for {name if name else 'Default'}..."
            if status_callback: status_callback(msg)
            yield msg
            
            if ref_text:
                prompt = model.create_voice_clone_prompt(ref_audio=audio, ref_text=ref_text, x_vector_only_mode=False)
            else:
                try:
                    prompt = model.create_voice_clone_prompt(ref_audio=audio, x_vector_only_mode=False)
                except:
                    prompt = model.create_voice_clone_prompt(ref_audio=audio, x_vector_only_mode=True)
            
            if name:
                role_prompts[name] = prompt
            if first_prompt is None:
                first_prompt = prompt

        if not first_prompt:
            yield "❌ No valid reference audio found."
            return

        # Phase 2: Parse Script and Generate
        # Pattern: ## Name ## Text
        segments = re.findall(r"##\s*(.*?)\s*##\s*(.*?)(?=##|$)", text, re.DOTALL)
        
        gen_start = time.time()
        all_wavs = []
        sr = 24000
        
        def process_text_part(part_text, current_prompt, current_instruction):
            """Helper to process a text segment with potential pauses"""
            part_wavs = []
            # Split by [pause:X]
            sub_parts = re.split(r"\[pause:(\d+\.?\d*)\]", part_text)
            for i, sub in enumerate(sub_parts):
                if i % 2 == 0: # Text
                    if sub.strip():
                        # We use generate_voice_clone. 
                        # Note: Qwen3-TTS usually takes 'instruct' in generate_voice_clone if supported, 
                        # otherwise we can prefix it to text or use the dedicated variant if needed.
                        # For base model cloning, 'instruct' is supported in recent versions.
                        w, _ = model.generate_voice_clone(
                            text=sub.strip(),
                            voice_clone_prompt=current_prompt,
                            instruct=current_instruction if current_instruction != "Standard" else None
                        )
                        wav = w[0]
                        if hasattr(wav, 'cpu'): wav = wav.cpu()
                        if not isinstance(wav, torch.Tensor): wav = torch.from_numpy(wav)
                        part_wavs.append(wav)
                else: # Pause
                    silence_sec = float(sub)
                    silence = torch.zeros(int(sr * silence_sec))
                    part_wavs.append(silence)
            return part_wavs

        if not segments:
            # Mono Mode
            print("Mode: Mono-character")
            all_wavs.extend(process_text_part(text, first_prompt, "Standard"))
        else:
            # Multi Mode
            print(f"Mode: Multi-speaker ({len(segments)} segments)")
            for i, (speaker, raw_content) in enumerate(segments):
                name = speaker.strip()
                content = raw_content.strip()
                prompt = role_prompts.get(name, first_prompt)
                
                # Extract Emotion [tag]
                emotion_match = re.match(r"^\s*\[(.*?)\]\s*(.*)", content, re.DOTALL)
                if emotion_match:
                    instr = emotion_match.group(1)
                    actual_text = emotion_match.group(2)
                else:
                    instr = "Standard"
                    actual_text = content
                
                msg = f"Generating chunk {i+1}/{len(segments)}..."
                if status_callback: status_callback(msg)
                yield msg
                
                all_wavs.extend(process_text_part(actual_text, prompt, instr))
                
                # Small gap between speakers if not the last one
                if i < len(segments) - 1:
                    all_wavs.append(torch.zeros(int(sr * 0.3)))

        if not all_wavs:
            yield "❌ No audio generated."
            return

        final_wav = torch.cat(all_wavs, dim=-1)
        gen_time = time.time() - gen_start

        # Save and Cleanup
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        audio_data = final_wav.numpy()
        sf.write(temp_file.name, audio_data, sr)

        total_time = time.time() - total_start
        audio_duration = len(final_wav) / sr
        
        # Word count logic from previous task
        # ... (Already handled in app_colab.py)

        print(f"✅ Done! Total: {total_time:.1f}s | Gen: {gen_time:.1f}s | Audio: {audio_duration:.1f}s")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        del all_wavs
        gc.collect()
        time.sleep(0.5)

        srt_content = ""
        if gen_srt:
            # Clean text for SRT generation (filter out names, emotions, pauses)
            clean_text = clean_script(text)
            srt_content = yield from generate_srt(clean_text, temp_file.name, total_start_time=total_start)

        yield (temp_file.name, srt_content)

    except Exception as e:
        msg = f"❌ Error in voice_clone: {str(e)}"
        import traceback
        traceback.print_exc()
        print(msg)
        yield msg

def custom_voice(text, voice_name, instruction, gen_srt=False, convert_punc=False, status_callback=None):
    """Generate speech using preset voices with chunk awareness"""
    if not text:
        return None, ""

    if convert_punc:
        text = unify_punctuation(text)

    try:
        total_start = time.time()
        # Phase 0: Load Model
        model = None
        for status in load_model("custom"):
            if isinstance(status, str):
                if status_callback: status_callback(status)
                yield status
            else:
                model = status
        
        if model is None:
            yield "❌ Model loading failed. Check console for details."
            return

        yield f"Generating with voice: {voice_name}..."
        gen_start = time.time()

        with torch.inference_mode():
            # Chunk text by paragraph
            paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
            all_wavs = []
            sr = 24000
            
            for i, p in enumerate(paragraphs):
                msg = f"Generating chunk {i+1}/{len(paragraphs)}..."
                if status_callback: status_callback(msg)
                yield msg
                print(f"   {msg}")
                
                if instruction and instruction.strip():
                    wavs, sr = model.generate_custom_voice(
                        text=p,
                        speaker=voice_name,
                        instruct=instruction
                    )
                else:
                    wavs, sr = model.generate_custom_voice(
                        text=p,
                        speaker=voice_name
                    )
                wav = wavs[0]
                if hasattr(wav, 'cpu'):
                    wav = wav.cpu()
                if not isinstance(wav, torch.Tensor):
                    wav = torch.from_numpy(wav)
                all_wavs.append(wav)
                
            # Concatenate chunks
            final_wav = torch.cat(all_wavs, dim=-1)
            gen_time = time.time() - gen_start

        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        # Ensure audio is on CPU and in numpy format for soundfile
        audio_data = final_wav.numpy()
        sf.write(temp_file.name, audio_data, sr)

        total_time = time.time() - total_start
        audio_duration = len(final_wav) / sr
        rtf = gen_time / audio_duration

        print(f"✅ Done! Total: {total_time:.1f}s | Gen: {gen_time:.1f}s | Audio: {audio_duration:.1f}s | RTF: {rtf:.2f}x")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Explicitly clear audio tensors to free VRAM before alignment
        del all_wavs
        gc.collect()
        time.sleep(1) # Breathe

        srt_content = ""
        if gen_srt:
            srt_content = generate_srt(text, temp_file.name)

        yield (temp_file.name, srt_content)

    except Exception as e:
        msg = f"❌ Error in custom_voice: {str(e)}"
        print(msg)
        yield msg

def voice_design(text, voice_description, gen_srt=False, convert_punc=False, status_callback=None):
    """Generate speech from text description with chunk awareness"""
    if not text or not voice_description:
        return None, ""

    if convert_punc:
        text = unify_punctuation(text)

    try:
        total_start = time.time()
        # Phase 0: Load Model
        model = None
        for status in load_model("design"):
            if isinstance(status, str):
                if status_callback: status_callback(status)
                yield status
            else:
                model = status
        
        if model is None:
            yield "❌ Model loading failed. Check console for details."
            return

        yield "Generating voice design..."
        gen_start = time.time()

        with torch.inference_mode():
            # Chunking
            paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
            all_wavs = []
            sr = 24000
            
            for i, p in enumerate(paragraphs):
                msg = f"Generating chunk {i+1}/{len(paragraphs)}..."
                if status_callback: status_callback(msg)
                yield msg
                print(f"   {msg}")
                
                wavs, sr = model.generate_voice_design(
                    text=p,
                    instruct=voice_description
                )
                wav = wavs[0]
                if hasattr(wav, 'cpu'):
                    wav = wav.cpu()
                if not isinstance(wav, torch.Tensor):
                    wav = torch.from_numpy(wav)
                all_wavs.append(wav)
                
            final_wav = torch.cat(all_wavs, dim=-1)
            gen_time = time.time() - gen_start

        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        audio_data = final_wav.numpy()
        sf.write(temp_file.name, audio_data, sr)

        total_time = time.time() - total_start
        audio_duration = len(final_wav) / sr
        rtf = gen_time / audio_duration

        print(f"✅ Done! Total: {total_time:.1f}s | Gen: {gen_time:.1f}s | Audio: {audio_duration:.1f}s | RTF: {rtf:.2f}x")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        del all_wavs
        gc.collect()
        time.sleep(0.5)

        yield (temp_file.name, "")

    except Exception as e:
        msg = f"❌ Error in voice_design: {str(e)}"
        print(msg)
        yield msg

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

import concurrent.futures

def generate_srt(text, audio_path, total_start_time=None):
    """Generate SRT subtitles using Forced Aligner and Robust Splitter (Generator)"""
    if not text or not audio_path:
        yield ""
        return
    try:
        def get_msg(m):
            if total_start_time:
                return f"{m} [Elapsed: {time.time() - total_start_time:.1f}s]"
            return m

        yield get_msg("Loading Aligner Engine...")
        def run_load():
            return get_aligner_pipe()
            
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(run_load)
            while not future.done():
                yield get_msg("Loading Aligner Engine...")
                time.sleep(1)
            model = future.result()
            
        yield get_msg("✅ Engine loaded. Starting alignment...")
        if model is None: 
            yield "❌ Aligner Engine failed to load."
            return
        
        yield get_msg("🔍 Aligning subtitles...")
        # Step 1: Split original text into balanced segments
        user_segments = smart_balanced_split(text)
        
        # Step 2: Get word-level alignment from model (Threaded to keep UI timer alive)
        def run_transcribe():
            return model.transcribe(
                audio=audio_path,
                context=text,
                return_time_stamps=True
            )
            
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(run_transcribe)
            while not future.done():
                yield get_msg("🔍 Aligning subtitles...")
                time.sleep(1)
            results = future.result()
        
        if not results or not results[0].time_stamps:
            print("⚠️ No timestamps generated.")
            yield ""
            return
        
        yield get_msg("📝 Formatting SRT...")
        # Step 3: Align original segments to timestamps
        aligned = align_robust(user_segments, results[0].time_stamps)
        
        # Step 4: Format SRT
        srt_content = ""
        for i, ((start, end), segment_text) in enumerate(zip(aligned, user_segments)):
            srt_content += f"{i+1}\n{format_timestamp(start)} --> {format_timestamp(end)}\n{segment_text}\n\n"
        
        print("✅ SRT generated successfully")
        unload_aligner()
        yield srt_content.strip()
    except Exception as e:
        print(f"❌ SRT Generation Error: {e}")
        unload_aligner()
        yield f"SRT Error: {e}"

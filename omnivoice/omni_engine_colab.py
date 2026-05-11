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
import ctypes
import random
import numpy as np

def trim_memory():
    """Force OS to reclaim memory. Crucial for Colab/Linux environments."""
    gc.collect()
    gc.collect()
    try:
        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
    except Exception:
        pass
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def wav_to_mp3(wav_path):
    """Convert WAV to MP3 for preview stability. Returns MP3 path."""
    if not wav_path or not os.path.exists(wav_path):
        return wav_path
    try:
        mp3_path = wav_path.replace(".wav", ".mp3")
        # Colab usually has ffmpeg. Use it to convert.
        import subprocess
        # -qscale:a 2 is a good variable bitrate for MP3 (~190kbps)
        subprocess.run(["ffmpeg", "-y", "-i", wav_path, "-codec:a", "libmp3lame", "-qscale:a", "2", mp3_path], 
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return mp3_path
    except Exception as e:
        print(f"⚠️ MP3 Conversion failed: {e}. Falling back to WAV.")
        return wav_path

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

# setup_asr() # No longer run at boot as requested

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
        trim_memory()
        time.sleep(2) # Breathe room for RAM reclamation

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
            attn_implementation="sdpa" if torch.cuda.is_available() else "eager",
            low_cpu_mem_usage=True
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

def load_qwen3tts(filepath):
    """Load a pre-compiled .qwen3tts voice prompt from disk.
    
    The .qwen3tts file is a torch.save() of the voice_clone_prompt object
    produced by model.create_voice_clone_prompt(). This allows skipping
    the expensive re-processing of reference audio each session.
    
    Returns:
        The voice_clone_prompt object, or None on failure.
    """
    if not filepath or not os.path.exists(filepath):
        return None
    try:
        prompt = torch.load(filepath, map_location="cpu", weights_only=False)
        print(f"✅ Loaded .qwen3tts: {os.path.basename(filepath)}")
        return prompt
    except Exception as e:
        print(f"❌ Failed to load .qwen3tts '{filepath}': {e}")
        return None


def compile_role(audio_path, transcript, role_name, status_callback=None):
    """Compile a reference audio + transcript into a portable .qwen3tts file.
    
    This replicates the 'Qwen3-TTS Prompt Manager' node from ComfyUI:
    the voice embedding is extracted once and saved as a tensor file
    that can be re-loaded instantly without repeating the reference audio step.
    
    Args:
        audio_path: Path to reference audio file.
        transcript: Transcript of the reference audio (improves quality).
        role_name: Name used for the output filename (e.g. 'Natasha').
        status_callback: Optional callable for status updates.
    
    Returns:
        (filepath, status_message) — filepath is None on failure.
    """
    if not audio_path:
        return None, "❌ Error: Reference audio is required."
    
    name = role_name.strip() if role_name and role_name.strip() else "unnamed"
    
    def _update(msg):
        print(msg)
        if status_callback:
            status_callback(msg)
    
    try:
        _update("Loading TTS model for compilation...")
        model = None
        for result in load_model("base"):
            if isinstance(result, str):
                _update(result)
            else:
                model = result
        
        if model is None:
            return None, "❌ Model loading failed."
        
        _update(f"Extracting voice timbre for '{name}'...")
        if transcript and transcript.strip():
            prompt = model.create_voice_clone_prompt(
                ref_audio=audio_path,
                ref_text=transcript.strip(),
                x_vector_only_mode=False
            )
        else:
            try:
                prompt = model.create_voice_clone_prompt(
                    ref_audio=audio_path,
                    x_vector_only_mode=False
                )
            except Exception:
                prompt = model.create_voice_clone_prompt(
                    ref_audio=audio_path,
                    x_vector_only_mode=True
                )
        
        # Save to outputs/ directory so it's accessible in Colab
        os.makedirs("outputs/roles", exist_ok=True)
        filename = f"{name}.qwen3tts"
        filepath = os.path.join("outputs/roles", filename)
        
        torch.save(prompt, filepath)
        msg = f"✅ Compiled '{filename}' successfully! ({os.path.getsize(filepath) / 1024:.1f} KB)"
        _update(msg)
        return filepath, msg
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, f"❌ Compilation failed: {str(e)}"


def voice_clone(text, role_bank_data, gen_srt=False, convert_punc=False,
                temperature=1.0, top_p=1.0, top_k=50, repetition_penalty=1.1,
                seed=42, status_callback=None):
    """Generate speech using a Role Bank and script tags (## Name ##, [pause], [emotion])"""
    # Set seeds
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

    # Log advanced params
    print(f"🎛️ Advanced params — temp={temperature}, top_p={top_p}, top_k={top_k}, rep_pen={repetition_penalty}, seed={seed}")
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
            qwen3tts_path = role.get('qwen3tts_path')  # Pre-compiled voice file
            
            prompt = None
            
            # --- Path A: Load from pre-compiled .qwen3tts file ---
            if qwen3tts_path:
                msg = f"Loading pre-compiled voice for '{name if name else 'Default'}'..."
                if status_callback: status_callback(msg)
                yield msg
                prompt = load_qwen3tts(qwen3tts_path)
                if prompt is None:
                    msg = f"⚠️ Could not load '{qwen3tts_path}', falling back to reference audio..."
                    if status_callback: status_callback(msg)
                    yield msg
            
            # --- Path B: Create from reference audio (original behaviour) ---
            if prompt is None:
                if not audio: continue
                
                msg = f"Creating prompt for {name if name else 'Default'}..."
                if status_callback: status_callback(msg)
                yield msg
                
                if ref_text:
                    prompt = model.create_voice_clone_prompt(ref_audio=audio, ref_text=ref_text, x_vector_only_mode=False)
                else:
                    try:
                        prompt = model.create_voice_clone_prompt(ref_audio=audio, x_vector_only_mode=False)
                    except Exception:
                        prompt = model.create_voice_clone_prompt(ref_audio=audio, x_vector_only_mode=True)
            
            if prompt is not None:
                if name:
                    role_prompts[name] = prompt
                if first_prompt is None:
                    first_prompt = prompt

        if not first_prompt:
            yield "❌ No valid reference audio or .qwen3tts found."
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
                        instr_arg = current_instruction if current_instruction != "Standard" else None
                        # Build kwargs; try advanced params first, fall back gracefully
                        try:
                            w, _ = model.generate_voice_clone(
                                text=sub.strip(),
                                voice_clone_prompt=current_prompt,
                                instruct=instr_arg,
                                temperature=temperature,
                                top_p=top_p,
                                top_k=top_k,
                                repetition_penalty=repetition_penalty,
                                subtalker_temperature=0.8, # Internal default
                            )
                        except TypeError:
                            # Older qwen_tts without advanced params — fall back silently
                            w, _ = model.generate_voice_clone(
                                text=sub.strip(),
                                voice_clone_prompt=current_prompt,
                                instruct=instr_arg,
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
                
                # Extract ALL leading instruct/emotion tags: [Confident] [Professional] etc.
                # Collects every consecutive [tag] at the start, joins as comma list for instruct param.
                leading_tags = re.findall(r"^\s*(?:\[([^\]]+)\]\s*)+", content)
                if leading_tags:
                    # Re-parse individually to get each tag separately
                    all_leading = re.findall(r"\[([^\]]+)\]", re.match(r"^(\s*\[[^\]]+\]\s*)+", content).group(0))
                    instr = ", ".join(tag.strip() for tag in all_leading)
                    # Strip all leading [tag] blocks from spoken text
                    actual_text = re.sub(r"^(\s*\[[^\]]+\]\s*)+", "", content).strip()
                    print(f"   🎭 Speaker '{name}' | instruct='{instr}'")
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

def custom_voice(text, voice_name, instruction, gen_srt=False, convert_punc=False,
                 temperature=1.0, top_p=1.0, top_k=50, repetition_penalty=1.1,
                 seed=42, status_callback=None):
    """Generate speech using preset voices with chunk awareness"""
    # Set seeds
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

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
                
                instr_arg = instruction if (instruction and instruction.strip() and instruction.strip() != "Standard") else None
                try:
                    wavs, sr = model.generate_custom_voice(
                        text=p,
                        speaker=voice_name,
                        instruct=instr_arg,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        repetition_penalty=repetition_penalty,
                        subtalker_temperature=0.8,
                    )
                except TypeError:
                    wavs, sr = model.generate_custom_voice(
                        text=p,
                        speaker=voice_name,
                        instruct=instr_arg,
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

def voice_design(text, voice_description, gen_srt=False, convert_punc=False,
                 temperature=1.0, top_p=1.0, top_k=50, repetition_penalty=1.1,
                 seed=42, status_callback=None, gen_qwen3tts=False):
    """Generate speech from text description with chunk awareness and multi-speaker support"""
    # Set seeds
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

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
        
        # Parse descriptions for multi-character
        desc_segments = re.findall(r"##\s*(.*?)\s*##\s*(.*?)(?=##|$)", voice_description, re.DOTALL)
        if desc_segments:
            role_instructions = {name.strip(): instr.strip() for name, instr in desc_segments}
        else:
            role_instructions = {"default": voice_description.strip()}

        # Parse text into segments handling ## Name ##
        segments = []
        current_speaker = "default" if not desc_segments else list(role_instructions.keys())[0]
        current_text = []

        for line in text.split('\n'):
            line = line.strip()
            if not line: continue
            
            m1 = re.match(r"^##\s*(.*?)\s*##(.*)", line)
            if m1:
                if current_text:
                    segments.append((current_speaker, '\n'.join(current_text)))
                current_speaker = m1.group(1).strip()
                current_text = [m1.group(2).strip()] if m1.group(2).strip() else []
                continue
                
            current_text.append(line)

        if current_text:
            segments.append((current_speaker, '\n'.join(current_text)))

        with torch.inference_mode():
            all_wavs = []
            sr = 24000
            
            speaker_audio = {}
            speaker_text = {}
            
            for i, (speaker, content) in enumerate(segments):
                if not content.strip(): continue
                msg = f"Generating chunk {i+1}/{len(segments)} ({speaker})..."
                if status_callback: status_callback(msg)
                yield msg
                print(f"   {msg}")
                
                instr = role_instructions.get(speaker, role_instructions.get("default", ""))
                
                paragraphs = [p.strip() for p in content.split('\n') if p.strip()]
                speaker_wavs = []
                for p in paragraphs:
                    try:
                        wavs, sr = model.generate_voice_design(
                            text=p,
                            instruct=instr,
                            temperature=temperature,
                            top_p=top_p,
                            top_k=top_k,
                            repetition_penalty=repetition_penalty,
                            subtalker_temperature=0.8,
                        )
                    except TypeError:
                        wavs, sr = model.generate_voice_design(
                            text=p,
                            instruct=instr
                        )
                    wav = wavs[0]
                    if hasattr(wav, 'cpu'): wav = wav.cpu()
                    if not isinstance(wav, torch.Tensor): wav = torch.from_numpy(wav)
                    speaker_wavs.append(wav)
                    all_wavs.append(wav)
                    
                if speaker not in speaker_audio:
                    speaker_audio[speaker] = []
                    speaker_text[speaker] = []
                speaker_audio[speaker].extend(speaker_wavs)
                speaker_text[speaker].append(content)
                
                if i < len(segments) - 1:
                    all_wavs.append(torch.zeros(int(sr * 0.3)))
                
            final_wav = torch.cat(all_wavs, dim=-1)
            gen_time = time.time() - gen_start

        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        audio_data = final_wav.numpy()
        sf.write(temp_file.name, audio_data, sr)
        
        # Save per-speaker temp wavs for potential qwen3tts compilation
        speaker_paths = {}
        for spk, spk_wavs in speaker_audio.items():
            spk_wav = torch.cat(spk_wavs, dim=-1)
            spk_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            sf.write(spk_file.name, spk_wav.numpy(), sr)
            speaker_paths[spk] = (spk_file.name, "\n".join(speaker_text[spk]))

        total_time = time.time() - total_start
        audio_duration = len(final_wav) / sr
        rtf = gen_time / audio_duration

        print(f"✅ Done! Total: {total_time:.1f}s | Gen: {gen_time:.1f}s | Audio: {audio_duration:.1f}s | RTF: {rtf:.2f}x")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        del all_wavs, speaker_audio
        gc.collect()
        time.sleep(0.5)

        srt_content = ""
        if gen_srt:
            srt_content = yield from generate_srt(text, temp_file.name)
            
        qwen3tts_files = []
        if gen_qwen3tts:
            yield "✅ Compiling .qwen3tts voices..."
            # Note: compiling will load base TTS model
            for spk, (spk_path, spk_txt) in speaker_paths.items():
                spk_name = get_slug(spk) if spk != "default" else get_slug(text)
                qwen_path, _ = compile_role(spk_path, spk_txt, spk_name, status_callback)
                if qwen_path:
                    qwen3tts_files.append(qwen_path)

        yield (temp_file.name, srt_content, qwen3tts_files)

    except Exception as e:
        msg = f"❌ Error in voice_design: {str(e)}"
        print(msg)
        yield msg

def get_asr_pipe():
    """Load Qwen3-ASR-0.6B model"""
    global ASR_PIPE
    if ASR_PIPE is None:
        setup_asr() # Install/Download on first use
        print("Loading Qwen3-ASR-0.6B model...")
        # Unload TTS model first to save VRAM
        global current_model, current_model_type
        if current_model is not None:
            print(f"Unloading {current_model_type} model before ASR...")
            del current_model
            current_model = None
            current_model_type = None
            trim_memory()
            time.sleep(1) # Breathe
        
        try:
            from qwen_asr import Qwen3ASRModel
            ASR_PIPE = Qwen3ASRModel.from_pretrained(
                "Qwen/Qwen3-ASR-0.6B",
                dtype=DTYPE,
                device_map=DEVICE,
                attn_implementation="sdpa" if torch.cuda.is_available() else "eager",
                low_cpu_mem_usage=True
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
                    trust_remote_code=True,
                    model_kwargs={"low_cpu_mem_usage": True}
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
        trim_memory()

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
        setup_asr() # Install/Download on first use
        print("Loading Qwen3-ForcedAligner-0.6B model...")
        # Unload TTS and ASR first to save VRAM
        global current_model, current_model_type
        if current_model is not None:
            print(f"Unloading {current_model_type} model before Aligner...")
            del current_model
            current_model = None
            current_model_type = None
            trim_memory()
            time.sleep(2) # Extra breathing room for OS
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
                attn_implementation="sdpa" if torch.cuda.is_available() else "eager",
                low_cpu_mem_usage=True
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
        trim_memory()

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
        
        # Crucial: Clear local references before calling global unload
        # Otherwise the model object stays in memory until this generator finishes
        del model
        if 'results' in locals():
            del results
            
        unload_aligner()
        yield srt_content.strip()
    except Exception as e:
        print(f"❌ SRT Generation Error: {e}")
        if 'model' in locals(): del model
        unload_aligner()
        yield f"SRT Error: {e}"

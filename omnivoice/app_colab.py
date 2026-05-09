import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["PYTHONWARNINGS"] = "ignore"
import gradio as gr
from omnivoice.omni_engine_colab import voice_clone, custom_voice, voice_design, transcribe_ref, compile_role
import time
import re

def count_words(text):
    """Smart word count: CJK characters count as 1, Latin words count as 1"""
    if not text: return 0
    # Count CJK characters (Chinese, Japanese, Korean)
    cjk_count = len(re.findall(r'[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]', text))
    # Count non-CJK words (sequences of Latin letters/numbers)
    latin_words = len(re.findall(r'[a-zA-Z0-9\']+', text))
    return cjk_count + latin_words

# Minimal CSS
custom_css = """
.gr-button-primary {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    border: none !important;
}
.status-msg {
    font-size: 0.9em;
    color: #a0aec0;
    margin-bottom: 8px;
    font-family: monospace;
}
"""

def package_zip(text, audio_path, srt_content):
    """Package audio and SRT into a ZIP for download using slug-based naming"""
    if not audio_path: return None
    import zipfile
    from omnivoice.omni_engine_colab import get_slug
    
    slug = get_slug(text)
    # Ensure outputs directory exists
    os.makedirs("outputs", exist_ok=True)
    # Create the ZIP in the outputs folder with the slug as the filename
    zip_path = os.path.join("outputs", f"{slug}.zip")
    
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        # Save as slug-based filename inside the ZIP
        zipf.write(audio_path, f"{slug}.wav")
        if srt_content:
            # Create a temporary SRT with the slug name for the ZIP
            base = os.path.splitext(audio_path)[0]
            srt_path = f"{base}.srt"
            with open(srt_path, 'w', encoding='utf-8') as f:
                f.write(srt_content)
            zipf.write(srt_path, f"{slug}.srt")
    return zip_path

def create_app():
    with gr.Blocks(title="Qwen3-TTS", css=custom_css) as demo:
        # Main Title
        gr.Markdown("# Qwen3-TTS")
        gr.Markdown("Advanced Text-to-Speech AI | Voice Cloning, Custom Voice & Voice Design")

        with gr.Tab("Voice Cloning"):
            # Utility functions for roles
            def on_transcribe(audio):
                if not audio: return ""
                return transcribe_ref(audio)

            def process_ref_zip(zip_file):
                if not zip_file: return None, ""
                import zipfile, shutil, uuid
                out_dir = os.path.abspath("outputs/refs")
                os.makedirs(out_dir, exist_ok=True)
                audio_path = None
                text_content = ""
                
                with zipfile.ZipFile(zip_file.name, 'r') as z:
                    for f in z.namelist():
                        if f.endswith('/') or f.startswith('__MACOSX') or os.path.basename(f).startswith('.'):
                            continue
                            
                        # Unique prefix to avoid collisions and Gradio cache issues
                        unique_prefix = str(uuid.uuid4())[:8]
                        
                        if f.lower().endswith(('.wav', '.mp3', '.flac')):
                            ext = os.path.splitext(f)[1]
                            dest_name = f"{unique_prefix}_{os.path.basename(f)}"
                            dest_path = os.path.join(out_dir, dest_name)
                            
                            # Extract and move
                            with z.open(f) as source, open(dest_path, "wb") as target:
                                shutil.copyfileobj(source, target)
                            
                            if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
                                audio_path = dest_path
                                
                        if f.lower().endswith('.txt'):
                            with z.open(f) as tf:
                                try:
                                    text_content = tf.read().decode('utf-8')
                                except:
                                    try:
                                        text_content = tf.read().decode('gbk', errors='ignore')
                                    except:
                                        text_content = ""
                
                return audio_path, text_content

            def process_ref_txt(txt_file):
                if not txt_file: return ""
                try:
                    with open(txt_file.name, 'r', encoding='utf-8') as tf:
                        return tf.read()
                except UnicodeDecodeError:
                    with open(txt_file.name, 'r', encoding='gbk') as tf:
                        return tf.read()

            def on_clone(text, 
                         name1, audio1, text1, qwen3ts1,
                         name2, audio2, text2, qwen3ts2,
                         name3, audio3, text3, qwen3ts3,
                         gen_srt, conv_punc,
                         temperature, top_p, repetition_penalty, subtalker_temperature):
                import time
                start_time = time.time()
                # Clear previous outputs immediately
                yield None, "", gr.update(visible=False), "Initializing..."
                
                # Build role bank data — include qwen3tts_path when a compiled file is uploaded
                role_bank_data = []
                if audio1 or qwen3ts1:
                    role_bank_data.append({'name': name1, 'audio': audio1, 'text': text1,
                                           'qwen3tts_path': qwen3ts1})
                if audio2 or qwen3ts2:
                    role_bank_data.append({'name': name2, 'audio': audio2, 'text': text2,
                                           'qwen3tts_path': qwen3ts2})
                if audio3 or qwen3ts3:
                    role_bank_data.append({'name': name3, 'audio': audio3, 'text': text3,
                                           'qwen3tts_path': qwen3ts3})
                
                tts_start = time.time()
                audio_path = None
                last_status = ""
                for status in voice_clone(text, role_bank_data, gen_srt=False, convert_punc=conv_punc,
                                          temperature=temperature, top_p=top_p,
                                          repetition_penalty=repetition_penalty,
                                          subtalker_temperature=subtalker_temperature,
                                          status_callback=lambda m: print(f"UI: {m}")):
                    if isinstance(status, str):
                        last_status = status
                        yield None, "", gr.update(visible=False), status
                    else:
                        audio_path, _ = status
                
                tts_dur = time.time() - tts_start
                
                if not audio_path:
                    if not last_status.startswith("Error") and not last_status.startswith("❌"):
                        yield None, gr.update(), gr.update(visible=False), "Generation failed."
                    return
                
                yield audio_path, gr.update(), gr.update(visible=False), "Audio ready. Aligning subtitles..."
                
                # Phase 2: Subtitles (Memory Intensive)
                asr_start = time.time()
                srt = ""
                if gen_srt:
                    from omnivoice.omni_engine_colab import generate_srt, clean_script
                    clean_text = clean_script(text)
                    for status in generate_srt(clean_text, audio_path, total_start_time=start_time):
                        if isinstance(status, str) and not status.startswith("1\n"): # Check if it's status or SRT
                            yield audio_path, gr.update(), gr.update(visible=False), status
                        else:
                            srt = status
                asr_dur = time.time() - asr_start
                
                # Performance metrics
                total_dur = time.time() - start_time
                word_count = count_words(text)
                perf_msg = f"Done! Total: {total_dur:.1f}s | Gen: {tts_dur:.1f}s | Asr: {asr_dur:.1f}s | Words: {word_count}"
                
                zip_path = package_zip(text, audio_path, srt)
                yield audio_path, srt, gr.update(value=zip_path, visible=True), perf_msg

            with gr.Row():
                with gr.Column():
                    input_text = gr.Textbox(
                        label="Input Text", 
                        placeholder="## Alex ## [shout] Hello! [pause:0.5] How are you?\n## Sara ## [happy] I'm fine!", 
                        lines=6
                    )
                    
                    with gr.Accordion("Role 1 (Primary)", open=True):
                        r1_name = gr.Textbox(label="Role Name", placeholder="e.g. Alex", value="")
                        r1_audio = gr.Audio(label="Reference Audio", type="filepath")
                        r1_text = gr.Textbox(label="Reference Transcript", placeholder="Text from the audio...")
                        with gr.Row():
                            r1_trans_btn = gr.Button("Trans Ref", variant="primary", size="sm")
                            r1_zip_btn = gr.UploadButton("Ref Zip", file_types=[".zip"], variant="primary", size="sm")
                            r1_txt_btn = gr.UploadButton("Ref Txt", file_types=[".txt"], variant="primary", size="sm")
                            r1_clear_btn = gr.Button("Clear", variant="secondary", size="sm")
                        r1_qwen3tts = gr.File(label="Upload .qwen3tts (skips ref audio)",
                                              file_types=[".qwen3tts"], type="filepath", visible=True)
                    
                    with gr.Accordion("Role 2", open=False):
                        r2_name = gr.Textbox(label="Role Name", placeholder="e.g. Sara")
                        r2_audio = gr.Audio(label="Reference Audio", type="filepath")
                        r2_text = gr.Textbox(label="Reference Transcript", placeholder="Text from the audio...")
                        with gr.Row():
                            r2_trans_btn = gr.Button("Trans Ref", variant="primary", size="sm")
                            r2_zip_btn = gr.UploadButton("Ref Zip", file_types=[".zip"], variant="primary", size="sm")
                            r2_txt_btn = gr.UploadButton("Ref Txt", file_types=[".txt"], variant="primary", size="sm")
                            r2_clear_btn = gr.Button("Clear", variant="secondary", size="sm")
                        r2_qwen3tts = gr.File(label="Upload .qwen3tts (skips ref audio)",
                                              file_types=[".qwen3tts"], type="filepath", visible=True)
                        
                    with gr.Accordion("Role 3", open=False):
                        r3_name = gr.Textbox(label="Role Name", placeholder="e.g. Bob")
                        r3_audio = gr.Audio(label="Reference Audio", type="filepath")
                        r3_text = gr.Textbox(label="Reference Transcript", placeholder="Text from the audio...")
                        with gr.Row():
                            r3_trans_btn = gr.Button("Trans Ref", variant="primary", size="sm")
                            r3_zip_btn = gr.UploadButton("Ref Zip", file_types=[".zip"], variant="primary", size="sm")
                            r3_txt_btn = gr.UploadButton("Ref Txt", file_types=[".txt"], variant="primary", size="sm")
                            r3_clear_btn = gr.Button("Clear", variant="secondary", size="sm")
                        r3_qwen3tts = gr.File(label="Upload .qwen3tts (skips ref audio)",
                                              file_types=[".qwen3tts"], type="filepath", visible=True)

                with gr.Column():
                    with gr.Group():
                        audio_out = gr.Audio(label="Generated Speech")
                    with gr.Group():
                        srt_out = gr.Textbox(label="SRT Preview", lines=6, interactive=False)
                    with gr.Group():
                        status_out = gr.Textbox(label="Status", value="", interactive=False, lines=2)
                    with gr.Row():
                        gen_srt = gr.Checkbox(label="Generate Subtitles", value=True)
                        conv_punc = gr.Checkbox(label="Smart Punctuation", value=True)
                    with gr.Accordion("Advanced TTS Settings", open=False):
                        with gr.Row():
                            temperature = gr.Slider(minimum=0.0, maximum=1.5, value=0.8, step=0.05,
                                                    label="Temperature", info="Controls randomness. Blog rec: 0.8")
                            top_p = gr.Slider(minimum=0.0, maximum=1.0, value=0.9, step=0.05,
                                              label="Top P", info="Nucleus sampling threshold. Blog rec: 0.9")
                        with gr.Row():
                            repetition_penalty = gr.Slider(minimum=1.0, maximum=2.0, value=1.1, step=0.05,
                                                           label="Repetition Penalty", info="Reduces repetition. Blog rec: 1.1")
                            subtalker_temperature = gr.Slider(minimum=0.0, maximum=1.5, value=0.8, step=0.05,
                                                              label="Subtalker Temperature", info="For secondary voice tokens")
                    btn = gr.Button("Generate Audio", variant="primary", size="lg")
                    zip_out = gr.DownloadButton("Download ZIP (WAV + SRT)", visible=False)

            # --- Callbacks ---
            # Role 1
            r1_trans_btn.click(on_transcribe, inputs=[r1_audio], outputs=[r1_text])
            r1_zip_btn.upload(process_ref_zip, inputs=[r1_zip_btn], outputs=[r1_audio, r1_text])
            r1_txt_btn.upload(process_ref_txt, inputs=[r1_txt_btn], outputs=[r1_text])
            r1_clear_btn.click(lambda: (None, None, "", None), outputs=[r1_name, r1_audio, r1_text, r1_qwen3tts])
            # Auto-fill role name from .qwen3tts filename
            r1_qwen3tts.upload(lambda f: os.path.splitext(os.path.basename(f))[0] if f else gr.update(),
                               inputs=[r1_qwen3tts], outputs=[r1_name])
            
            # Role 2
            r2_trans_btn.click(on_transcribe, inputs=[r2_audio], outputs=[r2_text])
            r2_zip_btn.upload(process_ref_zip, inputs=[r2_zip_btn], outputs=[r2_audio, r2_text])
            r2_txt_btn.upload(process_ref_txt, inputs=[r2_txt_btn], outputs=[r2_text])
            r2_clear_btn.click(lambda: (None, None, "", None), outputs=[r2_name, r2_audio, r2_text, r2_qwen3tts])
            r2_qwen3tts.upload(lambda f: os.path.splitext(os.path.basename(f))[0] if f else gr.update(),
                               inputs=[r2_qwen3tts], outputs=[r2_name])
            
            # Role 3
            r3_trans_btn.click(on_transcribe, inputs=[r3_audio], outputs=[r3_text])
            r3_zip_btn.upload(process_ref_zip, inputs=[r3_zip_btn], outputs=[r3_audio, r3_text])
            r3_txt_btn.upload(process_ref_txt, inputs=[r3_txt_btn], outputs=[r3_text])
            r3_clear_btn.click(lambda: (None, None, "", None), outputs=[r3_name, r3_audio, r3_text, r3_qwen3tts])
            r3_qwen3tts.upload(lambda f: os.path.splitext(os.path.basename(f))[0] if f else gr.update(),
                               inputs=[r3_qwen3tts], outputs=[r3_name])

            btn.click(
                on_clone,
                inputs=[input_text,
                        r1_name, r1_audio, r1_text, r1_qwen3tts,
                        r2_name, r2_audio, r2_text, r2_qwen3tts,
                        r3_name, r3_audio, r3_text, r3_qwen3tts,
                        gen_srt, conv_punc, temperature, top_p, repetition_penalty, subtalker_temperature],
                outputs=[audio_out, srt_out, zip_out, status_out]
            )

        with gr.Tab("Custom Voice"):
            with gr.Row():
                with gr.Column():
                    custom_text = gr.Textbox(label="Input Text", placeholder="Enter text...", lines=6)
                    custom_name = gr.Dropdown(
                        label="Character Voice", 
                        choices=["amanda", "denis", "jessica", "kevin", "lewis", "pippa", "stella", "tess", "vivienne"],
                        value="amanda"
                    )
                    custom_instr = gr.Textbox(label="Instruction", placeholder="e.g. happy, sad, whispered, shouting...", value="Standard")
                    with gr.Row():
                        custom_gen_srt = gr.Checkbox(label="Generate Subtitles", value=True)
                        custom_conv_punc = gr.Checkbox(label="Smart Punctuation", value=True)
                    custom_btn = gr.Button("Generate Audio", variant="primary", size="lg")
                with gr.Column():
                    custom_audio = gr.Audio(label="Generated Speech")
                    with gr.Group():
                        custom_srt = gr.Textbox(label="SRT Preview", lines=6, interactive=False)
                    with gr.Group():
                        custom_status = gr.Textbox(label="Status", interactive=False, lines=2)
                    custom_zip = gr.DownloadButton("Download ZIP (WAV + SRT)", visible=False)

            def on_custom(text, name, instr, gen_srt, conv_punc):
                import time
                start_time = time.time()
                # Clear previous outputs
                yield None, "", gr.update(visible=False), "Initializing..."
                
                # Phase 1: Audio
                tts_start = time.time()
                audio_path = None
                last_status = ""
                for status in custom_voice(text, name, instr, gen_srt=False, convert_punc=conv_punc, status_callback=lambda m: print(f"UI: {m}")):
                    if isinstance(status, str):
                        last_status = status
                        yield None, "", gr.update(visible=False), status
                    else:
                        audio_path, _ = status
                
                tts_dur = time.time() - tts_start
                
                if not audio_path:
                    if not last_status.startswith("Error"):
                        yield None, gr.update(), gr.update(visible=False), "Generation failed."
                    return
                
                yield audio_path, gr.update(), gr.update(visible=False), "Audio ready. Aligning subtitles..."
                
                # Phase 2: SRT
                asr_start = time.time()
                srt = ""
                if gen_srt:
                    from omnivoice.omni_engine_colab import generate_srt, clean_script
                    clean_text = clean_script(text)
                    for status in generate_srt(clean_text, audio_path, total_start_time=start_time):
                        if isinstance(status, str) and not status.startswith("1\n"):
                            yield audio_path, gr.update(), gr.update(visible=False), status
                        else:
                            srt = status
                asr_dur = time.time() - asr_start
                
                total_dur = time.time() - start_time
                word_count = count_words(text)
                perf_msg = f"Done! Total: {total_dur:.1f}s | Gen: {tts_dur:.1f}s | Asr: {asr_dur:.1f}s | Words: {word_count}"
                    
                zip_path = package_zip(text, audio_path, srt)
                yield audio_path, srt, gr.update(value=zip_path, visible=True), perf_msg

            custom_btn.click(
                on_custom,
                inputs=[custom_text, custom_name, custom_instr, custom_gen_srt, custom_conv_punc],
                outputs=[custom_audio, custom_srt, custom_zip, custom_status]
            )

        with gr.Tab("Voice Design"):
            with gr.Row():
                with gr.Column():
                    design_text = gr.Textbox(label="Input Text", placeholder="Enter text...", lines=6)
                    design_desc = gr.Textbox(
                        label="Voice Description", 
                        placeholder="e.g. A middle-aged man with a deep, raspy voice and a calm tone.",
                        lines=3
                    )
                    with gr.Row():
                        design_gen_srt = gr.Checkbox(label="Generate Subtitles", value=True)
                        design_conv_punc = gr.Checkbox(label="Smart Punctuation", value=True)
                    design_btn = gr.Button("Generate Audio", variant="primary", size="lg")
                with gr.Column():
                    design_audio = gr.Audio(label="Generated Speech")
                    with gr.Group():
                        design_srt = gr.Textbox(label="SRT Preview", lines=6, interactive=False)
                    with gr.Group():
                        design_status = gr.Textbox(label="Status", interactive=False, lines=2)
                    design_zip = gr.DownloadButton("Download ZIP (WAV + SRT)", visible=False)

            def on_design(text, desc, gen_srt, conv_punc):
                import time
                start_time = time.time()
                # Clear previous outputs
                yield None, "", gr.update(visible=False), "Initializing..."
                
                # Phase 1: Audio
                tts_start = time.time()
                audio_path = None
                last_status = ""
                for status in voice_design(text, desc, gen_srt=False, convert_punc=conv_punc, status_callback=lambda m: print(f"UI: {m}")):
                    if isinstance(status, str):
                        last_status = status
                        yield None, "", gr.update(visible=False), status
                    else:
                        audio_path, _ = status
                
                tts_dur = time.time() - tts_start
                
                if not audio_path:
                    if not last_status.startswith("Error"):
                        yield None, gr.update(), gr.update(visible=False), "Generation failed."
                    return
                
                yield audio_path, gr.update(), gr.update(visible=False), "Audio ready. Aligning subtitles..."
                
                # Phase 2: SRT
                asr_start = time.time()
                srt = ""
                if gen_srt:
                    from omnivoice.omni_engine_colab import generate_srt, clean_script
                    clean_text = clean_script(text)
                    for status in generate_srt(clean_text, audio_path, total_start_time=start_time):
                        if isinstance(status, str) and not status.startswith("1\n"):
                            yield audio_path, gr.update(), gr.update(visible=False), status
                        else:
                            srt = status
                asr_dur = time.time() - asr_start
                
                total_dur = time.time() - start_time
                word_count = count_words(text)
                perf_msg = f"Done! Total: {total_dur:.1f}s | Gen: {tts_dur:.1f}s | Asr: {asr_dur:.1f}s | Words: {word_count}"
                    
                zip_path = package_zip(text, audio_path, srt)
                yield audio_path, srt, gr.update(value=zip_path, visible=True), perf_msg

            design_btn.click(
                on_design,
                inputs=[design_text, design_desc, design_gen_srt, design_conv_punc],
                outputs=[design_audio, design_srt, design_zip, design_status]
            )

        # ─────────────────────────────────────────────
        # ROLE MAKER TAB — compile .qwen3tts files
        # ─────────────────────────────────────────────
        with gr.Tab("🔧 Role Maker"):
            gr.Markdown("""
            ### Compile a Voice → `.qwen3tts`
            Upload a reference audio clip and its transcript, give the role a name, then hit **Compile**.  
            The resulting `.qwen3tts` file can be uploaded into any Role panel in the **Voice Cloning** tab — no need to re-process the reference audio each session.
            """)
            with gr.Row():
                with gr.Column(scale=2):
                    maker_audio = gr.Audio(label="Reference Audio", type="filepath")
                    maker_text = gr.Textbox(
                        label="Transcript",
                        placeholder="Type the transcript, or use Trans Ref to auto-transcribe...",
                        lines=4
                    )
                    with gr.Row():
                        maker_trans_btn = gr.Button("Trans Ref", variant="primary", size="sm")
                        maker_zip_btn = gr.UploadButton("Ref Zip", file_types=[".zip"], variant="primary", size="sm")
                        maker_txt_btn = gr.UploadButton("Ref Txt", file_types=[".txt"], variant="primary", size="sm")
                        maker_clear_btn = gr.Button("Clear", variant="secondary", size="sm")
                    maker_name = gr.Textbox(
                        label="Role Name (used as filename)",
                        placeholder="e.g. Natasha"
                    )
                    maker_compile_btn = gr.Button("COMPILE TO .QWEN3TTS", variant="primary", size="lg")
                
                with gr.Column(scale=1):
                    gr.Markdown("### Output")
                    maker_download = gr.File(label="Download Compiled Voice (.qwen3tts)")
                    gr.Markdown("### System Status")
                    maker_status = gr.Textbox(
                        label="", value="Ready", interactive=False, lines=4
                    )

            # Maker callbacks
            maker_trans_btn.click(on_transcribe, inputs=[maker_audio], outputs=[maker_text])
            maker_zip_btn.upload(process_ref_zip, inputs=[maker_zip_btn], outputs=[maker_audio, maker_text])
            maker_txt_btn.upload(process_ref_txt, inputs=[maker_txt_btn], outputs=[maker_text])
            maker_clear_btn.click(lambda: (None, "", ""), outputs=[maker_audio, maker_text, maker_name])

            def on_compile(audio, text, name):
                """Wrapper: calls compile_role, yields status, returns file path."""
                updates = []
                status_msgs = []

                def _cb(msg):
                    status_msgs.append(msg)

                yield gr.update(), f"Starting compilation for '{name}'..."
                filepath, final_msg = compile_role(audio, text, name, status_callback=_cb)
                log = "\n".join(status_msgs[-6:]) + "\n" + final_msg  # Show last 6 status lines
                if filepath and os.path.exists(filepath):
                    yield gr.update(value=filepath, visible=True), log
                else:
                    yield gr.update(visible=False), log

            maker_compile_btn.click(
                on_compile,
                inputs=[maker_audio, maker_text, maker_name],
                outputs=[maker_download, maker_status]
            )

    return demo

if __name__ == "__main__":
    demo = create_app()
    demo.launch(share=True)

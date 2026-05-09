import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["PYTHONWARNINGS"] = "ignore"
import gradio as gr
from omnivoice.omni_engine_colab import voice_clone, custom_voice, voice_design, transcribe_ref
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
        gr.Markdown("# 🎙️ Qwen3-TTS")
        gr.Markdown("Advanced Text-to-Speech AI | Voice Cloning, Custom Voice & Voice Design")

        with gr.Tab("Voice Cloning"):
            gr.Markdown("### Clone any voice with 3+ seconds of audio")
            with gr.Row():
                with gr.Column():
                    clone_text = gr.Textbox(label="Text to Synthesize", placeholder="Enter text...", lines=4)
                    clone_audio = gr.Audio(label="Reference Audio (3+ seconds)", type="filepath")
                    clone_transcript = gr.Textbox(label="Reference Transcript (Optional)", placeholder="What's said in the reference audio... Highly recommended for best quality.", lines=3)
                    
                    with gr.Row():
                        trans_btn = gr.Button("Trans Ref", variant="secondary")
                        ref_zip_btn = gr.UploadButton("Ref Zip", file_types=[".zip"], variant="secondary")
                        ref_txt_btn = gr.UploadButton("Ref Txt", file_types=[".txt"], variant="secondary")
                        
                    with gr.Accordion("Advanced Settings", open=False):
                        clone_gen_srt = gr.Checkbox(label="Generate Subtitles", value=True)
                        clone_conv_punc = gr.Checkbox(label="Convert Punctuation", value=True)

                    clone_btn = gr.Button("Generate Speech", variant="primary", size="lg")
                with gr.Column():
                    clone_output = gr.Audio(label="Generated Speech")
                    with gr.Group():
                        clone_srt_preview = gr.Textbox(label="SRT Preview", lines=6, interactive=False)

                    with gr.Group():
                        clone_status = gr.Textbox(label="Status", value="", interactive=False, lines=2)
                    clone_zip_dl = gr.DownloadButton("📥 Download ZIP (WAV + SRT)", visible=False)

            # Transcription handler
            def on_transcribe(audio):
                if not audio: return ""
                return transcribe_ref(audio)

            def process_ref_zip(zip_file):
                if not zip_file: return None, ""
                import zipfile, tempfile
                audio_path = None
                text_content = ""
                tmp = tempfile.mkdtemp()
                with zipfile.ZipFile(zip_file.name, 'r') as z:
                    z.extractall(tmp)
                    for f in z.namelist():
                        if f.endswith(('.wav', '.mp3', '.flac')) and not f.startswith('__MACOSX') and not os.path.basename(f).startswith('.'):
                            audio_path = os.path.join(tmp, f)
                        if f.endswith('.txt') and not f.startswith('__MACOSX') and not os.path.basename(f).startswith('.'):
                            txt_path = os.path.join(tmp, f)
                            try:
                                with open(txt_path, 'r', encoding='utf-8') as tf:
                                    text_content = tf.read()
                            except UnicodeDecodeError:
                                with open(txt_path, 'r', encoding='gbk') as tf:
                                    text_content = tf.read()
                return audio_path, text_content

            def process_ref_txt(txt_file):
                if not txt_file: return ""
                try:
                    with open(txt_file.name, 'r', encoding='utf-8') as tf:
                        return tf.read()
                except UnicodeDecodeError:
                    with open(txt_file.name, 'r', encoding='gbk') as tf:
                        return tf.read()

            def on_clone(text, audio, transcript, gen_srt, conv_punc):
                import time
                start_time = time.time()
                
                # Phase 1: Generate Audio
                def update_status(msg):
                    # We can't yield from inside a nested function, 
                    # but Gradio handlers can't easily see this.
                    # Instead, we just let the print handles the console 
                    # and the main generator handles the yield.
                    pass
                
                # Note: To actually see the chunk updates in Gradio, 
                # we need to make the engine functions generators 
                # or handle updates differently.
                tts_start = time.time()
                audio_path = None
                last_status = ""
                for status in voice_clone(text, audio, transcript, gen_srt=False, convert_punc=conv_punc, status_callback=lambda m: print(f"UI: {m}")):
                    if isinstance(status, str):
                        last_status = status
                        yield gr.update(), gr.update(), gr.update(visible=False), status
                    else:
                        audio_path, _ = status
                
                tts_dur = time.time() - tts_start
                
                if not audio_path:
                    if not last_status.startswith("❌"):
                        yield None, gr.update(), gr.update(visible=False), "❌ Generation failed."
                    return
                
                # Show audio immediately
                yield audio_path, gr.update(), gr.update(visible=False), "⏳ Audio ready. Aligning subtitles..."
                
                # Phase 2: Subtitles (Memory Intensive)
                asr_start = time.time()
                srt = ""
                if gen_srt:
                    from omnivoice.omni_engine_colab import generate_srt
                    for status in generate_srt(text, audio_path, total_start_time=start_time):
                        if isinstance(status, str) and not status.startswith("1\n"): # Check if it's status or SRT
                            yield audio_path, gr.update(), gr.update(visible=False), status
                        else:
                            srt = status
                asr_dur = time.time() - asr_start
                
                # Performance metrics
                total_dur = time.time() - start_time
                word_count = count_words(text)
                perf_msg = f"✅ Total: {total_dur:.1f}s | Gen: {tts_dur:.1f}s | Asr: {asr_dur:.1f}s | Words: {word_count}"
                
                zip_path = package_zip(text, audio_path, srt)
                yield audio_path, srt, gr.update(value=zip_path, visible=True), perf_msg

            trans_btn.click(on_transcribe, inputs=[clone_audio], outputs=[clone_transcript])
            ref_zip_btn.upload(process_ref_zip, inputs=[ref_zip_btn], outputs=[clone_audio, clone_transcript])
            ref_txt_btn.upload(process_ref_txt, inputs=[ref_txt_btn], outputs=[clone_transcript])

            clone_btn.click(on_clone, inputs=[clone_text, clone_audio, clone_transcript, clone_gen_srt, clone_conv_punc], outputs=[clone_output, clone_srt_preview, clone_zip_dl, clone_status])

        with gr.Tab("Custom Voice"):
            gr.Markdown("### Use 9 preset character voices with style control")
            with gr.Row():
                with gr.Column():
                    custom_text = gr.Textbox(label="Text to Synthesize", placeholder="Enter text...", lines=4)
                    custom_voice_name = gr.Dropdown(
                        choices=["serena", "vivian", "ono_anna", "sohee", "aiden", "dylan", "eric", "ryan", "uncle_fu"],
                        label="Voice Character", value="serena"
                    )
                    custom_instruction = gr.Textbox(label="Style Instruction (Optional)", placeholder="e.g., 'speak slowly and cheerfully'", lines=2)
                    
                    with gr.Accordion("Advanced Settings", open=False):
                        custom_gen_srt = gr.Checkbox(label="Generate Subtitles", value=True)
                        custom_conv_punc = gr.Checkbox(label="Convert Punctuation", value=True)
                        
                    custom_btn = gr.Button("Generate Speech", variant="primary", size="lg")
                with gr.Column():
                    custom_output = gr.Audio(label="Generated Speech")
                    with gr.Group():
                        custom_srt_preview = gr.Textbox(label="SRT Preview", lines=6, interactive=False)

                    with gr.Group():
                        custom_status = gr.Textbox(label="Status", value="", interactive=False, lines=2)
                    custom_zip_dl = gr.DownloadButton("📥 Download ZIP (WAV + SRT)", visible=False)

            def on_custom(text, name, instr, gen_srt, conv_punc):
                import time
                start_time = time.time()
                # Phase 1: Audio
                tts_start = time.time()
                audio_path = None
                last_status = ""
                for status in custom_voice(text, name, instr, gen_srt=False, convert_punc=conv_punc, status_callback=lambda m: print(f"UI: {m}")):
                    if isinstance(status, str):
                        last_status = status
                        yield gr.update(), gr.update(), gr.update(visible=False), status
                    else:
                        audio_path, _ = status
                
                tts_dur = time.time() - tts_start
                
                if not audio_path:
                    if not last_status.startswith("❌"):
                        yield None, gr.update(), gr.update(visible=False), "❌ Generation failed."
                    return
                
                yield audio_path, gr.update(), gr.update(visible=False), "⏳ Audio ready. Aligning subtitles..."
                
                # Phase 2: SRT
                asr_start = time.time()
                srt = ""
                if gen_srt:
                    from omnivoice.omni_engine_colab import generate_srt
                    for status in generate_srt(text, audio_path, total_start_time=start_time):
                        if isinstance(status, str) and not status.startswith("1\n"):
                            yield audio_path, gr.update(), gr.update(visible=False), status
                        else:
                            srt = status
                asr_dur = time.time() - asr_start
                
                total_dur = time.time() - start_time
                word_count = count_words(text)
                perf_msg = f"✅ Total: {total_dur:.1f}s | Gen: {tts_dur:.1f}s | Asr: {asr_dur:.1f}s | Words: {word_count}"
                    
                zip_path = package_zip(text, audio_path, srt)
                yield audio_path, srt, gr.update(value=zip_path, visible=True), perf_msg

            custom_btn.click(on_custom, inputs=[custom_text, custom_voice_name, custom_instruction, custom_gen_srt, custom_conv_punc], outputs=[custom_output, custom_srt_preview, custom_zip_dl, custom_status])

        with gr.Tab("Voice Design"):
            gr.Markdown("### Design a unique voice from text description")
            with gr.Row():
                with gr.Column():
                    design_text = gr.Textbox(label="Text to Synthesize", placeholder="Enter text...", lines=4)
                    design_description = gr.Textbox(label="Voice Description", placeholder="A young female, cheerful, speaking clearly", lines=4)
                    
                    with gr.Accordion("Advanced Settings", open=False):
                        design_gen_srt = gr.Checkbox(label="Generate Subtitles", value=True)
                        design_conv_punc = gr.Checkbox(label="Convert Punctuation", value=True)
                        
                    design_btn = gr.Button("Generate Speech", variant="primary", size="lg")
                with gr.Column():
                    design_output = gr.Audio(label="Generated Speech")
                    with gr.Group():
                        design_srt_preview = gr.Textbox(label="SRT Preview", lines=6, interactive=False)

                    with gr.Group():
                        design_status = gr.Textbox(label="Status", value="", interactive=False, lines=2)
                    design_zip_dl = gr.DownloadButton("📥 Download ZIP (WAV + SRT)", visible=False)

            def on_design(text, desc, gen_srt, conv_punc):
                import time
                start_time = time.time()
                # Phase 1: Audio
                tts_start = time.time()
                audio_path = None
                last_status = ""
                for status in voice_design(text, desc, gen_srt=False, convert_punc=conv_punc, status_callback=lambda m: print(f"UI: {m}")):
                    if isinstance(status, str):
                        last_status = status
                        yield gr.update(), gr.update(), gr.update(visible=False), status
                    else:
                        audio_path, _ = status
                
                tts_dur = time.time() - tts_start
                
                if not audio_path:
                    if not last_status.startswith("❌"):
                        yield None, gr.update(), gr.update(visible=False), "❌ Generation failed."
                    return
                
                yield audio_path, gr.update(), gr.update(visible=False), "⏳ Audio ready. Aligning subtitles..."
                
                # Phase 2: SRT
                asr_start = time.time()
                srt = ""
                if gen_srt:
                    from omnivoice.omni_engine_colab import generate_srt
                    for status in generate_srt(text, audio_path, total_start_time=start_time):
                        if isinstance(status, str) and not status.startswith("1\n"):
                            yield audio_path, gr.update(), gr.update(visible=False), status
                        else:
                            srt = status
                asr_dur = time.time() - asr_start
                
                total_dur = time.time() - start_time
                word_count = count_words(text)
                perf_msg = f"✅ Total: {total_dur:.1f}s | Gen: {tts_dur:.1f}s | Asr: {asr_dur:.1f}s | Words: {word_count}"
                    
                zip_path = package_zip(text, audio_path, srt)
                yield audio_path, srt, gr.update(value=zip_path, visible=True), perf_msg

            design_btn.click(on_design, inputs=[design_text, design_description, design_gen_srt, design_conv_punc], outputs=[design_output, design_srt_preview, design_zip_dl, design_status])

    return demo

if __name__ == "__main__":
    demo = create_app()
    demo.launch(share=True)

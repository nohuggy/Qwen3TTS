import os
import gradio as gr
from omnivoice.omni_engine_colab import voice_clone, custom_voice, voice_design, transcribe_ref

# Minimal CSS
custom_css = """
.gr-button-primary {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    border: none !important;
}
"""

def package_zip(text, audio_path, srt_content):
    """Package audio and SRT into a ZIP for download using slug-based naming"""
    if not audio_path: return None
    import zipfile
    from omnivoice.omni_engine_colab import get_slug
    
    slug = get_slug(text)
    zip_path = audio_path.replace(".wav", ".zip")
    
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
                    clone_srt_preview = gr.Textbox(label="SRT Preview", lines=6, interactive=False)
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
                # Phase 1: Generate Audio
                audio_path, _ = voice_clone(text, audio, transcript, gen_srt=False, convert_punc=conv_punc)
                if not audio_path:
                    yield None, "❌ Generation failed.", gr.update(visible=False)
                    return
                
                # Show audio immediately
                yield audio_path, "Audio ready. Aligning subtitles...", gr.update(visible=False)
                
                # Phase 2: Subtitles (Memory Intensive)
                srt = ""
                if gen_srt:
                    from omnivoice.omni_engine_colab import generate_srt
                    srt = generate_srt(text, audio_path)
                
                zip_path = package_zip(text, audio_path, srt)
                yield audio_path, srt, gr.update(value=zip_path, visible=True)

            trans_btn.click(on_transcribe, inputs=[clone_audio], outputs=[clone_transcript])
            ref_zip_btn.upload(process_ref_zip, inputs=[ref_zip_btn], outputs=[clone_audio, clone_transcript])
            ref_txt_btn.upload(process_ref_txt, inputs=[ref_txt_btn], outputs=[clone_transcript])

            clone_btn.click(on_clone, inputs=[clone_text, clone_audio, clone_transcript, clone_gen_srt, clone_conv_punc], outputs=[clone_output, clone_srt_preview, clone_zip_dl])

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
                    custom_srt_preview = gr.Textbox(label="SRT Preview", lines=6, interactive=False)
                    custom_zip_dl = gr.DownloadButton("📥 Download ZIP (WAV + SRT)", visible=False)

            def on_custom(text, name, instr, gen_srt, conv_punc):
                # Phase 1: Audio
                audio_path, _ = custom_voice(text, name, instr, gen_srt=False, convert_punc=conv_punc)
                if not audio_path:
                    yield None, "❌ Generation failed.", gr.update(visible=False)
                    return
                
                yield audio_path, "Audio ready. Aligning subtitles...", gr.update(visible=False)
                
                # Phase 2: SRT
                srt = ""
                if gen_srt:
                    from omnivoice.omni_engine_colab import generate_srt
                    srt = generate_srt(text, audio_path)
                    
                zip_path = package_zip(text, audio_path, srt)
                yield audio_path, srt, gr.update(value=zip_path, visible=True)

            custom_btn.click(on_custom, inputs=[custom_text, custom_voice_name, custom_instruction, custom_gen_srt, custom_conv_punc], outputs=[custom_output, custom_srt_preview, custom_zip_dl])

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
                    design_srt_preview = gr.Textbox(label="SRT Preview", lines=6, interactive=False)
                    design_zip_dl = gr.DownloadButton("📥 Download ZIP (WAV + SRT)", visible=False)

            def on_design(text, desc, gen_srt, conv_punc):
                # Phase 1: Audio
                audio_path, _ = voice_design(text, desc, gen_srt=False, convert_punc=conv_punc)
                if not audio_path:
                    yield None, "❌ Generation failed.", gr.update(visible=False)
                    return
                
                yield audio_path, "Audio ready. Aligning subtitles...", gr.update(visible=False)
                
                # Phase 2: SRT
                srt = ""
                if gen_srt:
                    from omnivoice.omni_engine_colab import generate_srt
                    srt = generate_srt(text, audio_path)
                    
                zip_path = package_zip(text, audio_path, srt)
                yield audio_path, srt, gr.update(value=zip_path, visible=True)

            design_btn.click(on_design, inputs=[design_text, design_description, design_gen_srt, design_conv_punc], outputs=[design_output, design_srt_preview, design_zip_dl])

    return demo

if __name__ == "__main__":
    demo = create_app()
    demo.launch(share=True)

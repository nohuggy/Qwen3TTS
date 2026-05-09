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

        with gr.Tab("Voice Cloning (Multi-Speaker)"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 🎭 Role Bank")
                    with gr.Group():
                        gr.Markdown("#### Role 1 (Primary)")
                        r1_name = gr.Textbox(label="Role Name", placeholder="e.g. Alex", value="")
                        r1_audio = gr.Audio(label="Reference Audio", type="filepath")
                        r1_text = gr.Textbox(label="Reference Transcript", placeholder="Text from the audio...")
                    
                    with gr.Accordion("🎭 Role 2", open=False):
                        r2_name = gr.Textbox(label="Role Name", placeholder="e.g. Sara")
                        r2_audio = gr.Audio(label="Reference Audio", type="filepath")
                        r2_text = gr.Textbox(label="Reference Transcript", placeholder="Text from the audio...")
                        
                    with gr.Accordion("🎭 Role 3", open=False):
                        r3_name = gr.Textbox(label="Role Name", placeholder="e.g. Bob")
                        r3_audio = gr.Audio(label="Reference Audio", type="filepath")
                        r3_text = gr.Textbox(label="Reference Transcript", placeholder="Text from the audio...")

                with gr.Column(scale=1):
                    gr.Markdown("### 📜 Script")
                    input_text = gr.Textbox(
                        label="Input Text", 
                        placeholder="## Alex ## [shout] Hello! [pause:0.5] How are you?\n## Sara ## [happy] I'm fine!", 
                        lines=15
                    )
                    with gr.Row():
                        gen_srt = gr.Checkbox(label="Generate Subtitles", value=True)
                        conv_punc = gr.Checkbox(label="Smart Punctuation", value=True)
                    btn = gr.Button("🚀 Generate Audio", variant="primary")
                    audio_out = gr.Audio(label="Generated Speech")
                    srt_out = gr.Textbox(label="SRT Preview", lines=6, interactive=False)
                    status_out = gr.Textbox(label="Status", value="", interactive=False, lines=2)
                    zip_out = gr.DownloadButton("📥 Download ZIP (WAV + SRT)", visible=False)

            def on_clone(text, 
                         name1, audio1, text1, 
                         name2, audio2, text2, 
                         name3, audio3, text3, 
                         gen_srt, conv_punc):
                import time
                start_time = time.time()
                # Clear previous outputs immediately
                yield None, "", gr.update(visible=False), "⏳ Initializing..."
                
                # Build role bank data
                role_bank_data = []
                if audio1: role_bank_data.append({'name': name1, 'audio': audio1, 'text': text1})
                if audio2: role_bank_data.append({'name': name2, 'audio': audio2, 'text': text2})
                if audio3: role_bank_data.append({'name': name3, 'audio': audio3, 'text': text3})
                
                tts_start = time.time()
                audio_path = None
                last_status = ""
                for status in voice_clone(text, role_bank_data, gen_srt=False, convert_punc=conv_punc, status_callback=lambda m: print(f"UI: {m}")):
                    if isinstance(status, str):
                        last_status = status
                        yield None, "", gr.update(visible=False), status
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

            btn.click(
                on_clone,
                inputs=[input_text, r1_name, r1_audio, r1_text, r2_name, r2_audio, r2_text, r3_name, r3_audio, r3_text, gen_srt, conv_punc],
                outputs=[audio_out, srt_out, zip_out, status_out]
            )

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
                # Clear previous outputs
                yield None, "", gr.update(visible=False), "⏳ Initializing..."
                
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
                # Clear previous outputs
                yield None, "", gr.update(visible=False), "⏳ Initializing..."
                
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

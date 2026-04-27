import os
import re
import io
import gradio as gr
from dotenv import load_dotenv
from huggingface_hub import login
from google import genai
from google.genai import types
from PIL import Image
from music_generator import VideoGameMusicGenerator, AceStepMusicGenerator
from evaluator import MusicEvaluator
from midi_engine import MidiTranscriptionEngine

load_dotenv()
hf_token = os.environ.get("HF_TOKEN")
if hf_token:
    print("Authenticating with Hugging Face...")
    login(token=hf_token)
else:
    print("No HF_TOKEN found in environment. Proceeding without authentication.")

print("Initializing models...")
generator = VideoGameMusicGenerator()
ace_generator = AceStepMusicGenerator()
evaluator = MusicEvaluator()
transcription_engine = MidiTranscriptionEngine()
print("Models loaded successfully.")

def generate_race_asset(race_name, description, revisions, current_dir):
    if not race_name:
        return None
        
    prompt_text = f"Concept art for a video game race named '{race_name}'. Description: {description}."
    if revisions:
        prompt_text += f" Also incorporate these elements: {revisions}"
        
    try:
        client = genai.Client()
        
        # Using the high-efficiency Nano Banana 2 (Gemini 3.1 Flash Image)
        response = client.models.generate_content(
            model="gemini-3.1-flash-image-preview",
            contents=[prompt_text],
            config=types.GenerateContentConfig(
                response_modalities=["Image"],
                image_config=types.ImageConfig(
                    aspect_ratio="16:9", # Perfect for cinematic game backgrounds
                    image_size="4K"      # High-fidelity 4K output
                )
            )
        )

        # Determine output directory
        if current_dir and os.path.exists(current_dir):
            output_dir = current_dir
        else:
            base_output_dir = "Output"
            os.makedirs(base_output_dir, exist_ok=True)
            safe_race_name = "".join(c if c.isalnum() else "_" for c in race_name).lower()
            safe_race_name = re.sub(r'_+', '_', safe_race_name).strip('_')
            if not safe_race_name:
                safe_race_name = "unknown_race"
            folder_idx = 1
            while True:
                folder_name = f"{safe_race_name}_{folder_idx:02d}"
                output_dir = os.path.join(base_output_dir, folder_name)
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                    break
                folder_idx += 1

        # Process and save the image bits
        file_name = f"{race_name.strip().replace(' ', '_').lower()}.png"
        file_path = os.path.join(output_dir, file_name)
        for part in response.candidates[0].content.parts:
            if part.image:
                image = Image.open(io.BytesIO(part.image.bits))
                image.save(file_path)
                return file_path
    except Exception as e:
        print(f"Error generating image: {e}")
        return None
    return None

def _core_generate(race_name, description, revisions, duration, transcribe_midi, engine_choice):
    baseline_prompt, improved_prompt, ace_step_prompt, bpm, key = generator.create_prompts(description, revisions)
    
    sr_base, audio_base = None, None
    sr_imp, audio_imp = None, None
    sr_ace, audio_ace = None, None
    
    metrics_base = None
    metrics_imp = None
    metrics_ace = None

    if engine_choice == "MusicGen":
        # Generate Baseline
        generator.load_to_gpu()
        sr_base, audio_base = generator.generate_music(baseline_prompt, duration)
        
        # Generate Improved
        sr_imp, audio_imp = generator.generate_music(improved_prompt, duration)
        
        generator.offload_to_cpu()
        
        metrics_base = evaluator.evaluate_all(audio_base, sr_base, baseline_prompt)
        metrics_imp = evaluator.evaluate_all(audio_imp, sr_imp, improved_prompt)
        
    elif engine_choice == "ACE-Step SFT (Experimental)":
        ace_generator.load_to_gpu()
        sr_ace, audio_ace = ace_generator.generate_music(ace_step_prompt, duration, bpm=bpm, keyscale=key)
        ace_generator.offload_to_cpu()
        
        metrics_ace = evaluator.evaluate_all(audio_ace, sr_ace, ace_step_prompt)
    
    # Format Results
    def fmt(m, key):
        return f"{m[key]:.4f}" if m else "N/A"

    results_md = f"""
| Metric | Baseline | Improved | ACE-Step |
|---|---|---|---|
| Quality | {fmt(metrics_base, 'quality')} | {fmt(metrics_imp, 'quality')} | {fmt(metrics_ace, 'quality')} |
| Alignment | {fmt(metrics_base, 'alignment')} | {fmt(metrics_imp, 'alignment')} | {fmt(metrics_ace, 'alignment')} |
| Realism | {fmt(metrics_base, 'realism')} | {fmt(metrics_imp, 'realism')} | {fmt(metrics_ace, 'realism')} |
| Creativity | {fmt(metrics_base, 'creativity')} | {fmt(metrics_imp, 'creativity')} | {fmt(metrics_ace, 'creativity')} |
| Loopability | {fmt(metrics_base, 'loopability')} | {fmt(metrics_imp, 'loopability')} | {fmt(metrics_ace, 'loopability')} |
"""

    base_output_dir = "Output"
    os.makedirs(base_output_dir, exist_ok=True)
    
    safe_race_name = "".join(c if c.isalnum() else "_" for c in race_name).lower()
    safe_race_name = re.sub(r'_+', '_', safe_race_name).strip('_')
    if not safe_race_name:
        safe_race_name = "unknown_race"
        
    folder_idx = 1
    while True:
        folder_name = f"{safe_race_name}_{folder_idx:02d}"
        output_dir = os.path.join(base_output_dir, folder_name)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            break
        folder_idx += 1

    baseline_audio_path = os.path.join(output_dir, "baseline_audio.wav")
    improved_audio_path = os.path.join(output_dir, "improved_audio.wav")
    ace_audio_path = os.path.join(output_dir, "ace_step_audio.wav")
    prompt_file_path = os.path.join(output_dir, "prompt.md")

    if audio_base is not None:
        generator.save_audio(baseline_audio_path, sr_base, audio_base)
    if audio_imp is not None:
        generator.save_audio(improved_audio_path, sr_imp, audio_imp)
    if audio_ace is not None:
        generator.save_audio(ace_audio_path, sr_ace, audio_ace)

    revisions_line = f"\n\n**Revisions / Tweaks:** {revisions}" if revisions and revisions.strip() else ""
    prompt_file_content = (
        f"# Given Prompt\n\n"
        f"**Race Name:** {race_name}\n\n"
        f"**Race Description:** {description}"
        f"{revisions_line}\n\n"
        f"---\n\n"
        f"**Base Prompt** *(MusicGen)*\n"
        f"`{baseline_prompt}`\n\n"
        f"**Improved Prompt** *(MusicGen)*\n"
        f"`{improved_prompt}`\n\n"
        f"**ACE-Step Prompt** *(tag-based)*\n"
        f"`{ace_step_prompt}`\n\n"
        f"**Metrics**\n"
        f"{results_md.strip()}\n"
    )
    with open(prompt_file_path, "w", encoding="utf-8") as f:
        f.write(prompt_file_content)

    midi_path, xml_path, svg_path = None, None, None
    if transcribe_midi:
        if engine_choice == "MusicGen" and audio_imp is not None:
            midi_path, xml_path, svg_path = transcription_engine.generate_score(improved_audio_path, output_dir)
        elif engine_choice == "ACE-Step SFT (Experimental)" and audio_ace is not None:
            midi_path, xml_path, svg_path = transcription_engine.generate_score(ace_audio_path, output_dir)

    out_base_audio = baseline_audio_path if audio_base is not None else gr.update()
    out_imp_audio = improved_audio_path if audio_imp is not None else gr.update()
    out_ace_audio = ace_audio_path if audio_ace is not None else gr.update()
    
    out_base_prompt = baseline_prompt if engine_choice == "MusicGen" else gr.update()
    out_imp_prompt = improved_prompt if engine_choice == "MusicGen" else gr.update()
    out_ace_prompt = ace_step_prompt if engine_choice == "ACE-Step SFT (Experimental)" else gr.update()

    return out_base_audio, out_imp_audio, out_ace_audio, out_base_prompt, out_imp_prompt, out_ace_prompt, results_md, prompt_file_path, midi_path or gr.update(), xml_path or gr.update(), svg_path or gr.update(), output_dir

def generate_and_evaluate(race_name, description, revisions, duration, transcribe_midi, engine_choice):
    return _core_generate(race_name, description, revisions, duration, transcribe_midi, engine_choice)

def run_musicgen_only(race_name, description, revisions, duration, transcribe_midi):
    return _core_generate(race_name, description, revisions, duration, transcribe_midi, "MusicGen")

def run_acestep_only(race_name, description, revisions, duration, transcribe_midi):
    return _core_generate(race_name, description, revisions, duration, transcribe_midi, "ACE-Step SFT (Experimental)")

def transcribe_only(audio_path, output_dir):
    if not audio_path:
        return None, None, None
    import os
    if not output_dir or not os.path.exists(output_dir):
        output_dir = os.path.dirname(audio_path)
    midi_path, xml_path, svg_path = transcription_engine.generate_score(audio_path, output_dir)
    return midi_path, xml_path, svg_path

with gr.Blocks(title="Video Game Race Music AI") as demo:
    gr.Markdown("# Video Game Race Music AI")
    gr.Markdown("Describe a new video game race, and the AI will generate MIDI-orchestral music tailored to their culture.")
    
    with gr.Row():
        current_output_dir = gr.State(value="")
        with gr.Column():
            name_input = gr.Textbox(label="Race Name", placeholder="e.g., Goblin", lines=1)
            desc_input = gr.Textbox(label="Race Description", placeholder="e.g., An ancient race of tree-dwellers, strong in magic but physically weak, secretive and melodic.", lines=3)
            rev_input = gr.Textbox(label="Revisions / Tweaks", placeholder="e.g., make it faster and add drums", lines=1)
            duration_slider = gr.Slider(minimum=5, maximum=30, value=10, step=1, label="Duration (seconds)")
            transcribe_checkbox = gr.Checkbox(label="Generate MIDI/XML Stems (Takes extra time)", value=False)
            engine_choice = gr.Radio(choices=["MusicGen", "ACE-Step SFT (Experimental)"], value="MusicGen", label="Primary Generator Engine")
            gr.Markdown("**Note:** ACE-Step is an *experimental* local feature. It requires a CUDA GPU to function correctly. CPU execution will produce severe audio artifacts.")
            with gr.Row():
                generate_btn = gr.Button("Generate & Evaluate", variant="primary")
                generate_img_btn = gr.Button("Generate Concept Art", variant="secondary")
            
        with gr.Column():
            gr.Markdown("### Evaluation Results")
            results_out = gr.Markdown("Results will appear here.")
            download_prompt_out = gr.File(label="Download prompt.md")
            race_image_out = gr.Image(label="Race Concept Art", type="filepath", interactive=False)
            
    with gr.Row():
        with gr.Column():
            rerun_musicgen_btn = gr.Button("Re-run MusicGen")
            gr.Markdown("### Baseline Generation (MusicGen)")
            base_prompt_out = gr.Textbox(label="Baseline Prompt", interactive=False)
            base_audio_out = gr.Audio(label="Baseline Audio", type="filepath", loop=True)
            
        with gr.Column():
            gr.Markdown("### Improved Generation (MusicGen)")
            imp_prompt_out = gr.Textbox(label="Improved Prompt", interactive=False)
            imp_audio_out = gr.Audio(label="Improved Audio", type="filepath", loop=True)
            
        with gr.Column():
            run_acestep_btn = gr.Button("Run ACE-Step")
            gr.Markdown("### Experimental Generation (ACE-Step SFT)")
            ace_prompt_out = gr.Textbox(label="ACE-Step Prompt", interactive=False)
            ace_audio_out = gr.Audio(label="ACE-Step Audio", type="filepath", loop=True)

    with gr.Row():
        transcribe_btn = gr.Button("Transcribe Stems (MIDI/XML)", variant="secondary")

    with gr.Row():
        midi_out = gr.File(label="Download stems.mid")
        xml_out = gr.File(label="Download sheet_music.xml")

    with gr.Row():
        gr.Markdown("### Sheet Music Viewer")
        
    with gr.Row():
        svg_out = gr.Image(label="Sheet Music (SVG)", type="filepath", interactive=False)
            
    generate_btn.click(
        fn=generate_and_evaluate,
        inputs=[name_input, desc_input, rev_input, duration_slider, transcribe_checkbox, engine_choice],
        outputs=[base_audio_out, imp_audio_out, ace_audio_out, base_prompt_out, imp_prompt_out, ace_prompt_out, results_out, download_prompt_out, midi_out, xml_out, svg_out, current_output_dir]
    )

    generate_img_btn.click(
        fn=generate_race_asset,
        inputs=[name_input, desc_input, rev_input, current_output_dir],
        outputs=[race_image_out]
    )

    rerun_musicgen_btn.click(
        fn=run_musicgen_only,
        inputs=[name_input, desc_input, rev_input, duration_slider, transcribe_checkbox],
        outputs=[base_audio_out, imp_audio_out, ace_audio_out, base_prompt_out, imp_prompt_out, ace_prompt_out, results_out, download_prompt_out, midi_out, xml_out, svg_out, current_output_dir]
    )

    run_acestep_btn.click(
        fn=run_acestep_only,
        inputs=[name_input, desc_input, rev_input, duration_slider, transcribe_checkbox],
        outputs=[base_audio_out, imp_audio_out, ace_audio_out, base_prompt_out, imp_prompt_out, ace_prompt_out, results_out, download_prompt_out, midi_out, xml_out, svg_out, current_output_dir]
    )

    transcribe_btn.click(
        fn=transcribe_only,
        inputs=[imp_audio_out, current_output_dir],
        outputs=[midi_out, xml_out, svg_out]
    )

if __name__ == "__main__":
    demo.launch()

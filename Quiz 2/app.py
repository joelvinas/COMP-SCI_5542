import os
import re
import gradio as gr
from dotenv import load_dotenv
from huggingface_hub import login
from music_generator import VideoGameMusicGenerator
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
evaluator = MusicEvaluator()
transcription_engine = MidiTranscriptionEngine()
print("Models loaded successfully.")

def generate_and_evaluate(race_name, description, revisions, duration, transcribe_midi):
    baseline_prompt, improved_prompt = generator.create_prompts(description, revisions)
    
    # Generate Baseline
    sr_base, audio_base = generator.generate_music(baseline_prompt, duration)
    # Generate Improved
    sr_imp, audio_imp = generator.generate_music(improved_prompt, duration)
    
    # Evaluate
    metrics_base = evaluator.evaluate_all(audio_base, sr_base, baseline_prompt)
    metrics_imp = evaluator.evaluate_all(audio_imp, sr_imp, improved_prompt)
    
    # Format Results
    results_md = f"""
| Metric | Baseline | Improved |
|---|---|---|
| Quality | {metrics_base['quality']:.4f} | {metrics_imp['quality']:.4f} |
| Alignment | {metrics_base['alignment']:.4f} | {metrics_imp['alignment']:.4f} |
| Realism | {metrics_base['realism']:.4f} | {metrics_imp['realism']:.4f} |
| Creativity | {metrics_base['creativity']:.4f} | {metrics_imp['creativity']:.4f} |
| Loopability | {metrics_base['loopability']:.4f} | {metrics_imp['loopability']:.4f} |
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
    prompt_file_path = os.path.join(output_dir, "prompt.md")

    generator.save_audio(baseline_audio_path, sr_base, audio_base)
    generator.save_audio(improved_audio_path, sr_imp, audio_imp)

    prompt_file_content = f"""**Base Prompt**
`{baseline_prompt}`

**Improved Prompt**
`{improved_prompt}`

**Metrics**
{results_md.strip()}
"""
    with open(prompt_file_path, "w", encoding="utf-8") as f:
        f.write(prompt_file_content)

    midi_path, xml_path = None, None
    if transcribe_midi:
        midi_path, xml_path = transcription_engine.generate_score(improved_audio_path, output_dir)

    return baseline_audio_path, improved_audio_path, baseline_prompt, improved_prompt, results_md, prompt_file_path, midi_path, xml_path

def transcribe_only(audio_path):
    if not audio_path:
        return None, None
    import os
    output_dir = os.path.dirname(audio_path)
    midi_path, xml_path = transcription_engine.generate_score(audio_path, output_dir)
    return midi_path, xml_path

with gr.Blocks(title="Video Game Race Music AI") as demo:
    gr.Markdown("# Video Game Race Music AI")
    gr.Markdown("Describe a new video game race, and the AI will generate MIDI-orchestral music tailored to their culture.")
    
    with gr.Row():
        with gr.Column():
            name_input = gr.Textbox(label="Race Name", placeholder="e.g., Goblin", lines=1)
            desc_input = gr.Textbox(label="Race Description", placeholder="e.g., An ancient race of tree-dwellers, strong in magic but physically weak, secretive and melodic.", lines=3)
            rev_input = gr.Textbox(label="Revisions / Tweaks", placeholder="e.g., make it faster and add drums", lines=1)
            duration_slider = gr.Slider(minimum=5, maximum=30, value=10, step=1, label="Duration (seconds)")
            transcribe_checkbox = gr.Checkbox(label="Generate MIDI/XML Stems (Takes extra time)", value=False)
            generate_btn = gr.Button("Generate & Evaluate", variant="primary")
            
        with gr.Column():
            gr.Markdown("### Evaluation Results")
            results_out = gr.Markdown("Results will appear here.")
            download_prompt_out = gr.File(label="Download prompt.md")
            
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Baseline Generation")
            base_prompt_out = gr.Textbox(label="Baseline Prompt", interactive=False)
            base_audio_out = gr.Audio(label="Baseline Audio", type="filepath", loop=True)
            
        with gr.Column():
            gr.Markdown("### Improved Generation")
            imp_prompt_out = gr.Textbox(label="Improved Prompt", interactive=False)
            imp_audio_out = gr.Audio(label="Improved Audio", type="filepath", loop=True)
            midi_out = gr.File(label="Download stems.mid")
            xml_out = gr.File(label="Download sheet_music.xml")

    with gr.Row():
        transcribe_btn = gr.Button("Transcribe Stems (MIDI/XML)", variant="secondary")
            
    generate_btn.click(
        fn=generate_and_evaluate,
        inputs=[name_input, desc_input, rev_input, duration_slider, transcribe_checkbox],
        outputs=[base_audio_out, imp_audio_out, base_prompt_out, imp_prompt_out, results_out, download_prompt_out, midi_out, xml_out]
    )

    transcribe_btn.click(
        fn=transcribe_only,
        inputs=[imp_audio_out],
        outputs=[midi_out, xml_out]
    )

if __name__ == "__main__":
    demo.launch()

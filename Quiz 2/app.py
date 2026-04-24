import gradio as gr
from music_generator import VideoGameMusicGenerator
from evaluator import MusicEvaluator

print("Initializing models...")
generator = VideoGameMusicGenerator()
evaluator = MusicEvaluator()
print("Models loaded successfully.")

def generate_and_evaluate(description, revisions, duration):
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

    generator.save_audio("baseline_audio.wav", sr_base, audio_base)
    generator.save_audio("improved_audio.wav", sr_imp, audio_imp)

    prompt_file_content = f"""**Base Prompt**
`{baseline_prompt}`

**Improved Prompt**
`{improved_prompt}`

**Metrics**
{results_md.strip()}
"""
    with open("prompt.md", "w") as f:
        f.write(prompt_file_content)

    return "baseline_audio.wav", "improved_audio.wav", baseline_prompt, improved_prompt, results_md, "prompt.md"

with gr.Blocks(title="Video Game Race Music Generator AI") as demo:
    gr.Markdown("# Video Game Race Music Generator AI")
    gr.Markdown("Describe a new video game race, and the AI will generate retro 16-bit music tailored to their culture.")
    
    with gr.Row():
        with gr.Column():
            desc_input = gr.Textbox(label="Race Description", placeholder="e.g., An ancient race of tree-dwellers, strong in magic but physically weak, secretive and melodic.", lines=3)
            rev_input = gr.Textbox(label="Revisions / Tweaks", placeholder="e.g., make it faster and add drums", lines=1)
            duration_slider = gr.Slider(minimum=5, maximum=30, value=10, step=1, label="Duration (seconds)")
            generate_btn = gr.Button("Generate & Evaluate", variant="primary")
            
        with gr.Column():
            gr.Markdown("### Evaluation Results")
            results_out = gr.Markdown("Results will appear here.")
            download_prompt_out = gr.File(label="Download prompt.md")
            
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Baseline Generation")
            base_prompt_out = gr.Textbox(label="Baseline Prompt", interactive=False)
            base_audio_out = gr.Audio(label="Baseline Audio", type="filepath")
            
        with gr.Column():
            gr.Markdown("### Improved Generation")
            imp_prompt_out = gr.Textbox(label="Improved Prompt", interactive=False)
            imp_audio_out = gr.Audio(label="Improved Audio", type="filepath")
            
    generate_btn.click(
        fn=generate_and_evaluate,
        inputs=[desc_input, rev_input, duration_slider],
        outputs=[base_audio_out, imp_audio_out, base_prompt_out, imp_prompt_out, results_out, download_prompt_out]
    )

if __name__ == "__main__":
    demo.launch()

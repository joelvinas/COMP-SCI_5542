import argparse
from music_generator import VideoGameMusicGenerator
from evaluator import MusicEvaluator

def main():
    parser = argparse.ArgumentParser(description="Video Game Race Music Generator AI (CLI)")
    parser.add_argument("--description", type=str, required=True, help="Description of the video game race.")
    parser.add_argument("--revisions", type=str, default="", help="Optional revisions/tweaks.")
    parser.add_argument("--duration", type=int, default=10, help="Duration of generated music in seconds.")
    args = parser.parse_args()

    print("Initializing Generator...")
    generator = VideoGameMusicGenerator()
    print("Initializing Evaluator...")
    evaluator = MusicEvaluator()

    baseline_prompt, improved_prompt = generator.create_prompts(args.description, args.revisions)

    print(f"\n--- Baseline Generation ---")
    print(f"Prompt: {baseline_prompt}")
    sr_base, audio_base = generator.generate_music(baseline_prompt, args.duration)
    generator.save_audio("output_baseline.wav", sr_base, audio_base)
    print("Saved output_baseline.wav")

    print(f"\n--- Improved Generation ---")
    print(f"Prompt: {improved_prompt}")
    sr_imp, audio_imp = generator.generate_music(improved_prompt, args.duration)
    generator.save_audio("output_improved.wav", sr_imp, audio_imp)
    print("Saved output_improved.wav")

    print("\n--- Evaluation ---")
    print("Evaluating Baseline...")
    metrics_base = evaluator.evaluate_all(audio_base, sr_base, baseline_prompt)
    print("Evaluating Improved...")
    metrics_imp = evaluator.evaluate_all(audio_imp, sr_imp, improved_prompt)

    print("\n--- Results ---")
    print(f"{'Metric':<15} | {'Baseline':<10} | {'Improved':<10}")
    print("-" * 40)
    for k in metrics_base.keys():
        print(f"{k.capitalize():<15} | {metrics_base[k]:<10.4f} | {metrics_imp[k]:<10.4f}")

if __name__ == "__main__":
    main()

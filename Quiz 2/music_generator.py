import torch
import scipy.io.wavfile
import numpy as np
from transformers import AutoProcessor, MusicgenForConditionalGeneration

class VideoGameMusicGenerator:
    def __init__(self, model_id="facebook/musicgen-medium"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading {model_id} on {self.device}...")
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = MusicgenForConditionalGeneration.from_pretrained(model_id).to(self.device)
        self.sample_rate = self.model.config.audio_encoder.sampling_rate

    def create_prompts(self, race_description, revisions=""):
        baseline_prompt = race_description
        if revisions:
            baseline_prompt += f" {revisions}"
            
        improved_prompt = (
            f"16-bit SNES style video game music for a race characterized by: {race_description}. "
            "chiptune, synthetic instruments, loopable, rhythmic, atmospheric, retro gaming soundtrack."
        )
        if revisions:
            improved_prompt += f" Make it: {revisions}."
            
        return baseline_prompt, improved_prompt

    def generate_music(self, prompt, duration=10):
        # MusicGen uses 50 tokens per second of audio
        max_new_tokens = int(duration * 50) 
        
        inputs = self.processor(
            text=[prompt],
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            #audio_values = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
            # Adding guidance_scale and do_sample is CRITICAL
            audio_values = self.model.generate(
                **inputs, 
                max_new_tokens=max_new_tokens,
                do_sample=True,           # Enables creative variation
                guidance_scale=3.5,       # Forces the model to follow the text
                temperature=1.0,          # Standard randomness
                top_k=250                 # Limits noise in the sampling pool
            )            
            
        # audio_values is of shape (batch, channels, length)
        audio_data = audio_values[0, 0].cpu().numpy()

        # NORMALIZATION: Prevents the clipping/distortion you heard
        if np.abs(audio_data).max() > 0:
            audio_data = audio_data / np.abs(audio_data).max()  

        return self.sample_rate, audio_data

    def save_audio(self, filename, sample_rate, audio_data):
        scipy.io.wavfile.write(filename, rate=sample_rate, data=audio_data)

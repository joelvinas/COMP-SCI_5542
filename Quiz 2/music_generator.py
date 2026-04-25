import torch
import scipy.io.wavfile
import numpy as np
import re
from transformers import AutoProcessor, MusicgenForConditionalGeneration
from prompt_translator import PromptTranslator

class VideoGameMusicGenerator:
    def __init__(self, model_id="facebook/musicgen-medium"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading {model_id} on cpu (to save VRAM)...")
        self.processor = AutoProcessor.from_pretrained(model_id)
        # Initialize model on CPU by default to save VRAM
        self.model = MusicgenForConditionalGeneration.from_pretrained(model_id).to("cpu")
        self.sample_rate = self.model.config.audio_encoder.sampling_rate
        self.translator = PromptTranslator()

    def load_to_gpu(self):
        if self.device == "cuda":
            print("Moving MusicGen to GPU...")
            self.model = self.model.to("cuda")

    def offload_to_cpu(self):
        if self.device == "cuda":
            print("Moving MusicGen to CPU and clearing VRAM...")
            self.model = self.model.to("cpu")
            torch.cuda.empty_cache()
        self.translator = PromptTranslator()

    def create_prompts(self, race_description, revisions=""):
        translation = self.translator.translate(race_description)
        
        baseline_prompt = translation.get("baseline_prompt", f"16-bit SNES chiptune {race_description}")
        improved_params = translation.get("improved_parameters", {})
        
        params_str = ", ".join([f"{k}: {v}" for k, v in improved_params.items()])
        
        if revisions:
            baseline_prompt += f" {revisions}"
            
        improved_prompt = f"{baseline_prompt} {params_str}"
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

class AceStepMusicGenerator:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dit_handler = None
        self.is_initialized = False

    def initialize(self):
        if not self.is_initialized:
            print("Initializing ACE-Step Turbo model...")
            import sys
            import os
            # Ensure acestep is in path
            current_dir = os.path.dirname(os.path.abspath(__file__))
            if current_dir not in sys.path:
                sys.path.insert(0, current_dir)
            
            from acestep.handler import AceStepHandler
            self.dit_handler = AceStepHandler()
            
            # AceStep natively supports VRAM offloading
            self.dit_handler.initialize_service(
                project_root=current_dir,
                config_path="acestep-v15-turbo",
                device=self.device,
                offload_to_cpu=True,
                offload_dit_to_cpu=True
            )
            self.is_initialized = True

    def load_to_gpu(self):
        # Natively handled by AceStepHandler
        pass

    def offload_to_cpu(self):
        if self.is_initialized and self.device == "cuda":
            print("ACE-Step offloads automatically, just clearing cache...")
            torch.cuda.empty_cache()

    def generate_music(self, prompt, duration=10):
        self.initialize()
        from acestep.inference import GenerationParams, GenerationConfig, generate_music
        
        # We disable LLM thinking to save VRAM and latency since we use the Prompt Translator
        params = GenerationParams(
            task_type="text2music",
            caption=prompt,
            duration=duration,
            thinking=False, 
            inference_steps=8
        )
        config = GenerationConfig(
            batch_size=1,
            audio_format="wav"
        )
        
        print("Starting ACE-Step generation...")
        result = generate_music(self.dit_handler, None, params, config)
        
        # Handle various possible return types from generate_music
        audio_path = ""
        if result:
            if hasattr(result, "audios") and result.audios and len(result.audios) > 0:
                audio_path = result.audios[0].get("path", "")
            elif isinstance(result, dict) and "audios" in result and len(result["audios"]) > 0:
                audio_path = result["audios"][0].get("path", "")
            elif isinstance(result, tuple) and len(result) > 0:
                # If it's bizarrely a tuple, maybe the first element is the path or the dict
                if isinstance(result[0], str) and result[0].endswith(".wav"):
                    audio_path = result[0]
                elif isinstance(result[0], dict) and "path" in result[0]:
                    audio_path = result[0].get("path", "")
                    
        if audio_path and os.path.exists(audio_path):
            try:
                sample_rate, audio_data = scipy.io.wavfile.read(audio_path)
                return sample_rate, audio_data
            except Exception as e:
                print(f"Failed to read generated ACE-Step audio: {e}")
        else:
            print("ACE-Step audio generation failed. Result output:")
            if result:
                if hasattr(result, "error"):
                    print(f"Error: {result.error}")
                if hasattr(result, "status_message"):
                    print(f"Status: {result.status_message}")
                elif isinstance(result, dict):
                    print(f"Dict result: {result}")
        
        return 44100, np.zeros(44100)

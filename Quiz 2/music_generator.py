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

        # --- MusicGen prompts (natural language) ---
        baseline_prompt = translation.get("baseline_prompt", f"MIDI-orchestral {race_description}")
        improved_params = translation.get("improved_parameters", {})

        params_str = ", ".join([f"{k}: {v}" for k, v in improved_params.items()])

        if revisions:
            baseline_prompt += f" {revisions}"

        improved_prompt = f"{baseline_prompt} {params_str}"
        if revisions:
            improved_prompt += f" Make it: {revisions}."

        # --- ACE-Step prompt (tag-based) ---
        # Base tags come from the translator; improved_parameters are appended as tags,
        # mirroring the way improved_prompt is built from baseline_prompt + params.
        ace_step_base = translation.get(
            "ace_step_tags",
            f"MIDI-orchestral, {race_description}, high fidelity, clean mix, professional mastering, pure melodic tones"
        )
        if revisions:
            ace_step_base += f", {revisions}"

        ace_step_prompt = ace_step_base
        if improved_params:
            ace_tag_params = ", ".join([f"{v}" for v in improved_params.values()])
            ace_step_prompt = f"{ace_step_base}, {ace_tag_params}"

        bpm = improved_params.get("BPM")
        key = improved_params.get("Key")

        return baseline_prompt, improved_prompt, ace_step_prompt, bpm, key

    def generate_music(self, prompt, duration=10):
        # MusicGen uses 50 tokens per second of audio
        max_new_tokens = int(duration * 50) 
        
        inputs = self.processor(
            text=[prompt],
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
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
        self.llm_handler = None
        self.is_initialized = False

    def initialize(self):
        if not self.is_initialized:
            print("Initializing ACE-Step SFT model...")
            import sys
            import os
            # Ensure acestep is in path
            current_dir = os.path.dirname(os.path.abspath(__file__))
            if current_dir not in sys.path:
                sys.path.insert(0, current_dir)
            
            from acestep.handler import AceStepHandler
            from acestep.llm_inference import LLMHandler
            self.dit_handler = AceStepHandler()
            self.llm_handler = LLMHandler()
            
            # AceStep natively supports VRAM offloading
            self.dit_handler.initialize_service(
                project_root=current_dir,
                config_path="acestep-v15-sft",
                device=self.device,
                offload_to_cpu=True,
                offload_dit_to_cpu=True
            )
            
            print("Initializing ACE-Step LLM...")
            from acestep.model_downloader import get_checkpoints_dir
            ckpt_dir = str(get_checkpoints_dir())
            self.llm_handler.initialize(
                checkpoint_dir=ckpt_dir,
                lm_model_path="acestep-5Hz-lm-1.7B", 
                backend="pt",
                device=self.device,
                offload_to_cpu=True,
                dtype=torch.float32
            )
            
            self.is_initialized = True

    def load_to_gpu(self):
        # Natively handled by AceStepHandler
        pass

    def offload_to_cpu(self):
        if self.is_initialized and self.device == "cuda":
            print("ACE-Step offloads automatically, just clearing cache...")
            torch.cuda.empty_cache()

    def generate_music(self, prompt, duration=10, bpm=None, keyscale=None):
        self.initialize()
        from acestep.inference import GenerationParams, GenerationConfig, generate_music

        # Parse bpm as int if provided
        bpm_val = None
        if bpm:
            try:
                import re
                bpm_match = re.search(r'\d+', str(bpm))
                if bpm_match:
                    bpm_val = int(bpm_match.group(0))
            except Exception as e:
                print(f"Failed to parse bpm: {bpm}, error: {e}")

        # Re-enable LLM thinking to generate audio codes for the DiT
        params = GenerationParams(
            task_type="text2music",
            caption=prompt,
            instrumental=True,        # Use the explicit boolean flag to prevent vocals
            duration=duration,
            bpm=bpm_val,
            keyscale=str(keyscale) if keyscale else "",
            thinking=True,
            inference_steps=50,
            guidance_scale=7.0,       # Restore to standard SFT CFG
            dcw_enabled=False         # Disable Wavelet Correction for CPU stability
        )
        config = GenerationConfig(
            batch_size=1,
            audio_format="wav"
        )
        
        print("Starting ACE-Step generation...")
        result = generate_music(self.dit_handler, self.llm_handler, params, config)
        
        # Handle various possible return types from generate_music
        if result:
            if hasattr(result, "audios") and result.audios and len(result.audios) > 0:
                audio_dict = result.audios[0]
                if isinstance(audio_dict, dict):
                    audio_tensor = audio_dict.get("tensor")
                    if audio_tensor is not None:
                        # tensor is [channels, samples], convert to numpy [samples] or [samples, channels]
                        sample_rate = audio_dict.get("sample_rate", 44100)
                        audio_data = audio_tensor.cpu().numpy() #Extract to numpy without transposing first

                        # Force to 1D (Mono) if it's a standard single-channel output
                        if audio_data.ndim > 1:
                            # if [channels, samples], take the first channel
                            if audio_data.shape[0] < audio_data.shape[1]:
                                audio_data = audio_data[0] #Take the first channel
                            else:
                                # if [sample, channels], take the first channel
                                audio_data = audio_data[:,0]

                        # Print raw audio properties before normalization for debugging
                        raw_max = np.abs(audio_data).max()
                        print(f"DEBUG: ACE-Step raw audio max: {raw_max}, min: {audio_data.min()}")

                        # Add NORMALIZATION to prevent digital clipping
                        if raw_max > 0:
                            audio_data = audio_data / raw_max

                        # Ensure correct shape (remove extra channel dimension) 
                        if audio_data.ndim == 2 and audio_data.shape[1] == 1:
                            audio_data = audio_data.squeeze(1)
                        return sample_rate, audio_data
                    
                    # Fallback to reading file if tensor is missing but path exists
                    audio_path = audio_dict.get("path", "")
                    if audio_path and os.path.exists(audio_path):
                        try:
                            sample_rate, audio_data = scipy.io.wavfile.read(audio_path)
                            return sample_rate, audio_data
                        except Exception as e:
                            print(f"Failed to read generated ACE-Step audio file: {e}")

            print("ACE-Step audio generation failed. Result output:")
            if hasattr(result, "error"):
                print(f"Error: {result.error}")
            if hasattr(result, "status_message"):
                print(f"Status: {result.status_message}")
            elif isinstance(result, dict):
                print(f"Dict result: {result}")
        
        return 44100, np.zeros(44100)

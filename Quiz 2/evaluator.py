import numpy as np
import librosa
import torch
from transformers import ClapModel, ClapProcessor

class MusicEvaluator:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model_id = "laion/clap-htsat-unfused"
        print(f"Loading {model_id} on {self.device}...")
        self.processor = ClapProcessor.from_pretrained(model_id)
        self.model = ClapModel.from_pretrained(model_id).to(self.device)
        # resample to 48000 for CLAP
        self.clap_sr = 48000

    def compute_clap_similarity(self, audio, sr, text):
        # Librosa resampling if needed
        if sr != self.clap_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.clap_sr)
            
        inputs = self.processor(text=[text], audio=[audio], return_tensors="pt", padding=True, sampling_rate=self.clap_sr).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        # Cosine similarity
        audio_embeds = outputs.audio_embeds
        text_embeds = outputs.text_embeds
        cosine_sim = torch.nn.functional.cosine_similarity(audio_embeds, text_embeds).item()
        return cosine_sim

    def evaluate_quality(self, audio, sr):
        # Approximate quality by spectral contrast variance (complexity)
        S = np.abs(librosa.stft(audio))
        contrast = librosa.feature.spectral_contrast(S=S, sr=sr)
        return float(np.mean(np.var(contrast, axis=1)))

    def evaluate_alignment(self, audio, sr, aggregate_prompt):
        return self.compute_clap_similarity(audio, sr, aggregate_prompt)

    def evaluate_realism(self, audio, sr):
        # How close is it to "retro 16-bit video game music" vs "modern orchestral music"
        sim_retro = self.compute_clap_similarity(audio, sr, "retro 16-bit chiptune video game music")
        sim_modern = self.compute_clap_similarity(audio, sr, "modern realistic orchestral music")
        # Scaled to be mostly positive
        return max(0.0, sim_retro - sim_modern + 0.5)

    def evaluate_creativity(self, audio, sr, original_prompt):
        # Measure divergence from just the prompt or spectral variety
        cent = librosa.feature.spectral_centroid(y=audio, sr=sr)
        return float(np.var(cent)) / 1000000.0 # scale down

    def evaluate_loopability(self, audio, sr, window_sec=1.0):
        # Compare beginning and end
        window_samples = int(sr * window_sec)
        if len(audio) < window_samples * 2:
            return 0.0
        
        start_segment = audio[:window_samples]
        end_segment = audio[-window_samples:]
        
        # normalized cross correlation
        corr = np.correlate(start_segment, end_segment)
        norm1 = np.linalg.norm(start_segment)
        norm2 = np.linalg.norm(end_segment)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        score = corr[0] / (norm1 * norm2)
        return float(score)

    def evaluate_all(self, audio, sr, aggregate_prompt):
        if len(audio) == 0:
             return {"quality": 0.0, "alignment": 0.0, "realism": 0.0, "creativity": 0.0, "loopability": 0.0}
        
        # Ensure audio is mono for librosa and CLAP
        if audio.ndim == 2:
            if audio.shape[1] == 2:
                audio = np.mean(audio, axis=1)
            else:
                audio = audio.squeeze()
        
        return {
            "quality": self.evaluate_quality(audio, sr),
            "alignment": self.evaluate_alignment(audio, sr, aggregate_prompt),
            "realism": self.evaluate_realism(audio, sr),
            "creativity": self.evaluate_creativity(audio, sr, aggregate_prompt),
            "loopability": self.evaluate_loopability(audio, sr)
        }

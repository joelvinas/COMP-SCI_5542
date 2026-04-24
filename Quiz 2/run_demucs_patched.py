import sys
import torchaudio
import soundfile as sf
import torch

def _mock_load(filepath, *args, **kwargs):
    data, samplerate = sf.read(filepath, dtype='float32')
    if data.ndim == 1:
        tensor = torch.from_numpy(data).unsqueeze(0)
    else:
        tensor = torch.from_numpy(data).t()
    return tensor, samplerate

def _mock_save(filepath, src, sample_rate, *args, **kwargs):
    if src.ndim == 1:
        data = src.numpy()
    else:
        data = src.t().numpy()
    sf.write(filepath, data, sample_rate)

torchaudio.load = _mock_load
torchaudio.save = _mock_save

from demucs.separate import main

if __name__ == "__main__":
    main()

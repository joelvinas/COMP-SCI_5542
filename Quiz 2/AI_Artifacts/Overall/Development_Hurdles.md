# Development Hurdles

Based on the documentation and development history, here are the four greatest technical challenges encountered while building the application (excluding ACE-Step). 

These challenges were particularly difficult because they couldn't be fixed by simply writing better logic; they required "hacking" around broken third-party libraries, legacy dependencies, and OS-level limitations.

### 1. The Python 3.13 vs. Legacy Audio Dependency Hell
Because the environment was using bleeding-edge Python 3.13.5, it lacked the legacy `distutils` build tools. However, the Spotify `basic-pitch` library strictly required an outdated version of `resampy`, which in turn attempted to build an ancient version of `numpy` from source. This caused fatal crashes during the fundamental `pip install` phase. 
* **The Fix:** We had to bypass the strict dependency resolver entirely. We wrote a custom `install.py` script that installed the modern dependencies first, and then force-installed `basic-pitch` using the `--no-deps` flag to trick it into using the modern Python 3.13 compatible packages.

### 2. Windows FFmpeg / Torchaudio DLL Crash
During stem separation, Torchaudio threw a fatal crash looking for `libtorchcodec_core8.dll`. In newer versions of PyTorch, audio decoding is hardcoded to use `torchcodec`, which requires full FFmpeg C++ shared libraries to be manually installed in the Windows system PATH. 
* **The Fix:** To prevent users from having to manually configure Windows environment variables, we wrote a script (`run_demucs_patched.py`) that performed an in-memory "monkey-patch". It dynamically overrode `torchaudio.load()` and `torchaudio.save()` to route through the native Python `soundfile` library instead, flawlessly bypassing the FFmpeg requirement.

### 3. TensorFlow 2.21 Legacy Optimizer Bug
Even after `basic-pitch` was successfully installed, it crashed instantly upon trying to transcribe audio with the error: `'_UserObject' object has no attribute 'add_slot'`. The issue was that the default `basic-pitch` model was built on a 2022 TensorFlow SavedModel format that contained legacy optimizer slots. Modern TensorFlow (2.21) completely dropped support for this older formatting style.
* **The Fix:** We hacked the library's internal logic. By appending `.onnx` to the end of the hardcoded `ICASSP_2022_MODEL_PATH` in the code, we forced the application to completely ignore TensorFlow and instead load the model natively using the much more robust `onnxruntime` framework.

### 4. The "Missing" Demucs API
When building the transcription engine, the application crashed with `ModuleNotFoundError: No module named 'demucs.api'`. Despite the official documentation showing how to use the `.api` wrapper, it was never actually bundled into the stable PyPI release (`demucs 4.0.1`). 
* **The Fix:** Instead of trying to force a complex `git+https` source installation, we completely rewrote the `MidiTranscriptionEngine`. We built it to bypass the Python API entirely and instead cleanly invoke the `demucs` command-line interface via a background Python `subprocess`, dynamically harvesting the separated WAV stems from the output directory once the process finished.

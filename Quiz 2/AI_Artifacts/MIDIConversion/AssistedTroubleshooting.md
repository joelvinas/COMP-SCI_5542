# MIDI Transcription Engine - Assisted Troubleshooting Log

This document serves as an architectural post-mortem and troubleshooting log for the integration of the MIDI Transcription Engine into the `music_generator.py` application. 

The primary goal of this expansion was to accept the audio output of Facebook's `MusicGen` and transcribe it into discrete, multi-track MIDI stems (`stems.mid`) and MusicXML sheet music (`sheet_music.xml`) for use in professional DAWs.

Below are the major blockers encountered during the deployment to a Python 3.13.5 environment, along with their respective resolutions.

---

### 1. The `basic-pitch` & `numpy` Dependency Hell
**The Issue:**
During initial installation via `requirements.txt`, the build failed abruptly with a subprocess `ImpImporter` crash. This occurred because `basic-pitch` strictly requires an outdated version of the `resampy` library (`<0.4.3`). That older version of `resampy` forces pip to build an outdated version of `numpy`. However, because the environment is running a bleeding-edge version of Python (3.13.5), the legacy build tools needed to compile that old `numpy` version from source (`distutils`) no longer exist. 

**The Resolution:**
We bypassed the strict dependency resolver entirely. 
1. We removed `basic-pitch` from the main `requirements.txt`.
2. We manually added its underlying dependencies (`pretty_midi`, `mir_eval`, and crucially, `resampy>=0.4.3`) to force pip to grab the modern versions that are compatible with Python 3.13.
3. We created a custom `install.py` script that first installs the requirements normally, and then force-installs `basic-pitch` using the `--no-deps` flag to circumvent its outdated dependency tree restrictions.

---

### 2. Missing Demucs API in PyPI
**The Issue:**
When attempting to import Demucs (`import demucs.api`), the application crashed with `ModuleNotFoundError: No module named 'demucs.api'`. While the `.api` wrapper exists on the master branch of the Demucs GitHub repository, it was never actually bundled into the stable PyPI release (`demucs 4.0.1`). 

**The Resolution:**
Rather than downgrading or attempting a complex `git+https` installation (which failed due to `git` missing from the system PATH), we rewrote the `MidiTranscriptionEngine`. We bypassed the python API entirely and cleanly invoked the `demucs` command-line interface via a background Python `subprocess`, dynamically harvesting the separated WAV stems from the output directory.

---

### 3. TorchCodec / FFmpeg Shared Library Failure
**The Issue:**
During the `demucs` subprocess execution, Torchaudio threw a fatal crash: `ImportError: TorchCodec is required for load_with_torchcodec` and `OSError: Could not load this library: libtorchcodec_core8.dll`. 
In PyTorch 2.11+, `torchaudio.load()` is hardcoded to depend on `torchcodec` for audio decoding. On Windows, `torchcodec` requires full-shared FFmpeg `C++` DLLs to be manually installed and present in the system PATH. 

**The Resolution:**
To avoid forcing the user to manually configure FFmpeg through Windows system variables, we dynamically monkey-patched `torchaudio` in-memory. We created a lightweight wrapper script (`run_demucs_patched.py`) that overrides `torchaudio.load` and `torchaudio.save` to utilize the native Python `soundfile` backend instead. The transcription engine was updated to route all Demucs jobs through this patched wrapper, flawlessly bypassing the FFmpeg requirement.

---

### 4. Gemini 404 NOT_FOUND Error
**The Issue:**
The Prompt Translator returned a 404 Error: `models/gemini-3.1-pro is not found for API version v1beta`.

**The Resolution:**
Although colloquially referred to as "Gemini 3.1 Pro", the official internal endpoint string required by Google's API for that specific version is `gemini-3.1-pro-preview`. The string was updated in `prompt_translator.py`.

---

### 5. Basic-Pitch API Signature Update
**The Issue:**
The application crashed on stem transcription with `predict_and_save() missing 1 required positional argument: 'model_or_model_path'`. The newer version of `basic-pitch` (0.4.0) modified its inference signature to strictly require the model path.

**The Resolution:**
Updated `midi_engine.py` to import `ICASSP_2022_MODEL_PATH` from the `basic_pitch` library and pass it explicitly into the function call.

---

### 6. TensorFlow Legacy Optimizer Bug
**The Issue:**
Even with the correct model path, transcription failed with: `AttributeError("'_UserObject' object has no attribute 'add_slot'")`. 
The default `basic-pitch` model format (a 2022 TensorFlow SavedModel) contains legacy optimizer slots. Bleeding-edge TensorFlow 2.21 is incompatible with this older formatting style and crashes during the load sequence.

**The Resolution:**
We discovered that `basic-pitch` includes built-in fallback logic for ONNX. By appending `.onnx` to the end of the `ICASSP_2022_MODEL_PATH` in `midi_engine.py` (i.e., `model_or_model_path=f"{ICASSP_2022_MODEL_PATH}.onnx"`), we forced `basic-pitch` to completely ignore TensorFlow and instead load the model natively using the much more robust `onnxruntime` framework. 

---

### 7. Gradio Application State Caching
**The Issue:**
Despite applying the ONNX fix, the UI continued to silently fail and drop the `sheet_music.xml` file generation. 

**The Resolution:**
Gradio does not automatically hot-reload underlying python module dependencies (like `midi_engine.py`) while the server is actively running. The terminal process was running a 21-minute old version of the code in memory. A manual restart of the `python app.py` terminal command refreshed the state and allowed the fixes to execute.

---

### 8. Secure Hugging Face Authentication
**The Issue:**
The Hugging Face models (`facebook/musicgen-medium` and `laion/clap-htsat-unfused`) were downloading unauthenticated, risking potential rate limits or timeouts.

**The Resolution:**
Updated `app.py` to load `.env` variables immediately on boot, explicitly grabbing the `HF_TOKEN` and passing it to `huggingface_hub.login()`. This ensures that all subsequent model initializations are securely authenticated without hardcoding credentials into the source code.

---

### 9. TensorFlow Lite Deprecation Warnings
**The Issue:**
During inference, `basic-pitch` imports `tensorflow.lite` as a fallback, which triggered a massive block of verbose `UserWarning` traces about the interpreter's deprecation in TF 2.20. It also printed root logger warnings about missing `coremltools` and `tflite-runtime` packages.

**The Resolution:**
We dynamically filtered out these noisy warnings by importing the `warnings` and `logging` modules at the top of `midi_engine.py`, explicitly ignoring all `UserWarning` traces from the `tensorflow` module, and setting the root logger to strictly `ERROR` level. This completely cleans up the terminal output during transcription.

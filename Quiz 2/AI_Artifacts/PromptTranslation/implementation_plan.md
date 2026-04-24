# PromptTranslator Implementation Plan

This plan details the implementation of a new dynamic `PromptTranslator` class that uses the Gemini API to intelligently convert narrative and cultural race descriptions into structured, generation-ready musical prompts, fulfilling the zero-hardcoding and semantic mapping requirements.

## User Review Required

> [!IMPORTANT]
> Please review the chosen approach for the Gemini API integration and the JSON parsing strategy. The plan assumes the use of the `google-genai` (or `google-generativeai`) Python SDK and requires an API key to be available in the environment.

## Open Questions

> [!WARNING]
> 1. **Gemini SDK Version:** Are you currently using the `google-generativeai` package or the newer `google-genai` package for API access? I will add `google-generativeai` to the requirements if none is specified.
> 2. **API Key Authentication:** Do you already have `GEMINI_API_KEY` set in your environment variables, or would you like me to add an input field in the Gradio app for the user to provide it?
> 3. **Prompt Return Structure:** The requirements state the output signature must return a JSON object with `baseline_prompt` and `improved_parameters`. Should `create_prompts` then construct the final string by concatenating `baseline_prompt` with the values in `improved_parameters`?

## Proposed Architecture

### 1. The `PromptTranslator` Class (`prompt_translator.py`)
- **Initialization:** Loads the Gemini model (e.g., `gemini-1.5-pro` or `gemini-3.1-pro` depending on availability) with the specified System Instructions.
- **System Instructions:** Configured exactly as requested:
  - **Role:** Music Theory Specialist for 16-bit Video Game Music.
  - **Task:** Translate cultural flavor text into a MusicGen Signature.
  - **Rules:** Semantic Mapping (Size -> BPM/Freq, Env -> Acoustics, Culture -> Instruments). "Noise-Strip" filtering. Avoidance rules for literal sound words ("laughter", "coins", "trash").
  - **Formatting:** Return structured JSON. Prefix outputs with "16-bit SNES chiptune".
- **`translate(raw_prompt)` Method:** 
  - Sends the raw user text to the Gemini API.
  - Parses the returned JSON.
  - Returns a dictionary with `baseline_prompt` and `improved_parameters` (BPM, Key, Lead).

### 2. Integration with `music_generator.py`
- Modify `create_prompts(self, race_description, revisions="")` to instantiate/call the `PromptTranslator`.
- The `baseline_prompt` will be constructed from the Gemini output's `baseline_prompt`.
- The `improved_prompt` will concatenate the `baseline_prompt` with the `improved_parameters` (e.g., "{baseline} {BPM} BPM, Key of {Key}, {Lead} lead.").

### 3. Application Updates
- Update `requirements.txt` to include `google-generativeai`.

## Proposed Changes

### `requirements.txt`
#### [MODIFY] `requirements.txt`
Add `google-generativeai` (or `google-genai`) and `pydantic` (for structured output enforcement).

### `prompt_translator.py`
#### [NEW] `prompt_translator.py`
Create the standalone `PromptTranslator` class containing the Gemini integration logic.

### `music_generator.py`
#### [MODIFY] `music_generator.py`
Import `PromptTranslator` and refactor `create_prompts` to use the dynamic JSON output from the Gemini API instead of the previous hardcoded regex logic.

## Verification Plan

### Automated/Manual Tests
- Ensure `GEMINI_API_KEY` is set.
- Run `python cli.py --description "A massive race of stone-golems living in fiery volcanoes"`
- Verify that the Gemini API returns a valid JSON structure avoiding literal keywords and instead mapping "massive/stone" to heavy bass and "fiery volcanoes" to warm, distorted acoustics.
- Confirm `app.py` successfully runs and generates audio without crashing due to prompt formatting.

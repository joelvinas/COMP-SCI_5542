import os
import json
from google import genai
from pydantic import BaseModel, Field
from dotenv import load_dotenv

class ImprovedParameters(BaseModel):
    BPM: str = Field(description="BPM of the track")
    Key: str = Field(description="Musical key of the track")
    Lead: str = Field(description="Lead instrument of the track")

class PromptTranslatorOutput(BaseModel):
    baseline_prompt: str = Field(
        description=(
            "A clean MIDI-orchestral natural-language sentence for MusicGen. "
            "Must be prefixed with 'MIDI-orchestral '. "
            "Example: 'MIDI-orchestral epic tribal cinematic music in a dark ritualistic style, "
            "using driving taiko drums and haunting vocal chants.'"
        )
    )
    ace_step_tags: str = Field(
        description=(
            "A comma-separated tag string optimised for ACE-Step SFT diffusion. "
            "No sentences — only descriptive tags. Include genre, mood, style, and primary instruments. "
            "Do NOT include BPM, Key, or Lead here; those are added separately from improved_parameters. "
            "Example: 'MIDI-orchestral, epic, tribal, cinematic, dark, ritualistic, "
            "taiko drums, cello ensemble, haunting vocal chants, high fidelity, clean mix'"
        )
    )
    improved_parameters: ImprovedParameters = Field(
        description="Precise musical parameters derived from the race's energy and environment."
    )

class PromptTranslator:
    def __init__(self):
        # Load environment variables from a local .env file if it exists
        load_dotenv()
        
        api_key = os.environ.get("GEMINI_API_KEY")
        
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found. Please set it in your environment variables or a .env file.")
            
        self.client = genai.Client(api_key=api_key)

        self.system_instruction = (
            "Role: You are a Lead Audio Director specializing in high-fidelity Video Game Music.\n"
            "Task: Translate physical race lore into musical prompts for TWO different AI models: "
            "MusicGen (natural language) and ACE-Step SFT (tag-based).\n\n"

            "PHASE 1: MUSICAL MAPPING LOGIC:\n"
            "1. PHYSICAL -> FREQUENCY: Translate size/power to pitch. Tiny = High-register. Large = Deep resonant sub-bass.\n"
            "2. ENVIRONMENT -> TEXTURE: Forest = Shimmering pads. Cavern = Low-pass filters with long delays.\n"
            "3. CULTURE -> RHYTHM: Artisans = Syncopated staccato. Warriors = Driving low-mid percussion.\n"
            "4. NOISE-STRIP: PURGE all non-musical nouns (e.g., 'hair', 'snouts', 'houses'). "
            "Replace them with instrumental equivalents.\n\n"

            "PHASE 2: MusicGen OUTPUT (baseline_prompt):\n"
            "Write a single clean natural-language sentence for MusicGen. "
            "Use an 'instrument_first' or 'mood_emphasized' Prompt2MusicBench template. "
            "Prefix with 'MIDI-orchestral '. Do NOT include raw BPM/Key/Lead values here.\n"
            "Example: 'MIDI-orchestral dark ritualistic tribal music, led by deep taiko drums and low cello drones.'\n\n"

            "PHASE 3: ACE-Step SFT OUTPUT (ace_step_tags):\n"
            "Write a comma-separated list of descriptive tags — NO full sentences. "
            "Tags should cover: genre, mood, style adjectives, texture, and primary instruments. "
            "Always end with quality tags: 'high fidelity, clean mix, professional mastering'. "
            "Do NOT include BPM, Key, or Lead values here — those are appended automatically from improved_parameters.\n"
            "Example: 'MIDI-orchestral, epic, tribal, cinematic, dark, ritualistic, taiko drums, "
            "deep cello ensemble, low drones, high fidelity, clean mix, professional mastering'\n\n"

            "PHASE 4: IMPROVED PARAMETERS:\n"
            "Extract precise BPM (integer), musical Key (e.g. 'D minor'), and Lead instrument description. "
            "These are appended to both the MusicGen improved prompt and the ACE-Step tag prompt.\n\n"

            "OUTPUT REQUIREMENT:\n"
            "Return a structured JSON with fields: baseline_prompt, ace_step_tags, improved_parameters."
        )

    def translate(self, raw_prompt: str) -> dict:
        try:
            response = self.client.models.generate_content(
                model="gemini-3.1-flash-lite-preview",
                contents=raw_prompt,
                config=genai.types.GenerateContentConfig(
                    system_instruction=self.system_instruction,
                    response_mime_type="application/json",
                    response_schema=PromptTranslatorOutput,
                    temperature=0.9,
                ),
            )
            return json.loads(response.text)
        except Exception as e:
            print(f"Error during Gemini translation: {e}")
            return {
                "baseline_prompt": f"MIDI-orchestral {raw_prompt}",
                "ace_step_tags": f"MIDI-orchestral, {raw_prompt}, high fidelity, clean mix, professional mastering, pure melodic tones",
                "improved_parameters": {"BPM": "120", "Key": "C Minor", "Lead": "Strings"}
            }

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
    baseline_prompt: str = Field(description="The cleaned musical instruction prefixed with MIDI-orchestral.")
    improved_parameters: ImprovedParameters = Field(description="Parameters derived from the text's energy.")

class PromptTranslator:
    def __init__(self):
        # Load environment variables from a local .env file if it exists
        load_dotenv()
        
        api_key = os.environ.get("GEMINI_API_KEY")
        
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found. Please set it in your environment variables or a .env file.")
            
        self.client = genai.Client(api_key=api_key)
        
        self.system_instruction = (
            "Role: You are a Music Theory Specialist for MIDI-orchestral Video Game Music.\n"
            "Task: Translate cultural flavor text into a MusicGen Signature.\n"
            "Core Logic Requirements:\n"
            "- Zero-Hardcoding Rule: Do not use specific race names (e.g. Goblins, Ogres). Use a Semantic Mapping Pattern to extract attributes.\n"
            "- Physical-to-Musical Extraction:\n"
            "  - Map Size/Power (e.g., 'towering', 'tiny') to Frequency/BPM (e.g., 'Heavy sub-bass' vs 'High-pitched staccato').\n"
            "  - Map Environment (e.g., 'jungle', 'underground') to Acoustics (e.g., 'Short dry reverb' vs 'Spacious dark hall reverb').\n"
            "  - Map Culture (e.g., 'tribal', 'hoarding') to Instrumentation (e.g., 'Woodblocks/Percussion' vs 'Metallic/Found-object synths').\n"
            "- The 'Noise-Strip' Filter: Automatically remove 'clipping-prone' adjectives (e.g., physical colors, specific nouns like 'trash', and abstract actions like 'hoarding') that lack a direct musical signal.\n"
            "Avoidance Rules: Never include the words 'laughter', 'coins', or 'trash'. Replace them with 'Rhythmic pizzicato strings' or 'Randomized woodblock staccato'.\n"
            "Formatting: Prefix all outputs with 'MIDI-orchestral '."
        )

    def translate(self, raw_prompt: str) -> dict:
        try:
            response = self.client.models.generate_content(
                model="gemini-3.1-pro-preview",
                contents=raw_prompt,
                config=genai.types.GenerateContentConfig(
                    system_instruction=self.system_instruction,
                    response_mime_type="application/json",
                    response_schema=PromptTranslatorOutput,
                    temperature=0.7,
                ),
            )
            return json.loads(response.text)
        except Exception as e:
            print(f"Error during Gemini translation: {e}")
            return {
                "baseline_prompt": f"MIDI-orchestral {raw_prompt}",
                "improved_parameters": {"BPM": "120", "Key": "C Minor", "Lead": "Strings"}
            }

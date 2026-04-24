We will be designing and building an AI application in Python for video game music using modern pre-trained foundation models to handle Music Generation with Diffusion Audio.
The system will:
* Prompt the user to give a text prompt that describes a new race for the video game world, to include strengths, weaknesses and quirks of the culture. 
* Using the user prompt, generate a short snippet of loopable video game music (for game systems like the Super Nintendo or Nintendo 64) that embodies the given prompt
* Allow the user to revise the music with subsequent prompts

Suggested Models for use:
•	MusicGen: https://huggingface.co/facebook/musicgen-small
•	AudioCraft GitHub: https://github.com/facebookresearch/audiocraft
•	AudioLDM2: https://huggingface.co/cvssp/audioldm2
•	Stable Audio Open: https://huggingface.co/stabilityai/stable-audio-open-1.0

The system must include:
1. Pretrained Foundation Model
Use Hugging Face or open-source models such as 
•	Whisper Small: https://huggingface.co/openai/whisper-small
•	MusicGen Small: https://huggingface.co/facebook/musicgen-small
•	AudioLDM2: https://huggingface.co/cvssp/audioldm2
•	SpeechT5: https://huggingface.co/microsoft/speecht5_tts

2. Prompt / Input Engineering
better prompts for music generation
style prompts

3. Metrics for Evaluation
Compare baseline vs improved settings to handle the following metrics
* musical quality (melodic coherence, rhythmic precision, harmonic accuracy, and structural integrity)
* alignment (to the aggregate prompt)
* realism (compared to video game music found on platforms pre-2000)
* creativity (divergence from the prompt)
* loopability (how seemless the sound can be looped)
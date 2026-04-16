# pip install diffusers transformers accelerate opencv-python Pillow
# pip install --upgrade torch torchvision sympy
# pip install gTTS
import os

# Mount Local

# Define the target output directory in Local
# The provided URL 'https://drive.google.com/drive/folders/1juw87zkUDP_Wv_rOub9ETncr0fDI_O9r' points to a folder.
# We will use this folder ID to create a path in the mounted drive.
local_output_path = os.path.join('.', 'Store Output') # Output to local directory

# Create the directory if it doesn't exist
os.makedirs(local_output_path, exist_ok=True)

print(f"Local output path set to: {local_output_path}")
from gtts import gTTS
import os

briefing_text = "The cold-medicine aisle is empty. I've updated the planogram to fill the top two shelves with our generic store-brand labels until the shipment arrives."
tts = gTTS(briefing_text)
tts.save(os.path.join(local_output_path, "manager_briefing.mp3"))
# Uses T4 GPU
import os
import json
import torch
import cv2
import numpy as np
from PIL import Image
import urllib.request
from diffusers import StableDiffusionPipeline, StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler

# --- STEP A: Generate the Synthetic Pharmacy Base Layout ---
def generate_pharmacy_base(sd_pipe):
    print("Generating Synthetic Pharmacy Base Layout...")
    # Explicitly defining the brand colors in the base
    base_prompt = ("Interior photo of a modern pharmacy store, red and white color scheme, bright lighting. "
                   "Wide angle view of a central 4-row white metal shelf display with various health goods. "
                   "Photorealistic, 8k.")

    base_image = sd_pipe(base_prompt, num_inference_steps=30, guidance_scale=8.5).images[0]
    if base_image is None:
        raise ValueError("Failed to generate base image.")

    base_image.save(os.path.join(local_output_path, "pharmacy_base_layout.png"))
    print("Pharmacy Base Layout saved.")
    return base_image

# --- STEP B: Update the Main Pipeline to use the Pharmacy Base ---
def main_pharmacy_scenario():
    sd_pipe, controlnet_pipe = setup_pipeline()

    # 1. Generate/Load the Pharmacy Base
    base_image = generate_pharmacy_base(sd_pipe)

    # 2. Extract Structure
    control_image = get_canny_edges(base_image)

    # 3. Define the Pharmacy 'Disruption' Metadata matching the manager's briefing
    pharmacy_metadata = {
        "room_type": "pharmacy aisle",
        "status": "Restocked with Generics",
        "briefing_context": "The cold-medicine aisle was empty. Top two shelves are now filled with generic store-brand labels.",
        "substitutes": ["generic white-label store-brand medicine bottles", "simple healthcare packaging"],
        "constraints": "Maintain the existing red and white store color scheme and shelf structure."
    }

    # 4. Generate the Update
    structured_prompt = generate_structured_prompt(pharmacy_metadata)
    print(f"Updating Planogram: {structured_prompt}")

    # We use a high controlnet_conditioning_scale to ensure the layout/colors don't drift
    updated_planogram = controlnet_pipe(
        structured_prompt,
        image=control_image,
        num_inference_steps=30,
        guidance_scale=8.0,
        controlnet_conditioning_scale=1.0
    ).images[0]

    updated_planogram.save(os.path.join(local_output_path, "pharmacy_updated_planogram.png"))
    print("Updated planogram saved to Local.")
def get_canny_edges(image, low_threshold=100, high_threshold=200):
    image_np = np.array(image)
    image_np = cv2.Canny(image_np, low_threshold, high_threshold)
    image_np = image_np[:, :, None]
    image_np = np.concatenate([image_np, image_np, image_np], axis=2)
    return Image.fromarray(image_np)

def generate_structured_prompt(metadata):
    """
    Strictly maps briefing metadata to prompt while enforcing color and structure constraints.
    """
    prompt = f"A professional photo of a {metadata.get('room_type')}. "

    # Enforce color scheme
    prompt += "The store retains its original red and white color scheme. "

    # Incorporate the briefing context
    if "briefing_context" in metadata:
        prompt += f"Update: {metadata['briefing_context']} "

    substitutes = metadata.get('substitutes', [])
    if substitutes:
        prompt += f"The top shelves now feature {', '.join(substitutes)}. "

    prompt += "Highly organized, clean shelves, bright retail lighting, 8k, photorealistic."
    return prompt

def setup_pipeline():
    print("Loading Stable Diffusion and ControlNet models...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    sd_pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=dtype)
    sd_pipe.scheduler = UniPCMultistepScheduler.from_config(sd_pipe.scheduler.config)

    controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=dtype)
    controlnet_pipe = StableDiffusionControlNetPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=dtype)
    controlnet_pipe.scheduler = UniPCMultistepScheduler.from_config(controlnet_pipe.scheduler.config)

    if device == "cuda":
        sd_pipe.enable_model_cpu_offload()
        controlnet_pipe.enable_model_cpu_offload()
    else:
        sd_pipe = sd_pipe.to(device)
        controlnet_pipe = controlnet_pipe.to(device)

    return sd_pipe, controlnet_pipe

def main():
    main_pharmacy_scenario()

if __name__ == '__main__':
    main()

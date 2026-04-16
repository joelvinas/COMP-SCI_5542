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
from PIL import Image, ImageDraw
import urllib.request
from diffusers import StableDiffusionPipeline, StableDiffusionControlNetPipeline, ControlNetModel, StableDiffusionInpaintPipeline, UniPCMultistepScheduler

# --- Helper Functions ---
def get_canny_edges(image, low_threshold=100, high_threshold=200):
    image_np = np.array(image)
    image_np = cv2.Canny(image_np, low_threshold, high_threshold)
    image_np = image_np[:, :, None]
    image_np = np.concatenate([image_np, image_np, image_np], axis=2)
    return Image.fromarray(image_np)

def create_top_shelf_mask(image, top_percent=0.4):
    """
    Creates a binary mask where the top `top_percent` of the image is white (to be modified)
    and the rest is black (to be preserved).
    """
    width, height = image.size
    mask = Image.new("L", (width, height), 0) # Black (preserve)
    draw = ImageDraw.Draw(mask)
    draw.rectangle([0, 0, width, int(height * top_percent)], fill=255) # White (modify)
    return mask

# --- Pipeline Loading ---
def setup_pipeline():
    print("Loading Stable Diffusion, ControlNet, and Inpainting models...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    # 1. Standard Pipeline (Template)
    sd_pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=dtype)
    sd_pipe.scheduler = UniPCMultistepScheduler.from_config(sd_pipe.scheduler.config)

    # 2. ControlNet Pipeline (Base)
    controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=dtype)
    controlnet_pipe = StableDiffusionControlNetPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=dtype)
    controlnet_pipe.scheduler = UniPCMultistepScheduler.from_config(controlnet_pipe.scheduler.config)

    # 3. Inpainting Pipeline (Updated)
    inpaint_pipe = StableDiffusionInpaintPipeline.from_pretrained("runwayml/stable-diffusion-inpainting", torch_dtype=dtype)
    inpaint_pipe.scheduler = UniPCMultistepScheduler.from_config(inpaint_pipe.scheduler.config)

    if device == "cuda":
        sd_pipe.enable_model_cpu_offload()
        controlnet_pipe.enable_model_cpu_offload()
        inpaint_pipe.enable_model_cpu_offload()
    else:
        sd_pipe = sd_pipe.to(device)
        controlnet_pipe = controlnet_pipe.to(device)
        inpaint_pipe = inpaint_pipe.to(device)

    return sd_pipe, controlnet_pipe, inpaint_pipe

# --- STEP 1: Generate the Empty Synthetic Pharmacy Template ---
def generate_empty_template(sd_pipe):
    print("Generating Empty Pharmacy Template Layout...")
    template_prompt = ("Interior photo of a modern pharmacy store, completely empty white metal shelves, "
                       "red and white color scheme, bright retail lighting. "
                       "Wide angle view of a central 4-row white metal shelf display with nothing on the shelves. "
                       "Photorealistic, 8k.")
    
    template_image = sd_pipe(template_prompt, num_inference_steps=30, guidance_scale=8.5).images[0]
    
    template_image.save(os.path.join(local_output_path, "pharmacy_template_empty.png"))
    print("Empty Pharmacy Template Layout saved.")
    return template_image

# --- STEP 2: Fill Template to create Base Image ---
def generate_base_stock(controlnet_pipe, control_image):
    print("Generating Fully Stocked Base Layout...")
    base_prompt = ("Interior photo of a modern pharmacy store, red and white color scheme, bright lighting. "
                   "Wide angle view of a central 4-row white metal shelf display perfectly filled with various standard health goods and colorful medicine boxes. "
                   "Highly organized, clean shelves. Photorealistic, 8k.")

    base_image = controlnet_pipe(
        base_prompt,
        image=control_image,
        num_inference_steps=30,
        guidance_scale=8.0,
        controlnet_conditioning_scale=1.0
    ).images[0]
    
    base_image.save(os.path.join(local_output_path, "pharmacy_base_layout.png"))
    print("Pharmacy Base Layout saved.")
    return base_image

# --- Main App ---
def main_pharmacy_scenario():
    # 0. Load Pipes
    sd_pipe, controlnet_pipe, inpaint_pipe = setup_pipeline()

    # 1. Generate Empty Template
    template_image = generate_empty_template(sd_pipe)

    # 2. Extract Structure
    control_image = get_canny_edges(template_image)

    # 3. Generate Fully Stocked Base using Structure
    base_image = generate_base_stock(controlnet_pipe, control_image)

    # 4. Generate Targetly Updated Image by Masking Top Shelf
    print("Updating Planogram via Targeted Inpainting...")
    mask_image = create_top_shelf_mask(base_image, top_percent=0.4)
    
    pharmacy_metadata = {
        "room_type": "pharmacy aisle",
        "status": "Restocked with Generics",
        "briefing_context": "The cold-medicine aisle was empty. Top two shelves are now filled with generic store-brand labels.",
        "substitutes": ["generic white-label store-brand medicine bottles", "simple healthcare packaging"],
        "constraints": "Maintain the existing red and white store color scheme and shelf structure."
    }

    # Focus prompt on what is appearing in the 'masked' area
    update_prompt = f"A professional photo of a {pharmacy_metadata['room_type']}. "
    update_prompt += f"The shelves feature {', '.join(pharmacy_metadata['substitutes'])}. "
    update_prompt += "Highly organized, generic generic medicine boxes, clean arrangement, bright retail lighting, 8k."

    updated_planogram = inpaint_pipe(
        prompt=update_prompt,
        image=base_image,
        mask_image=mask_image,
        num_inference_steps=30,
        guidance_scale=8.0
    ).images[0]

    updated_planogram.save(os.path.join(local_output_path, "pharmacy_updated_planogram.png"))
    print("Updated planogram saved to Local.")

def main():
    main_pharmacy_scenario()

if __name__ == '__main__':
    main()

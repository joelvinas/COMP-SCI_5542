# Resilient Shelf: Retail Layout Visualization

This repository contains the localized execution pipeline for creating tailored planograms and visually rendering retail layout shelf setups.

## Project Overview
The "Resilient Shelf" system transforms stock data into realistic retail layout visualizations utilizing a combination of pandas dataframe manipulation, SVG composition, OpenCV computer vision tools, and Stable Diffusion inpainting.

## File Structure
- `Manage_SKUs.py`: Automatically mocks inventory logic and integrates with Stable Diffusion (and `rembg`) to establish a dynamic, isolated catalog of individual SKU product images within the local `sku_assets` folder.
- `Planogram_Updater.py`: The core standalone executable script. It builds data-driven shelf assignments aiming for 80% visual coverage and prioritizes placement based on SKU velocity and margin. The script blends planar SKU placement via vectors into a unified background enhanced by Stable Diffusion.
- `requirements.txt`: Identifies all functional python package dependencies (`diffusers`, `opencv-python`, `Pillow`, `pandas`, `rembg`, `torch`, etc.).

## Setup Instructions

This system has been upgraded to execute as standalone local scripts over its previous iterative Colab execution. Given it utilizes deep learning models locally, executing this with a supported CUDA backend for `torch` is highly recommended.

1. Ensure Python 3.9+ is installed and configured in your environment.
2. Install pip dependencies seamlessly via the requirements file:
   ```bash
   pip install -r requirements.txt
   ```
3. (Optional) Run the SKU generation:
   ```bash
   python Manage_SKUs.py
   ```
4. Run the Planogram Pipeline to compose layouts and output media:
   ```bash
   python Planogram_Updater.py
   ```

## Generation & Outputs (`Store Output` directory)
The Planogram Updater automatically outputs the following to the `/Store Output/` location:
1. **pharmacy_intermediate_template.svg**: Intermediate anchor showcasing logic rules populating shelving constraints while deliberately leaving shelf 3 completely bare.
2. **pharmacy_base_layout.svg**: Base model incorporating High-Velocity stock selection for shelf 3.
3. **pharmacy_updated_planogram.svg**: An overriding update model simulating inventory modifications and shuffling across shelf 3 constraints.
4. **pharmacy_transition_video.svg**: An automatically coded, browser-ready animated SVG layout that flawlessly handles infinite cross-fade loops representing the Base and Updated shelving.
5. **pharmacy_transition_video.mp4**: A strictly rendered 30fps structural timeline encoded seamlessly by OpenCV using composited Pillow `.png` masks to generate an accurate cross-fade without web rendering dependencies. Note that this file inherently includes all placed vector elements baked into the resolution.
6. **manager_briefing.mp3**: An orchestrated TTS explanation created locally outlining the planogram transition.

## Use of AI Tools (Disclosure)
- **Large Language Models (LLM)**: Used an LLM to scaffold the Python boilerplate for the Diffusers pipeline and structure the repository components. It also assisted in writing proper prompt templates for structured inference.
- **Computer Vision Utilities**: Script utilizes heavily customized `Pillow` array compositing and OpenCV alpha-channel interpolations (`addWeighted`) to bypass vector animation headaches across various rendering dependencies to yield `.mp4` structures out-of-the-box.

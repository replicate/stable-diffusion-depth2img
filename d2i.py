import torch
from PIL import Image
import numpy as np
from diffusers import StableDiffusionDepth2ImgPipeline
from pathlib import Path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def pad_image(input_image):
    pad_w, pad_h = (
        np.max(((2, 2), np.ceil(np.array(input_image.size) / 64).astype(int)), axis=0)
        * 64
        - input_image.size
    )
    im_padded = Image.fromarray(
        np.pad(np.array(input_image), ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")
    )
    w, h = im_padded.size
    if w == h:
        return im_padded
    elif w > h:
        new_image = Image.new(im_padded.mode, (w, w), (0, 0, 0))
        new_image.paste(im_padded, (0, (w - h) // 2))
        return new_image
    else:
        new_image = Image.new(im_padded.mode, (h, h), (0, 0, 0))
        new_image.paste(im_padded, ((h - w) // 2, 0))
        return new_image


def predict(
    input_image,
    prompt,
    negative_prompt,
    steps,
    num_samples,
    scale,
    seed,
    strength,
    model,
    depth_image=None,
):
    depth = None
    if depth_image is not None:
        depth_image = pad_image(depth_image)
        depth_image = depth_image.resize((512, 512))
        depth = np.array(depth_image.convert("L"))
        depth = depth.astype(np.float32) / 255.0
        depth = depth[None, None]
        depth = torch.from_numpy(depth)
    init_image = input_image.convert("RGB")
    image = pad_image(init_image)  # resize to integer multiple of 32
    image = image.resize((512, 512))
    result = model(
        image=image,
        prompt=prompt,
        negative_prompt=negative_prompt,
        # depth_image=depth,
        # seed=seed,
        strength=strength,
        num_inference_steps=steps,
        guidance_scale=scale,
        num_images_per_prompt=num_samples,
    )
    return result["images"]

# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input, Path
from d2i import predict
from PIL import Image
from diffusers import StableDiffusionDepth2ImgPipeline

MODEL_ID = "stabilityai/stable-diffusion-2-depth"
MODEL_CACHE = "diffusers-cache"


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        print("Loading pipeline...")
        self.pipe = StableDiffusionDepth2ImgPipeline.from_pretrained(
            MODEL_ID,
            cache_dir=MODEL_CACHE,
            local_files_only=True,
        ).to("cuda")

    def predict(
        self,
        input_image: Path = Input(description="Grayscale input image"),
        prompt: str = Input(description="Prompt text", default=""),
        negative_prompt: str = Input(description="Negative prompt text", default=""),
        steps: int = Input(description="Number of steps", default=50),
        num_samples: int = Input(description="Number of samples", default=1),
        scale: float = Input(description="Scale", default=1.0),
        seed: int = Input(description="Seed", default=-1),
        strength: float = Input(description="Strength", default=0.5),
        depth_image: Path = Input(description="Depth image", default=None),
    ) -> Path:
        """Run a single prediction on the model"""
        input_image = Image.open(input_image)
        images = predict(
            input_image,
            prompt,
            negative_prompt,
            steps,
            num_samples,
            scale,
            seed,
            strength,
            self.pipe,
        )
        # write image to tmp
        images[0].save("output.png")
        return Path("output.png")

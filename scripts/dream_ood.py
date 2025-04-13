import argparse, os, sys, glob
import cv2
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from imwatermark import WatermarkEncoder
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid
import time
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import contextmanager, nullcontext

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler

from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from transformers import AutoFeatureExtractor

# Load safety model
safety_model_id = "CompVis/stable-diffusion-safety-checker"
safety_feature_extractor = AutoFeatureExtractor.from_pretrained(safety_model_id)
safety_checker = StableDiffusionSafetyChecker.from_pretrained(safety_model_id)

def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())

def numpy_to_pil(images):
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    return [Image.fromarray(image) for image in images]

def get_class_names(opt):
    if opt.id_data == 'in100':
        return ['stingray', 'hen', 'magpie', 'kite', 'vulture', ...]  # Truncated for brevity
    else:
        return ['apples', 'aquarium fish', 'baby', 'bear', 'beaver', ...]  # Truncated for brevity

def get_prompt(opt):
    import random
    chozen_class = random.choice(get_class_names(opt))
    article = 'an' if chozen_class[0] in 'aeiou' else 'a'
    return f"A high-quality image of {article} {chozen_class}", chozen_class

def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if verbose:
        if m: print("missing keys:", m)
        if u: print("unexpected keys:", u)
    model.cuda().eval()
    return model

def put_watermark(img, wm_encoder=None):
    if wm_encoder:
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img = wm_encoder.encode(img, 'dwtDct')
        img = Image.fromarray(img[:, :, ::-1])
    return img

def load_replacement(x):
    try:
        hwc = x.shape
        y = Image.open("assets/rick.jpeg").convert("RGB").resize((hwc[1], hwc[0]))
        y = (np.array(y)/255.0).astype(x.dtype)
        assert y.shape == x.shape
        return y
    except Exception:
        return x

def check_safety(x_image):
    safety_checker_input = safety_feature_extractor(numpy_to_pil(x_image), return_tensors="pt")
    x_checked_image, has_nsfw_concept = safety_checker(images=x_image, clip_input=safety_checker_input.pixel_values)
    for i, nsfw in enumerate(has_nsfw_concept):
        if nsfw:
            x_checked_image[i] = load_replacement(x_checked_image[i])
    return x_checked_image, has_nsfw_concept

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="a painting of a virus monster playing guitar")
    parser.add_argument("--outdir", type=str, default="/nobackup-fast/txt2img-samples-in100-demo/")
    parser.add_argument("--skip_grid", action='store_true')
    parser.add_argument("--id_data", type=str, default='in100')
    parser.add_argument("--skip_save", action='store_true')
    parser.add_argument("--ddim_steps", type=int, default=50)
    parser.add_argument("--plms", action='store_true')
    parser.add_argument("--laion400m", action='store_true')
    parser.add_argument("--gaussian_scale", type=float, default=0.0)
    parser.add_argument("--fixed_code", action='store_true')
    parser.add_argument("--ddim_eta", type=float, default=0.0)
    parser.add_argument("--n_iter", type=int, default=1)
    parser.add_argument("--H", type=int, default=512)
    parser.add_argument("--W", type=int, default=512)
    parser.add_argument("--C", type=int, default=4)
    parser.add_argument("--f", type=int, default=8)
    parser.add_argument("--n_samples", type=int, default=3)
    parser.add_argument("--loaded_embedding", type=str, default='/nobackup-slow/dataset/my_xfdu/diffusion/outlier_npos_embed.npy')
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--n_rows", type=int, default=0)
    parser.add_argument("--scale", type=float, default=7.5)
    parser.add_argument("--from-file", type=str)
    parser.add_argument("--config", type=str, default="configs/stable-diffusion/v1-inference.yaml")
    parser.add_argument("--ckpt", type=str, default="/nobackup-slow/dataset/my_xfdu/diffusion/sd-v1-4.ckpt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--precision", type=str, choices=["full", "autocast"], default="autocast")
    opt = parser.parse_args()

    # The actual pipeline logic would go here.

if __name__ == "__main__":
    main()
import pdb
from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).absolute().parents[0].absolute()
sys.path.insert(0, str(PROJECT_ROOT))
import os
import torch
import numpy as np
from PIL import Image
import cv2

import random
import time
import pdb

from pipelines_HandFit.pipeline_handfit_inp import HandFitPipeline
from pipelines_HandFit.unet_garm_2d_condition import UNetGarm2DConditionModel
from pipelines_HandFit.unet_vton_2d_condition import UNetVton2DConditionModel
from HandFit.pose_guider import Handprocessor
from HandFit.pose_control import ControlNet
from diffusers import UniPCMultistepScheduler, EulerAncestralDiscreteScheduler
from diffusers import AutoencoderKL
# from diffusers.models.embeddings import PositionNet

import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoProcessor, CLIPVisionModelWithProjection

VIT_PATH = "/home/tione/notebook/facenet/hand/ckpt/clip-vit-large-patch14"
# VAE_PATH = "/apdcephfs_cq8/share_2992679/private/byronjiang/try-on/code/OOTDiffusion/checkpoints/ootd/vae/"
VAE_PATH = '/home/tione/notebook/facenet/hand/ckpt/stable-diffusion-v1-5/vae'
# 换成sd1.5的目录的vae
MODEL_PATH = "/home/tione/notebook/facenet/benchmark/baselines/VTON-HandFit-infer/checkpoints/ootd"

class HandFit:

    def __init__(self, gpu_id, modelroot, controlroot):
        self.gpu_id = 'cuda:' + str(gpu_id)

        vae = AutoencoderKL.from_pretrained(
            VAE_PATH,
            torch_dtype=torch.float16,
        )

        unet_garm = UNetGarm2DConditionModel()
        unet_garm_path = f'{modelroot}/unet_garm/diffusion_pytorch_model.bin'
        unet_garm.load_state_dict(torch.load(unet_garm_path))
        unet_garm.half()
        unet_vton = UNetVton2DConditionModel(in_channels=9)
        unet_vton.set_ip_adapter()
        unet_vton_path = f'{modelroot}/unet_vton/diffusion_pytorch_model.bin'
        unet_vton.load_state_dict(torch.load(unet_vton_path))
        unet_vton.half()
        handprocessor = Handprocessor()
        controlnet = ControlNet(image_size=32,
                                    in_channels=8,
                                    hint_channels=3,
                                    model_channels=320,
                                    attention_resolutions=[4,2,1],
                                    num_res_blocks=2,
                                    channel_mult=[1,2,4,4],
                                    num_heads=8,
                                    use_spatial_transformer=True,
                                    transformer_depth=1,
                                    context_dim=768,
                                    use_checkpoint=True,
                                    # num_attention_blocks=4,
                                    # disable_middle_self_attn=True,
                                    legacy=False,
                                    use_fp16=True)
        handprocessor_path = f'{controlroot}/handprocessor/diffusion_pytorch_model.bin'
        handprocessor.load_state_dict(torch.load(handprocessor_path))
        controlnet_path = f'{controlroot}/controlnet/diffusion_pytorch_model.bin'
        controlnet.load_state_dict(torch.load(controlnet_path))
        
        handprocessor.to(self.gpu_id).half().eval()
        controlnet.to(self.gpu_id).half().eval()
        self.pipe = HandFitPipeline.from_pretrained(
            MODEL_PATH,
            unet_garm=unet_garm,
            unet_vton=unet_vton,
            vae=vae,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
            safety_checker=None,
            requires_safety_checker=False,  
            controlnet = controlnet,
            handprocessor = handprocessor,
        ).to(self.gpu_id)

        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
        
        self.auto_processor = AutoProcessor.from_pretrained(VIT_PATH)
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(VIT_PATH).to(self.gpu_id)

        self.text_embeddings = torch.load("/home/tione/notebook/facenet/benchmark/baselines/VTON-HandFit-infer/text_embedding.pt")


    def __call__(self,
                image_garm=None,
                image_vton=None,
                mask=None,
                image_ori=None,
                num_samples=1,
                num_steps=20,
                image_scale=1.0,
                seed=-1,
                pose_image=None,
                denspose_image = None,
                depth_image = None,
                mano_6d = None,
                image_bbox = None,
                image_pos2d = None,
                vertices = None,
                handtype = None,
                dino_embedding=None,
    ):
        if seed == -1:
            random.seed(time.time())
            seed = random.randint(0, 2147483647)
        print('Initial seed: ' + str(seed))
        generator = torch.manual_seed(seed)

        with torch.no_grad():
            prompt_image = self.auto_processor(images=image_garm, return_tensors="pt").to(self.gpu_id)
            prompt_image = self.image_encoder(prompt_image.data['pixel_values']).image_embeds
            prompt_image = prompt_image.unsqueeze(1)
            prompt_embeds = self.text_embeddings["empty"].to(self.gpu_id)
            prompt_embeds[:, 1:] = prompt_image[:]

            images = self.pipe(prompt_embeds=prompt_embeds,
                        image_garm=image_garm,
                        image_vton=image_vton, 
                        mask=mask,
                        image_ori=image_ori,
                        num_inference_steps=num_steps,
                        image_guidance_scale=image_scale,
                        num_images_per_prompt=num_samples,
                        generator=generator,
                        pose_image=pose_image,
                        denspose_image = denspose_image,
                        depth_image = depth_image,
                        mano_6d = mano_6d,
                        image_bbox = image_bbox,
                        image_pos2d = image_pos2d,
                        vertices = vertices,
                        handtype = handtype,
                        dino_embedding = dino_embedding
            ).images

        return images

    def reload_model(self, modelroot, controlroot):
        self.pipe.unet_garm.load_state_dict(torch.load(os.path.join(modelroot, "unet_garm", "diffusion_pytorch_model.bin")))
        self.pipe.unet_vton.load_state_dict(torch.load(os.path.join(modelroot, "unet_vton", "diffusion_pytorch_model.bin")))
        self.pipe.controlnet.load_state_dict(torch.load(os.path.join(controlroot, "controlnet", "diffusion_pytorch_model.bin")))
        self.pipe.handprocessor.load_state_dict(torch.load(os.path.join(controlroot, "handprocessor", "diffusion_pytorch_model.bin")))
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

from pipelines_ootd.pipeline_ootd import OotdPipeline
from pipelines_ootd.unet_garm_2d_condition import UNetGarm2DConditionModel
from pipelines_ootd.unet_vton_2d_condition import UNetVton2DConditionModel
from ootd.pose_guider import PoseGuider,PositionNet_mano_dino_ln_nobbox
from ootd.pose_control import ControlNet
from diffusers import UniPCMultistepScheduler
from diffusers import AutoencoderKL

import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoProcessor, CLIPVisionModelWithProjection
from transformers import CLIPTextModel, CLIPTokenizer

# VIT_PATH = "../checkpoints/clip-vit-large-patch14"
# VAE_PATH = "../checkpoints/ootd"
# UNET_PATH = "../checkpoints/ootd/ootd_dc/checkpoint-36000"
# MODEL_PATH = "../checkpoints/ootd"

VIT_PATH = "/apdcephfs_cq8/share_2992679/ckpt/clip-vit-large-patch14"
VAE_PATH = "/apdcephfs_cq8/share_2992679/private/byronjiang/try-on/code/OOTDiffusion/checkpoints/ootd/vae/"
# VAE_PATH = '/apdcephfs_cq8/share_2992679/ckpt/stable-diffusion-v1-5/vae'
VAE_PATH = "/apdcephfs_cq8/share_2992679/ckpt/sd-vae-ft-ema"

# 换成sd1.5的目录的vae
MODEL_PATH = "/apdcephfs_cq8/share_2992679/private/byronjiang/try-on/code/yt-image-try-on-sd15/checkpoints/ootd"



class OOTDiffusionbaseline:

    def __init__(self, gpu_id, model_root):
        self.gpu_id = 'cuda:' + str(gpu_id)

        vae = AutoencoderKL.from_pretrained(
            VAE_PATH,
            subfolder="vae",
            torch_dtype=torch.float16,
        )

        # unet_garm = UNetGarm2DConditionModel.from_pretrained(
        #     UNET_PATH,
        #     subfolder="unet_garm",
        #     torch_dtype=torch.float16,
        #     use_safetensors=True,
        # )
        # unet_vton = UNetVton2DConditionModel.from_pretrained(
        #     UNET_PATH,
        #     subfolder="unet_vton",
        #     torch_dtype=torch.float16,
        #     use_safetensors=True,
        # )

        # unet_garm = UNetGarm2DConditionModel.from_pretrained(
        #     "/apdcephfs_cq8/share_2992679/private/byronjiang/try-on/code/wear-any-way/run/ootd_trained_dc_768_dwpose_all_newagn_garm_scaled/checkpoint-40000/unet_garm",
        #     torch_dtype=torch.float16,
        # )
        # unet_vton = UNetVton2DConditionModel.from_pretrained(
        #     "/apdcephfs_cq8/share_2992679/private/byronjiang/try-on/code/wear-any-way/run/ootd_trained_dc_768_dwpose_all_newagn_garm_scaled/checkpoint-40000/unet_vton",
        #     torch_dtype=torch.float16,
        # )
        # pose_guider = PoseGuider(conditioning_embedding_channels=320, conditioning_channels=3, block_out_channels=(16, 32, 96, 256))
        # pose_guider.load_state_dict(torch.load("/apdcephfs_cq8/share_2992679/private/byronjiang/try-on/code/wear-any-way/run/ootd_trained_dc_768_dwpose_all_newagn_garm_scaled/checkpoint-40000/pose_guider/diffusion_pytorch_model.bin"))
        
        # unet_garm = UNetGarm2DConditionModel.from_pretrained(
        #     "/apdcephfs_cq8/share_2992679/private/byronjiang/try-on/code/wear-any-way/run/ootd_trained_dc_768_dwpose_hd_dc_upper_newagn_garm_scaled/checkpoint-40000/unet_garm",
        #     torch_dtype=torch.float16,
        # )
        # unet_vton = UNetVton2DConditionModel.from_pretrained(
        #     "/apdcephfs_cq8/share_2992679/private/byronjiang/try-on/code/wear-any-way/run/ootd_trained_dc_768_dwpose_hd_dc_upper_newagn_garm_scaled/checkpoint-40000/unet_vton",
        #     torch_dtype=torch.float16,
        # )
        # pose_guider = PoseGuider(conditioning_embedding_channels=320, conditioning_channels=3, block_out_channels=(16, 32, 96, 256))
        # pose_guider.load_state_dict(torch.load("/apdcephfs_cq8/share_2992679/private/byronjiang/try-on/code/wear-any-way/run/ootd_trained_dc_768_dwpose_hd_dc_upper_newagn_garm_scaled/checkpoint-40000/pose_guider/diffusion_pytorch_model.bin"))


        # pose_guider.to(self.gpu_id).half()
        unet_garm = UNetGarm2DConditionModel()
        unet_garm_path = f'{model_root}/unet_garm/diffusion_pytorch_model.bin'
        unet_garm.load_state_dict(torch.load(unet_garm_path))
        unet_garm.half()
        unet_vton = UNetVton2DConditionModel()
        # unet_vton.set_ip_adapter()
        unet_vton_path = f'{model_root}/unet_vton/diffusion_pytorch_model.bin'
        unet_vton.load_state_dict(torch.load(unet_vton_path))
        unet_vton.half()

        # positionnet_nobbox = PositionNet_mano_dino_ln_nobbox()
        # positionnet_nobbox_path  = f'{model_root}/positionnet/diffusion_pytorch_model.bin'
        # positionnet_nobbox.load_state_dict(torch.load(positionnet_nobbox_path))

        # controlnet = ControlNet( image_size=32,
        #                             in_channels=8,
        #                             hint_channels=3,
        #                             model_channels=320,
        #                             attention_resolutions=[4,2,1],
        #                             num_res_blocks=2,
        #                             channel_mult=[1,2,4,4],
        #                             num_heads=8,
        #                             use_spatial_transformer=True,
        #                             transformer_depth=1,
        #                             context_dim=768,
        #                             use_checkpoint=True,
        #                             # num_attention_blocks=4,
        #                             # disable_middle_self_attn=True,
        #                             legacy=False,
        #                             use_fp16=True)
        # controlnet_path = f'{model_root}/controlnet/diffusion_pytorch_model.bin'
        # controlnet.load_state_dict(torch.load(controlnet_path))
        
        
        # positionnet_nobbox.to(self.gpu_id).half()
        # # # # dino_linear.to(self.gpu_id).half()
        # controlnet.to(self.gpu_id).half().eval()

        self.pipe = OotdPipeline.from_pretrained(
            MODEL_PATH,
            unet_garm=unet_garm,
            unet_vton=unet_vton,
            vae=vae,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
            safety_checker=None,
            requires_safety_checker=False,
            # pose_guider=pose_guider
            # controlnet = controlnet,
            # positionnet_nobbox = positionnet_nobbox,

        ).to(self.gpu_id)

        # self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
        
        # self.auto_processor = AutoProcessor.from_pretrained(VIT_PATH)
        # self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(VIT_PATH).to(self.gpu_id)

        # self.tokenizer = CLIPTokenizer.from_pretrained(
        #     MODEL_PATH,
        #     subfolder="tokenizer",
        # )
        # self.text_encoder = CLIPTextModel.from_pretrained(
        #     MODEL_PATH,
        #     subfolder="text_encoder",
        # ).to(self.gpu_id)
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
        # self.pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(self.pipe.scheduler.config)
        
        self.auto_processor = AutoProcessor.from_pretrained(VIT_PATH)
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(VIT_PATH).to(self.gpu_id)

        self.text_embeddings = torch.load("/apdcephfs_cq8/share_2992679/private/byronjiang/try-on/code/yt-image-try-on-sd15/run/text_embedding.pt")

        # self.text_embeddings = torch.load("/apdcephfs_cq8/share_2992679/private/byronjiang/try-on/code/yt-image-try-on-sd15/run/text_embedding.pt")


    def tokenize_captions(self, captions, max_length):
        inputs = self.tokenizer(
            captions, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        return inputs.input_ids


    def __call__(self,
                model_type='hd',
                category='upperbody',
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
            # prompt_image = self.auto_processor(images=image_garm, return_tensors="pt").to(self.gpu_id)
            # prompt_image = self.image_encoder(prompt_image.data['pixel_values']).image_embeds
            # prompt_image = prompt_image.unsqueeze(1)
            # if model_type == 'hd':
            #     prompt_embeds = self.text_encoder(self.tokenize_captions([""], 2).to(self.gpu_id))[0]
            #     prompt_embeds[:, 1:] = prompt_image[:]
            # elif model_type == 'dc':
            #     prompt_embeds = self.text_encoder(self.tokenize_captions([category], 3).to(self.gpu_id))[0]
            #     prompt_embeds = torch.cat([prompt_embeds, prompt_image], dim=1)
            # else:
            #     raise ValueError("model_type must be \'hd\' or \'dc\'!")

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

            # images = self.pipe(prompt_embeds=prompt_embeds,
            #             image_garm=image_garm,
            #             image_vton=image_vton, 
            #             mask=mask,
            #             image_ori=image_ori,
            #             num_inference_steps=num_steps,
            #             image_guidance_scale=image_scale,
            #             num_images_per_prompt=num_samples,
            #             generator=generator,
            #             pose_image=pose_image,
            #             model_parse=model_parse,
            # ).images

        return images

import gradio as gr
import os
from pathlib import Path
import sys
import torch
from PIL import Image, ImageOps

from run.utils_ootd_v1 import get_mask_location_v1
PROJECT_ROOT = Path(__file__).absolute().parents[1].absolute()
sys.path.insert(0, str(PROJECT_ROOT))

from preprocess.humanparsing.run_parsing import Parsing
from HandFit.inference_handfit_inp import HandFit

from transformers import AutoImageProcessor, AutoModel
from hamer_depth import get_depth_info

from HandFit.dwpose import DWposeDetector
import numpy as np
import cv2
import math

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from densepose import add_densepose_config
from densepose.vis.extractor import DensePoseResultExtractor
from densepose.vis.densepose_results import DensePoseResultsFineSegmentationVisualizer as Visualizer

cfg = get_cfg()
add_densepose_config(cfg)
cfg.merge_from_file("checkpoints/densepose/densepose_rcnn_R_50_FPN_s1x.yaml")
cfg.MODEL.WEIGHTS = "checkpoints/densepose/model_final_162be9.pkl"
cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
predictor = DefaultPredictor(cfg)

model_folder = './dinov2'
dino_processor = AutoImageProcessor.from_pretrained(model_folder)
dino_model = AutoModel.from_pretrained(model_folder).cuda()
dino_model.half().eval()

dwprocessor = DWposeDetector()
parsing_model = Parsing(0)

handfit_upper = HandFit(0, modelroot='./run/trained_checkpoints/upper-checkpoint', \
    controlroot="./run/trained_checkpoints/upper-checkpoint")
            
example_path = os.path.join(os.path.dirname(__file__), 'examples')
model_hd = os.path.join(example_path, 'model/00611_00.jpg')
garment_hd = os.path.join(example_path, 'garment/048554_1.jpg')


def pad_and_resize(im, new_width=768, new_height=1024, pad_color=(255, 255, 255)):
    old_width, old_height = im.size
    
    # 计算缩放比例和新尺寸
    ratio_w = new_width / old_width
    ratio_h = new_height / old_height
    if ratio_w < ratio_h:
        new_size = (new_width, round(old_height * ratio_w))
    else:
        new_size = (round(old_width * ratio_h), new_height)
    
    # 缩放图像
    im_resized = im.resize(new_size, Image.LANCZOS)

    # 计算pad的大小
    pad_w = math.ceil((new_width - im_resized.width) / 2)
    pad_h = math.ceil((new_height - im_resized.height) / 2)

    # 创建一个新的背景画布并填充颜色
    new_im = Image.new('RGB', (new_width, new_height), pad_color)
    
    # 将缩放后的图像粘贴到新的画布上
    new_im.paste(im_resized, (pad_w, pad_h))

    return new_im, pad_w, pad_h
    
def unpad_and_resize(padded_im, pad_w, pad_h, original_width, original_height):
    width, height = padded_im.size


    # 确定裁剪区域，确保不超过图像边界
    left = pad_w
    top = pad_h
    right = width - pad_w
    bottom = height - pad_h
    
    # 裁剪出不含填充的中间区域
    cropped_im = padded_im.crop((left, top, right, bottom))

    # 进行逆向缩放至原始尺寸
    resized_im = cropped_im.resize((original_width, original_height), Image.LANCZOS)

    return resized_im

def process(vton_img_path, garm_img_path, n_samples=1, n_steps=20, image_scale=2.0, seed=-1):
    category="upper_body"
    with torch.inference_mode():
        garm_img = Image.open(garm_img_path)
        vton_img = Image.open(vton_img_path)
        model_image_size = vton_img.size
        new_width = 768
        new_height = 1024
        vton_img, pad_w, pad_h = pad_and_resize(vton_img, new_width=new_width, new_height=new_height)
        garm_img, _, _ = pad_and_resize(garm_img, new_width=new_width, new_height=new_height)

        pose_image, keypoints, _, candidate = dwprocessor(np.array(vton_img)[:,:,::-1])
        
        pose_image = pose_image[:,:,::-1] 
        pose_image = Image.fromarray(pose_image)

        outputs = predictor(np.array(vton_img)[:,:,::-1])['instances']        
        results = DensePoseResultExtractor()(outputs)
        arr = np.zeros((vton_img.height, vton_img.width,3), dtype=np.uint8)
        out_frame = Visualizer(alpha=1, inplace=True).visualize(arr, results)
        densepose = out_frame[:,:,::-1]

        # hand parameters
        depth_img, result_list, crop_imgs = get_depth_info(vton_img, vton_img_path)
        if depth_img is None:
            black_image = np.zeros((new_height, new_width, 3), dtype=np.uint8)
            depth_img = Image.fromarray(black_image).convert('RGB')
            # handtype = torch.tensor( [-1,-1]).unsqueeze(-1).to('cuda').half()
            handtype = torch.tensor([-1, -1]).unsqueeze(-1).unsqueeze(0).expand(n_samples, 2, 1).to('cuda').half()
            image_pos2d = torch.zeros(n_samples,2,42).to('cuda').half()
            vertices = torch.zeros(n_samples,2,1024).to('cuda').half()
            image_bbox = torch.zeros(n_samples,2,4).to('cuda').half()
            mano_6d = torch.zeros(n_samples,2,96).to('cuda').half()
            dino_embedding = torch.zeros(n_samples,2,1536).to('cuda').half()
        else:
            depth_img = Image.fromarray(depth_img).convert('RGB')
            handtype,image_pos2d,vertices,image_bbox, mano_6d = result_list
            new_embedding = []
            
            for i in range(len(image_bbox)):
                
                x1, y1, x2, y2 = image_bbox[i]
                if x1 == 0 and y1 == 0 and x2 == 0 and y2 == 0:
                    continue
                crop_img = vton_img.crop((x1, y1, x2, y2))
                crop_img_pad = pad_and_resize(crop_img, new_width=224, new_height=224)[0]
                crop_img_pad = dino_processor(crop_img_pad, return_tensors="pt").to('cuda')
                with torch.no_grad():
                    outputs = dino_model(**crop_img_pad).last_hidden_state.mean(dim=1).view(-1)
                new_embedding.append(outputs)
            # dino_embedding = []
            if len(new_embedding)==1:
                new_embedding.append(torch.zeros_like(new_embedding[0]))
                dino_embedding = torch.stack( new_embedding,dim=0).cuda().half()
            else:
                dino_embedding = torch.stack( new_embedding,dim=0).cuda().half()
            dino_embedding = dino_embedding.unsqueeze(0).to('cuda').half().expand(n_samples, 2,-1)
            handtype = torch.tensor(handtype).unsqueeze(-1).to('cuda').half().expand(n_samples, 2, -1)
            vertices = torch.tensor(vertices).unsqueeze(0).to('cuda').half().expand(n_samples, 2, -1)
            image_bbox = torch.tensor(image_bbox)
            image_pos2d = torch.tensor(image_pos2d)
            mano_6d = torch.tensor(mano_6d).unsqueeze(0).to('cuda').half().contiguous().view(1,2,96).expand(n_samples, 2,96)
            
            xmin_ymin = image_bbox[:,:2]  # [1, 2]
            image_pos2d = image_pos2d - xmin_ymin.unsqueeze(1)
            bbox_widths = image_bbox[:, 2] - image_bbox[:, 0]
            bbox_heights = image_bbox[:, 3] - image_bbox[:, 1]
            epsilon = 1e-10
            bbox_widths = bbox_widths.float() + epsilon
            bbox_heights = bbox_heights.float() + epsilon
            image_pos2d[:,:,1] = image_pos2d[:,:,1] / bbox_heights.unsqueeze(1)       
            image_pos2d[:,:,0] = image_pos2d[:,:,0] / bbox_widths.unsqueeze(1)  
            image_bbox = image_bbox.float()
            image_bbox[:, [0, 2]] = image_bbox[:, [0, 2]] / 768.   
            image_bbox[:, [1, 3]] = image_bbox[:, [1, 3]] / 1024.  
            image_bbox = image_bbox.unsqueeze(0).to('cuda').half().expand(n_samples, 2,4)
            image_pos2d = image_pos2d.unsqueeze(0).to('cuda').half().contiguous().view(1,2,42).expand(n_samples, 2,42)

        model_parse, _ = parsing_model(vton_img)
        
        mask, mask_gray = get_mask_location_v1(category, model_parse, candidate, \
                                               width=new_width, height=new_height, densepose=densepose, depthmap=depth_img)
        densepose = Image.fromarray(densepose)
        vton_img = vton_img.resize((new_width, new_height), Image.LANCZOS)
        mask = mask.resize((new_width, new_height), Image.NEAREST)
        mask_gray = mask_gray.resize((new_width, new_height), Image.NEAREST)
        pose_image = pose_image.resize((new_width, new_height), Image.NEAREST)
        masked_vton_img = Image.composite(mask_gray, vton_img, mask)

        images = handfit_upper(
            image_garm=garm_img,   
            image_vton=masked_vton_img, 
            mask=mask, 
            image_ori=vton_img, 
            num_samples=n_samples,
            num_steps=n_steps,
            image_scale=image_scale,
            seed=seed,
            pose_image=pose_image,
            denspose_image = densepose,
            depth_image = depth_img,
            mano_6d = mano_6d,
            image_bbox = image_bbox,
            image_pos2d = image_pos2d,
            vertices = vertices,
            handtype = handtype,
            dino_embedding=dino_embedding,
        )

    out_images = []
    for img in images:
        out_images.append(unpad_and_resize(img, pad_w, pad_h, model_image_size[0], model_image_size[1]))
    out_images.append(masked_vton_img)
    out_images.append(pose_image)
    out_images.append(depth_img)
    out_images.append(densepose)
    return out_images


block = gr.Blocks().queue()
with block:
    with gr.Row():
        gr.Markdown("## VTON-HandFit")
    with gr.Row():
        with gr.Column():
            vton_img = gr.Image(label="Model", sources='upload', type="filepath", height=384, value=model_hd)
            example = gr.Examples(
                label="Examples (upper-body)",
                inputs=vton_img,
                examples_per_page=7,
                examples=[
                    os.path.join(example_path, 'model/00611_00.jpg'),
                    os.path.join(example_path, 'model/00888_00.jpg'),
                    os.path.join(example_path, 'model/01057_00.jpg'),
                    os.path.join(example_path, 'model/14008_00.jpg'),
                ])
        with gr.Column():
            garm_img = gr.Image(label="Garment", sources='upload', type="filepath", height=384, value=garment_hd)
            example = gr.Examples(
                label="Examples (upper-body)",
                inputs=garm_img,
                examples_per_page=7,
                examples=[
                    os.path.join(example_path, 'garment/048554_1.jpg'),
                    os.path.join(example_path, 'garment/01155_00.jpg'),
                    os.path.join(example_path, 'garment/050181_1.jpg'),
                    os.path.join(example_path, 'garment/02518_00.jpg'),
                ])
        with gr.Column():
            result_gallery_dc = gr.Gallery(label='Output', show_label=False, elem_id="gallery", preview=True, scale=1)   
    with gr.Column():
        run_button = gr.Button(value="Run")
        n_samples = gr.Slider(label="Images", minimum=1, maximum=4, value=1, step=1)
        n_steps = gr.Slider(label="Steps", minimum=1, maximum=40, value=20, step=1)
        image_scale = gr.Slider(label="Guidance scale", minimum=1.0, maximum=5.0, value=2.0, step=0.1)
        seed = gr.Slider(label="Seed", minimum=-1, maximum=2147483647, step=1, value=-1)
        
    ips_dc = [vton_img, garm_img, n_samples, n_steps, image_scale, seed]
    run_button.click(fn=process, inputs=ips_dc, outputs=[result_gallery_dc])

block.launch(server_name='0.0.0.0', server_port=8081)

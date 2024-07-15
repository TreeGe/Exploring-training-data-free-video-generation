import os, pdb
from glob import glob
import argparse
import numpy as np
import torch
import requests
from PIL import Image

from lavis.models import load_model_and_preprocess

from src.utils.ddim_inv import DDIMInversion
from src.utils.scheduler import DDIMInverseScheduler

# if torch.cuda.is_available():
#     device = "cuda"
# else:
device = "cuda"

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_image', type=str, default='output/test/12.4/bear.png')
    parser.add_argument('--results_folder', type=str, default='output/test/12.4')
    parser.add_argument('--num_ddim_steps', type=int, default=70)
    parser.add_argument('--model_path', type=str, default='model_path')
    parser.add_argument('--use_float_16', action='store_true')
    args = parser.parse_args()

    # make the output folders
    os.makedirs(os.path.join(args.results_folder, "inversion"), exist_ok=True)
    os.makedirs(os.path.join(args.results_folder, "prompt"), exist_ok=True)

    if args.use_float_16:
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32


    # load the BLIP model  加载BLIP
    model_blip, vis_processors, _ = load_model_and_preprocess(name="blip_caption", model_type="base_coco", is_eval=True, device=torch.device(device))#作用生成与图像相关的语言描述
    # make the DDIM inversion pipeline  加载DDIM
    pipe = DDIMInversion.from_pretrained(args.model_path, torch_dtype=torch_dtype).to(device)  #将图像变为高斯噪声？
    pipe.scheduler = DDIMInverseScheduler.from_config(pipe.scheduler.config)


    # if the input is a folder, collect all the images as a list
    if os.path.isdir(args.input_image):
        l_img_paths = sorted(glob(os.path.join(args.input_image, "*.png")))
    else:
        l_img_paths = [args.input_image]


    for img_path in l_img_paths:
        bname = os.path.basename(img_path).split(".")[0]
        img = Image.open(img_path).resize((512,512), Image.Resampling.LANCZOS)
        # generate the caption

        _image = vis_processors["eval"](img).unsqueeze(0).to(device) #(1.3.384.384) 对图像进行BLIP可视化处理
        prompt_str = model_blip.generate({"image": _image})[0]   #通过BLIP生成图像的文本提示
        x_inv, x_inv_image, x_dec_img = pipe(
            prompt_str,
            guidance_scale=1,
            num_inversion_steps=args.num_ddim_steps,
            img=img,
            torch_dtype=torch_dtype
        )
        # #第一个输出x_inv后面需要用到的加完噪声的latent(1,4,64,64)
        # #第二个x_inv_image加完噪声的latent经过decode得到的加噪图像？
        #
        # # save the inversion
        torch.save(x_inv[0], os.path.join(args.results_folder, f"inversion/{bname}_{args.num_ddim_steps}.pt"))   #保存的是高斯噪声？
        # save the prompt string
        with open(os.path.join(args.results_folder, f"prompt/{bname}.txt"), "w") as f:
            f.write(prompt_str)


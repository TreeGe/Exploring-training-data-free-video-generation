import os

import argparse
import torch
import glob

from diffusers import DDIMScheduler
from utils.edit_directions import construct_direction
from src.edit_pipeline import EditingPipeline

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--inversion',default='output/test/inversion/1_0.pt',type=str)
    parser.add_argument('--prompt', default='output/test/prompt/1_0.txt',type=str)
    parser.add_argument('--task_name', type=str, default='ree2reeduringfall')
    parser.add_argument('--results_folder', type=str, default='output/test')
    parser.add_argument('--num_ddim_steps', type=int, default=50)
    parser.add_argument('--model_path', type=str, default='model_path')
    parser.add_argument('--xa_guidance', default=0.1, type=float)
    parser.add_argument('--negative_guidance_scale', default=5.0, type=float)
    parser.add_argument('--use_float_16', action='store_true')
    parser.add_argument('--samples_times', type=int, default=1)

    args = parser.parse_args()

    os.makedirs(os.path.join(args.results_folder, "edit"), exist_ok=True)
    os.makedirs(os.path.join(args.results_folder, "reconstruction"), exist_ok=True)

    if args.use_float_16:
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    # if the inversion is a folder, the prompt should also be a folder
    assert (os.path.isdir(args.inversion)==os.path.isdir(args.prompt)), "If the inversion is a folder, the prompt should also be a folder"
    if os.path.isdir(args.inversion):
        l_inv_paths = sorted(glob.glob(os.path.join(args.inversion, "*.pt")))
        l_bnames = [os.path.basename(x) for x in l_inv_paths]
        l_prompt_paths = [os.path.join(args.prompt, x.replace(".pt",".txt")) for x in l_bnames]
    else:
        l_inv_paths = [args.inversion]   #获取加噪后的图像
        l_prompt_paths = [args.prompt]   #获取文本提示

    # Make the editing pipeline  制作编辑管道,加载预训练好的模型
    pipe = EditingPipeline.from_pretrained(args.model_path, torch_dtype=torch_dtype).to(device) #加载pipline
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)


    for inv_path, prompt_path in zip(l_inv_paths, l_prompt_paths):
        prompt_str = open(prompt_path).read().strip()  #原始图像的文本提示
        rec_pil, edit_pil = pipe(prompt_str,
                num_inference_steps=args.num_ddim_steps,
                x_in=torch.load(inv_path).unsqueeze(0),
                edit_dir=construct_direction(args.task_name),
                guidance_amount=args.xa_guidance,
                guidance_scale=args.negative_guidance_scale,
                negative_prompt=prompt_str , # use the unedited prompt for the negative prompt
                samples_times = args.samples_times

        )
        
        bname = os.path.basename(args.inversion).split(".")[0]
        for i in range(0,args.samples_times):
            (edit_pil[i])[0].save(os.path.join(args.results_folder, f"1000_0.{i}_1.png"))


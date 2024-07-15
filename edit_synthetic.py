import os

import argparse
import torch

from diffusers import DDIMScheduler
from src.utils.edit_directions import construct_direction
from src.edit_pipeline_inter import EditingPipeline

# if torch.cuda.is_available():
#     device = "cuda"
# else:
device = "cuda"


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt_str', type=str, default='The mountain at sunrise')
    parser.add_argument('--random_seed', default=7)
    parser.add_argument('--task_name', type=str, default='The mountain at sunrise2The mountain at sunse')
    parser.add_argument('--results_folder', type=str, default='output/test')
    parser.add_argument('--num_ddim_steps', type=int, default=50)
    parser.add_argument('--model_path', type=str, default='model_path')
    parser.add_argument('--xa_guidance', default=0.1, type=float)
    parser.add_argument('--negative_guidance_scale', default=5.0, type=float)
    parser.add_argument('--use_float_16', action='store_true')
    parser.add_argument('--samples_times', type=int ,default=10)
    args = parser.parse_args()

    os.makedirs(args.results_folder, exist_ok=True)

    if args.use_float_16:
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    # make the input noise map
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.random_seed)  #设置随机种子，让每次生成的随机种子都相同
    else:
        torch.manual_seed(args.random_seed)

    x = torch.randn((1,4,64,64), device=device)
    x_1 = torch.randn((1,4,64,64), device=device)

    # Make the editing pipeline
    pipe = EditingPipeline.from_pretrained(args.model_path, torch_dtype=torch_dtype).to(device)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.safety_checker = lambda images, clip_input: (images, False)


    rec_pil, edit_pil= pipe(args.prompt_str,
        num_inference_steps=args.num_ddim_steps,
        x_in=x,
        # x_1 = x_1,
        edit_dir=construct_direction(args.task_name),
        guidance_amount=args.xa_guidance,
        guidance_scale=args.negative_guidance_scale,
        negative_prompt="", # use the empty string for the negative prompt
        samples_times=args.samples_times
    )

    rec_pil[0].save(f"output/test/9.9/{args.random_seed}_1.png")
    for i in range(0,args.samples_times):        
        (edit_pil[i])[0].save(f"output/test/9.9/{args.random_seed}{i}_1.png")
    
    
    # (edit_pil[0])[0].save(os.path.join(args.results_folder, f"edit_01.png"))
    # rec_pil[0].save(os.path.join(args.results_folder, f"reconstruction_01.png"))
    # res[0].save(os.path.join(args.results_folder, f"res_1.png"))

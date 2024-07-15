import os, pdb
import numpy as np
import argparse
import torch
from glob import glob
import requests
from PIL import Image
from diffusers import DDIMScheduler
from utils.edit_directions import construct_direction
from src.edit_pipeline_1 import EditingPipeline
from lavis.models import load_model_and_preprocess
from utils.ddim_inv import DDIMInversion
from utils.scheduler import DDIMInverseScheduler


if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

    
def load_sentence_embeddings(l_sentences, tokenizer, text_encoder, device=device):
    with torch.no_grad():
        l_embeddings = []
        for sent in l_sentences:
            text_inputs = tokenizer(
                    sent,
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )
            text_input_ids = text_inputs.input_ids
            prompt_embeds = text_encoder(text_input_ids.to(device), attention_mask=None)[0]
            l_embeddings.append(prompt_embeds)
    return torch.cat(l_embeddings, dim=0).mean(dim=0).unsqueeze(0)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('--task_name', type=str, default='The mountain at sunrise2The mountain at sunse')
    # parser.add_argument('--results_folder', type=str, default='output/test')
    parser.add_argument('--input_image', type=str, default='output/test/9.13/video/4_rec.png')
    parser.add_argument('--num_ddim_steps', type=int, default=50)
    # parser.add_argument('--model_path', type=str, default='CompVis/stable-diffusion-v1-4')
    parser.add_argument('--model_path', type=str, default='model_path')
    parser.add_argument('--xa_guidance', default=0.1, type=float)
    parser.add_argument('--negative_guidance_scale', default=5.0, type=float)
    parser.add_argument('--use_float_16', action='store_true')
    
    parser.add_argument('--input_sentence', default="lake in noon")
    parser.add_argument('--output_sentence', default="lake with fog")
    parser.add_argument('--samples_times', type=int, default=1)

    args = parser.parse_args()


    if args.use_float_16:
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    pipe1 = DDIMInversion.from_pretrained(args.model_path, torch_dtype=torch.float32).to('cpu')  #将图像变为高斯噪声？
    pipe1.scheduler = DDIMInverseScheduler.from_config(pipe1.scheduler.config)    
    
    
    # Make the editing pipeline  制作编辑管道,加载预训练好的模型
    pipe = EditingPipeline.from_pretrained(args.model_path, torch_dtype=torch_dtype).to('cuda') #加载pipline
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)


    input_mean_emb = load_sentence_embeddings([args.input_sentence], pipe.tokenizer, pipe.text_encoder, device='cuda')  #tensor类型
    output_mean_emb = load_sentence_embeddings([args.output_sentence], pipe.tokenizer, pipe.text_encoder, device='cuda') #tensor类型
    
    
    img = Image.open(args.input_image).resize((512,512), Image.Resampling.LANCZOS)
    x_inv, x_inv_image, x_dec_img = pipe1(
        args.input_sentence,
        guidance_scale=1,
        num_inversion_steps=args.num_ddim_steps,
        img=img,
        torch_dtype=torch_dtype
    )
    edit_dir=(output_mean_emb.mean(0)-input_mean_emb.mean(0)).unsqueeze(0)
    rec_pil, edit_pil = pipe(args.input_sentence,
        num_inference_steps=args.num_ddim_steps,
        x_in=x_inv,
        edit_dir=edit_dir,
        # edit_dir=construct_direction('ree2reeduringfall'),
        guidance_amount=args.xa_guidance,
        guidance_scale=args.negative_guidance_scale,
        negative_prompt=args.input_sentence, # use the unedited prompt for the negative prompt
        samples_times = args.samples_times

    )
    
    for i in range(0,args.samples_times):
        (edit_pil[i])[0].save(f"output/test/9.13/video/{i}_edit.png")




        
        
        
        



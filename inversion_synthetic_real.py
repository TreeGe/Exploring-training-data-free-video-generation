import os, pdb
import numpy as np
import argparse
import torch
from glob import glob
import requests
from PIL import Image
from diffusers import DDIMScheduler
from src.edit_pipeline_inter import EditingPipeline
from lavis.models import load_model_and_preprocess
from src.utils.ddim_inv import DDIMInversion
from src.utils.scheduler import DDIMInverseScheduler


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
    parser.add_argument('--num_ddim_steps', type=int, default=50)
    # parser.add_argument('--model_path', type=str, default='CompVis/stable-diffusion-v1-4')
    parser.add_argument('--model_path', type=str, default='model_path')
    parser.add_argument('--random_seed', default=8)
    parser.add_argument('--xa_guidance', default=0.1, type=float)
    parser.add_argument('--negative_guidance_scale', default=5.0, type=float)
    parser.add_argument('--use_float_16', action='store_true')
    parser.add_argument('--samples_times', type=int, default=4)
    
    parser.add_argument('--input_sentence', default="mountain in spring")
    parser.add_argument('--output_sentence', default="mountain in fall")

    args = parser.parse_args()

    if args.use_float_16:
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    
    
    # Make the editing pipeline  制作编辑管道,加载预训练好的模型
    pipe = EditingPipeline.from_pretrained(args.model_path, torch_dtype=torch_dtype).to(device) #加载pipline
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.safety_checker = lambda images, clip_input: (images, False)


    input_mean_emb = load_sentence_embeddings([args.input_sentence], pipe.tokenizer, pipe.text_encoder, device='cuda')
    output_mean_emb = load_sentence_embeddings([args.output_sentence], pipe.tokenizer, pipe.text_encoder, device='cuda')
    
    # make the input noise map
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.random_seed)
    else:
        torch.manual_seed(args.random_seed)

    x = torch.randn((1,4,64,64), device=device)
    
    # for inv_path, prompt_path in zip(l_inv_paths, l_prompt_paths):
    #     prompt_str = open(prompt_path).read().strip()  #原始图像的文本提示
    edit_dir=output_mean_emb.mean(0)-input_mean_emb.mean(0).unsqueeze(0)
    rec_pil, edit_pil = pipe(args.input_sentence,
        num_inference_steps=args.num_ddim_steps,
        x_in=x,
        edit_dir=edit_dir,
        guidance_amount=args.xa_guidance,
        guidance_scale=args.negative_guidance_scale,
        negative_prompt='',# use the unedited prompt for the negative prompt
        samples_times=args.samples_times
    )

        
    rec_pil[0].save(f"output/test/9.20/{args.random_seed}_rec.png")
    for i in range(0,args.samples_times):
        (edit_pil[i])[0].save(os.path.join(f"output/test/9.20/{args.random_seed}_{i}.png"))




        
        
        
        



import os
import argparse
import torch
import imageio
from PIL import Image
from diffusers import DDIMScheduler
from edit_pipeline_inter_flow import EditingPipeline
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_ddim_steps', type=int, default=50)
    parser.add_argument('--model_path', type=str, default='')
    parser.add_argument('--model_path', type=str, default='model_path')
    parser.add_argument('--xa_guidance', default=0.1, type=float)
    parser.add_argument('--negative_guidance_scale', default=5.0, type=float)
    parser.add_argument('--use_float_16', action='store_true')
    parser.add_argument('--samples_times', type=int, default=6)

    parser.add_argument('--input_img', default="output/test/real_image/girffe.png")
    parser.add_argument('--input_sentence', default="a giraffe standing next to a lush green field")
    parser.add_argument('--input', default="noon")
    parser.add_argument('--output', default="night")


    args = parser.parse_args()

    os.makedirs(os.path.join(args.results_folder, "edit"), exist_ok=True)
    os.makedirs(os.path.join(args.results_folder, "reconstruction"), exist_ok=True)

    if args.use_float_16:
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    pipe1 = DDIMInversion.from_pretrained(args.model_path, torch_dtype=torch_dtype).to(device)
    pipe1.scheduler = DDIMInverseScheduler.from_config(pipe1.scheduler.config)

    img = Image.open(args.input_img)
    x_inv, x_inv_image, x_dec_img = pipe1(
        args.input_sentence,
        guidance_scale=1,
        num_inversion_steps=args.num_ddim_steps,
        img=img,
        torch_dtype=torch_dtype
    )

    pipe = EditingPipeline.from_pretrained(args.model_path, torch_dtype=torch_dtype).to(device)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.safety_checker = lambda images, clip_input: (images, False)


    input_mean_emb = load_sentence_embeddings([args.input], pipe.tokenizer, pipe.text_encoder, device='cuda')
    output_mean_emb = load_sentence_embeddings([args.output], pipe.tokenizer, pipe.text_encoder, device='cuda')

    edit_dir = ((output_mean_emb.mean(0) - input_mean_emb.mean(0)).unsqueeze(0))
    rec_pil, edit_pil, edit_inter = pipe(args.input_sentence,
                                         num_inference_steps=args.num_ddim_steps,
                                         x_in=x_inv,
                                         edit_dir=edit_dir,
                                         guidance_amount=args.xa_guidance,
                                         guidance_scale=args.negative_guidance_scale,
                                          negative_prompt=args.input_sentence,
                                         samples_times=args.samples_times
                                         )

    rec_pil[0].save(f"args.results_folder/rec.png")
    for i in range(0, args.samples_times):
        (edit_pil[i]).save(os.path.join(f"args.results_folder/rec.png"))

    imageio.mimsave(f'args.results_folder/gene_default.gif', 'GIF', duration=250)

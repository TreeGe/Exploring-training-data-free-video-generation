import os
import pdb, sys

import numpy as np
import torch
from typing import Any, Callable, Dict, List, Optional, Union

from diffusers import DDIMScheduler
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput

sys.path.insert(0, "src/utils")
from src.utils.base_pipeline import BasePipeline
from src.utils.cross_attention import prep_unet
from numpy import arange
import math

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


class EditingPipeline(BasePipeline):
    def __call__(
            self,
            prompt: Union[str, List[str]] = None,
            height: Optional[int] = None,
            width: Optional[int] = None,
            num_inference_steps: int = 50,
            guidance_scale: float = 7.5,
            negative_prompt: Optional[Union[str, List[str]]] = None,
            num_images_per_prompt: Optional[int] = 1,
            eta: float = 0.0,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            latents: Optional[torch.FloatTensor] = None,
            prompt_embeds: Optional[torch.FloatTensor] = None,
            negative_prompt_embeds: Optional[torch.FloatTensor] = None,
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,

            # pix2pix parameters
            guidance_amount=0.1,
            edit_dir=None,
            x_in=None,
            samples_times: int = 10,
            only_sample=False,  # only perform sampling, and no editing

    ):
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.cross_attention_kwargs = cross_attention_kwargs
        self.guidance_amount = guidance_amount
        self.edit_dir = edit_dir
        self.samples_times = samples_times
        x_in.to(dtype=self.unet.dtype, device=self._execution_device)  # 对应xt

        # 0. modify the unet to be useful :D
        self.unet = prep_unet(self.unet)  # 加载Unet2DConditionModel



        # 2. Default height and width to unet unet的默认高度和宽度 与采用的预训练模型有关
        height = height or self.unet.config.sample_size * self.vae_scale_factor  # 512
        width = width or self.unet.config.sample_size * self.vae_scale_factor  # 512

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        self.do_classifier_free_guidance = guidance_scale > 1.0
        self.x_in = x_in.to(dtype=self.unet.dtype, device=self._execution_device)  # 1，4，64，64
        # 3. Encode input prompt = 2x77x768  对提示进行编码？
        self.prompt_embeds = self._encode_prompt(prompt, device, num_images_per_prompt, self.do_classifier_free_guidance,
                                            negative_prompt, prompt_embeds=prompt_embeds,
                                            negative_prompt_embeds=negative_prompt_embeds, )
        # 2×77×768 promt是根据图像通过BLIP生成的文本描述
        # prompt编码有区别:promp = （输入的文本 / 提取的文本） negative_prompt = (None / 提取的文本)

        # 4. Prepare timesteps 准备时间步长
        self.scheduler.set_timesteps(num_inference_steps, device=device)  # 从1000步忠选取一定的步数
        self.timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables  准备潜在的变量
        num_channels_latents = self.unet.in_channels

        # randomly sample a latent code if not provided 如果没有提供，随机抽样潜在代码
        # 随机生成一个潜在向量？ 这里给了潜在向量(经过inversion操作得到的加噪图像潜在向量)
        self.latents = self.prepare_latents(batch_size * num_images_per_prompt, num_channels_latents, height, width,
                                       self.prompt_embeds.dtype, device, generator, x_in, )
        # 1,4,64,64

        self.latents_init = self.latents.clone()
        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        self.extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
    def first_step(self):
        # 1. setup all caching objects 设置所有缓存对象
            d_ref_t2attn = {}  # reference cross attention maps 参考交叉注意力地图 存放原文本与x(t->1)之间的注意力图
            # 7. First Denoising loop for getting the reference cross attention maps  用于获得参考交叉注意力图的第一去噪循环->得到源文本和加噪图像之间的注意力图
            num_warmup_steps = len(self.timesteps) - self.num_inference_steps * self.scheduler.order  # 0
            with torch.no_grad():
                with self.progress_bar(total=self.num_inference_steps) as progress_bar:
                    for i, t in enumerate(self.timesteps):  # 采样50步
                        # expand the latents if we are doing classifier free guidance
                        latent_model_input = torch.cat([self.latents] * 2) if self.do_classifier_free_guidance else self.latents
                        # latent_model_input:[2,4,64,64] 复制一个？
                        latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                        # predict the noise residual
                        noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=self.prompt_embeds,
                                               cross_attention_kwargs=self.cross_attention_kwargs, ).sample
                        # 通过unet预测噪声？

                        # add the cross attention map to the dictionary  在字典中添加交叉注意力图
                        d_ref_t2attn[t.item()] = {}
                        for name, module in self.unet.named_modules():
                            module_name = type(module).__name__
                            if module_name == "CrossAttention" and 'attn2' in name:  # 不进入
                                attn_mask = module.attn_probs  # size is num_channel （16，4096，77）,s*s,77
                                d_ref_t2attn[t.item()][name] = attn_mask.detach().cpu()  # 记录下当前步长源文本和图像之间的注意力图

                        # perform guidance
                        if self.do_classifier_free_guidance:
                            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                            # 将noise_pred拆分成两部分 noise_pred_uncond1=noise_pred_text
                            noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                        # compute the previous noisy sample x_t -> x_t-1
                        latents = self.scheduler.step(noise_pred, t, latents, **self.extra_step_kwargs).prev_sample
                        # 逆采样的过程？

                        # call the callback, if provided
                        if i == len(self.timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                            progress_bar.update()

            # make the reference image (reconstruction)
            image_rec = self.numpy_to_pil(self.decode_latents(latents.detach()))  # 通过扩散模型采样到的图像
            return image_rec,d_ref_t2attn


    def second_step(self,d_ref_t2attn):
        images = []
        np1 = list(arange(1, math.e, (math.e - 1) / self.samples_times, 'd') + (math.e - 1) / self.samples_times)
        for i in range(1, self.samples_times + 1):
            edit_temp = self.edit_dir.clone()
            edit_temp = edit_temp * math.log(np1[i - 1])
            print(edit_temp)
            prompt_embeds_edit = self.prompt_embeds.clone()  # 2×77×768  原图像文本提示的编码
            prompt_embeds_edit[1:2] += edit_temp
            # 编辑方向加入prompt_embeds_edit

            latents = self.latents_init  # 源图像的加入步长为50的噪声
            # Second denoising loop for editing the text prompt 用于编辑文本提示的第二次去噪循环
            num_warmup_steps = len(self.timesteps) - self.num_inference_steps * self.scheduler.order  # 0
            with self.progress_bar(total=self.num_inference_steps) as progress_bar:
                for i, t in enumerate(self.timesteps):
                    # expand the latents if we are doing classifier free guidance 如果我们正在做分类器自由引导，扩展潜在的
                    latent_model_input = torch.cat([self.latents] * 2) if self.do_classifier_free_guidance else self.latents
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                    x_in = latent_model_input.detach().clone()
                    x_in.requires_grad = True

                    opt = torch.optim.SGD([x_in], lr=self.guidance_amount)

                    # predict the noise residual 预测噪声残差
                    # 送入的是加噪后图像的潜在向量，原文本描述和编辑方向的CLIP embedding
                    noise_pred = self.unet(x_in, t, encoder_hidden_states=prompt_embeds_edit.detach(),
                                           cross_attention_kwargs=self.cross_attention_kwargs, ).sample

                    loss = 0.0  # 计算源文本和图像注意力图和编辑方向与图像的注意力图的L2损失
                    for name, module in self.unet.named_modules():
                        module_name = type(module).__name__
                        if module_name == "CrossAttention" and 'attn2' in name:
                            curr = module.attn_probs  # size is num_channel,s*s,77 #当前的注意力图
                            ref = d_ref_t2attn[t.item()][name].detach().to(device)  # 之前的注意力图
                            loss += ((curr - ref) ** 2).sum((1, 2)).mean(0)
                            # loss += (abs(curr - ref)).sum((1, 2)).mean(0) + 0*((curr-ref)**2).sum((1,2)).mean(0)
                    loss.backward(retain_graph=False)
                    opt.step()

                    # recompute the noise
                    with torch.no_grad():
                        noise_pred = self.unet(x_in.detach(), t, encoder_hidden_states=prompt_embeds_edit,
                                               cross_attention_kwargs=self.cross_attention_kwargs, ).sample

                    latents = x_in.detach().chunk(2)[0]

                    # perform guidance
                    if self.do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                    # compute the previous noisy sample x_t -> x_t-1
                    latents = self.scheduler.step(noise_pred, t, latents, **self.extra_step_kwargs).prev_sample

                    # call the callback, if provided
                    if i == len(self.timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                        progress_bar.update()

            # 8. Post-processing
            image = self.decode_latents(latents.detach())

            # 9. Run safety checker
            image, has_nsfw_concept = self.run_safety_checker(image, device, self.prompt_embeds.dtype)

            # 10. Convert to PIL
            image_edit = self.numpy_to_pil(image)
            images.append(image_edit)

        return  images

    def forward(self):
        image_rec , d_ref_t2attn = self.first_step()
        images = self.second_step(d_ref_t2attn)
        return image_rec,images

# def editpipe():
#     pipe = EditingPipeline.from_pretrained('model_path', torch_dtype=torch.float16).to(device)
#     pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
#     pipe.safety_checker = lambda images, clip_input: (images, False)
#
#
#     return pipe

# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
import cv2
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
from IPython.display import display
from tqdm.notebook import tqdm


def text_under_image(image: np.ndarray, text: str, text_color: Tuple[int, int, int] = (0, 0, 0)):
    h, w, c = image.shape
    offset = int(h * .2)
    img = np.ones((h + offset, w, c), dtype=np.uint8) * 255
    font = cv2.FONT_HERSHEY_SIMPLEX
    # font = ImageFont.truetype("/usr/share/fonts/truetype/noto/NotoMono-Regular.ttf", font_size)
    img[:h] = image
    textsize = cv2.getTextSize(text, font, 1, 2)[0]
    text_x, text_y = (w - textsize[0]) // 2, h + offset - textsize[1] // 2
    cv2.putText(img, text, (text_x, text_y ), font, 1, text_color, 2)
    return img


def view_images(images, num_rows=1, offset_ratio=0.02):
    if type(images) is list:
        num_empty = len(images) % num_rows
    elif images.ndim == 4:
        num_empty = images.shape[0] % num_rows
    else:
        images = [images]
        num_empty = 0

    empty_images = np.ones(images[0].shape, dtype=np.uint8) * 255
    images = [image.astype(np.uint8) for image in images] + [empty_images] * num_empty
    num_items = len(images)

    h, w, c = images[0].shape
    offset = int(h * offset_ratio)
    num_cols = num_items // num_rows
    image_ = np.ones((h * num_rows + offset * (num_rows - 1),
                      w * num_cols + offset * (num_cols - 1), 3), dtype=np.uint8) * 255
    for i in range(num_rows):
        for j in range(num_cols):
            image_[i * (h + offset): i * (h + offset) + h:, j * (w + offset): j * (w + offset) + w] = images[
                i * num_cols + j]

    pil_img = Image.fromarray(image_)
    display(pil_img)


def diffusion_step(model, controller, latents, context, t, guidance_scale, low_resource=False):
    if low_resource:
        noise_pred_uncond = model.unet(latents, t, encoder_hidden_states=context[0])["sample"]
        noise_prediction_text = model.unet(latents, t, encoder_hidden_states=context[1])["sample"]
    else:
        latents_input = torch.cat([latents] * 2)
        noise_pred = model.unet(latents_input, t, encoder_hidden_states=context)["sample"]
        noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
    latents = model.scheduler.step(noise_pred, t, latents)["prev_sample"]
    latents = controller.step_callback(latents)
    return latents


def latent2image(vae, latents):
    latents = 1 / 0.18215 * latents
    image = vae.decode(latents)['sample']
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()
    image = (image * 255).astype(np.uint8)
    return image


def init_latent(latent, model, height, width, generator, batch_size):
    if latent is None:
        latent = torch.randn(
            (1, model.unet.in_channels, height // 8, width // 8),
            generator=generator,
        )
    latents = latent.expand(batch_size,  model.unet.in_channels, height // 8, width // 8).to(model.device)
    return latent, latents


@torch.no_grad()
def text2image_ldm(
    model,
    prompt:  List[str],
    controller,
    num_inference_steps: int = 50,
    guidance_scale: Optional[float] = 7.,
    generator: Optional[torch.Generator] = None,
    latent: Optional[torch.FloatTensor] = None,
):
    register_attention_control(model, controller)
    height = width = 256
    batch_size = len(prompt)
    
    uncond_input = model.tokenizer([""] * batch_size, padding="max_length", max_length=77, return_tensors="pt")
    uncond_embeddings = model.bert(uncond_input.input_ids.to(model.device))[0]
    
    text_input = model.tokenizer(prompt, padding="max_length", max_length=77, return_tensors="pt")
    text_embeddings = model.bert(text_input.input_ids.to(model.device))[0]
    latent, latents = init_latent(latent, model, height, width, generator, batch_size)
    context = torch.cat([uncond_embeddings, text_embeddings])
    
    model.scheduler.set_timesteps(num_inference_steps)
    for t in tqdm(model.scheduler.timesteps):
        latents = diffusion_step(model, controller, latents, context, t, guidance_scale)
    
    image = latent2image(model.vqvae, latents)
   
    return image, latent


@torch.no_grad()
def text2image_ldm_stable(
    model,
    controller,
    ref_image_latent: torch.FloatTensor = None,
    ref_image_embed: Optional[torch.FloatTensor] = None,  # from pre-trained image encoder
    prompt: Union[str, List[str]] = None,
    ref_prompt: Union[str, List[str]] = None,
    res_prompt_scale: float = 0.0,
    extra_ref_image_latents: Optional[List[torch.FloatTensor]] = None,
    extra_ref_image_embeds: Optional[List[torch.FloatTensor]] = None,
    extra_ref_image_scales: Optional[List[float]] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
    num_inference_steps: int = 50,
    guidance_scale: float = 5.0,
    guidance_scale_ref: float = 0.0,
    negative_prompt: Optional[Union[str, List[str]]] = None,
    num_images_per_prompt: Optional[int] = 1,
    eta: float = 0.0,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.FloatTensor] = None,
    negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    output_type: Optional[str] = "pil",
    return_dict: bool = True,
    callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
    callback_steps: int = 1,
    cross_attention_kwargs: Optional[Dict[str, Any]] = None,
    st: int = 1000,
    warm_up_ratio = 0.,
    warm_up_start_scale = 0.,
    refine_step: int = 0,
    refine_eta: float = 1.,
    refine_emb_scale: float = 0.8,
    refine_guidance_scale: float = 3.0,
):
    register_attention_control(model, controller)
    
    # 0. Default height and width to unet
    height = model.unet.config.sample_size * model.vae_scale_factor
    width = model.unet.config.sample_size * model.vae_scale_factor

    # 1. Check inputs. Raise error if not correct
    model.check_inputs(
            prompt, height, width, callback_steps, negative_prompt, negative_prompt_embeds
        )
    
     # 2. Define call parameters
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        raise ValueError("prompt_emeds not suppported, please use prompt (string) or list of prompt")
        
    device = model._execution_device
    
    do_classifier_free_guidance = guidance_scale > 1.0
    
    # 4. Prepare timesteps
    model.scheduler.set_timesteps(num_inference_steps, device = device)
    timesteps = model.scheduler.timesteps
    
    # 5. Prepare latent variables
    num_channels_latents = model.unet.in_channels
    latents = model.prepare_latents(
        batch_size * num_images_per_prompt,
        num_channels_latents,
        height,
        width,
        ref_image_embed.dtype,
        device,
        generator,
        latents,
    )
    
    # 6. Prepare extra step kwargs.
    extra_step_kwargs = model.prepare_extra_step_kwargs(generator, eta)
    
    # 7. Denoising loop
    num_warmup_steps = len(timesteps) - num_inference_steps * model.scheduler.order
    with model.progress_bar(total = num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            if t <= st:
                # expand the latents if we are doing classifier free guidance
                if ref_prompt is not None:
                    latent_model_input = torch.cat([latents] * 3)
                else:
                    latent_model_input = torch.cat([latents] * 2)
                latent_model_input = model.scheduler.scale_model_input(latent_model_input, t)
                
                # generate epsilon conditioned on input image and reference prompt
                if model.promptnet.config.with_noise:
                    cond_latents = model.scheduler.scale_model_input(latents, t)
                else:
                    cond_latents = ref_image_latent
                
                if ref_prompt is not None or refine_step > 0:
                    # scheduler has to be DDIM
                    prev_t = t - model.scheduler.config.num_train_timesteps // model.scheduler.num_inference_steps
                    alpha_prod_t = model.scheduler.alphas_cumprod[t]
                    alpha_prod_t_prev = model.scheduler.alphas_cumprod[prev_t] if prev_t >= 0 else model.scheduler.final_alpha_cumprod
                    variance = model.scheduler._get_variance(t, prev_t)
                    sigma_t = refine_eta * variance ** (0.5)
                    
                    for _ in range(refine_step):
                        noise = torch.randn_like(latents)
                        noise_pred_text_0, noise_pred_uncond, noise_pred_ref = \
                        model.generate_epsilons(input_img = ref_image_latent, input_img_clip_embed = ref_image_embed,
                                                time = t,
                                                promptnet_cond = cond_latents, prompt = prompt,
                                                ref_prompt = ref_prompt,
                                                latent_model_input = latent_model_input,
                                                negative_prompt = negative_prompt,
                                                negative_prompt_embeds = negative_prompt_embeds,
                                                num_images_per_prompt = num_images_per_prompt, device = device,
                                                cross_attention_kwargs = cross_attention_kwargs,
                                                scale = refine_emb_scale
                                               )
                        eps = refine_guidance_scale * (noise_pred_text_0 - noise_pred_uncond) + noise_pred_uncond
                        if extra_ref_image_latents is not None:
                            guidance_sum = np.sum(extra_ref_image_scales) + guidance_scale
                            eps = eps * guidance_scale / guidance_sum
                            assert len(extra_ref_image_latents) == len(extra_ref_image_scales) == len(extra_ref_image_embeds)
                            for extra_ind in range(len(extra_ref_image_latents)):
                                ref_image_embed_ = extra_ref_image_embeds[extra_ind]
                                ref_image_latent_ = extra_ref_image_latents[extra_ind]
                                extra_guidance_scale_ = extra_ref_image_scales[extra_ind]
                                noise_pred_text_0 = noise_pred_uncond, _ = \
                                model.generate_epsilons(input_img = ref_image_latent_,
                                                        input_image_clip_embed = ref_image_embed_, time = t,
                                                        promptnet_cond = cond_latents, prompt = prompt,
                                                        ref_prompt = ref_prompt,
                                                        latent_model_input = latent_model_input,
                                                        negative_prompt = negative_prompt,
                                                        negative_prompt_embeds = negative_prompt_embeds,
                                                        num_images_per_prompt = num_images_per_prompt,
                                                        device = device,
                                                        cross_attention_kwargs = cross_attention_kwargs,
                                                        scale = refine_emb_scale,
                                                       )
                                eps += (refine_guidance_scale * (noise_pred_text_0 - noise_pred_uncond) + noise_pred_uncond) * extra_guidance_scale_/guidance_sum
                                                    
                        latents = latents - eps * (sigma_t ** 2 * torch.sqrt(1 - alpha_prod_t))/(1 - alpha_prod_t_prev) \
                                      + sigma_t * noise * torch.sqrt((1 - alpha_prod_t) * (2 - 2*alpha_prod_t_prev - sigma_t**2))/(1-alpha_prod_t_prev)
                        if ref_prompt is not None:
                                latent_model_input = torch.cat([latents] * 3)
                        else:
                            latent_model_input = torch.cat([latents] * 2)
                        latent_model_input = model.scheduler.scale_model_input(latent_model_input, t)

                        # generate epsilon conditioned on input image and reference prompt
                        if model.promptnet.config.with_noise:
                            cond_latents = model.scheduler.scale_model_input(latents, t)
                        else:
                            cond_latents = ref_image_latent

                noise_pred_text_0, noise_pred_uncond, noise_pred_ref = \
                model.generate_epsilons(input_img=ref_image_latent, input_img_clip_embed=ref_image_embed, time=t,
                                               promptnet_cond=cond_latents, prompt=prompt, ref_prompt=ref_prompt,
                                               latent_model_input=latent_model_input,
                                               negative_prompt=negative_prompt,
                                               negative_prompt_embeds=negative_prompt_embeds,
                                               num_images_per_prompt=num_images_per_prompt, device=device,
                                               cross_attention_kwargs=cross_attention_kwargs,
                                      )
                
                s_1 = guidance_scale
                s_2 = guidance_scale_ref
                
                noise_pred = noise_pred_uncond + s_1 * (
                    noise_pred_text_0 - noise_pred_uncond) \
                + s_2 * (noise_pred_ref - noise_pred_uncond)
                
                # if we have some extra image conditions
                if extra_ref_image_latents is not None:
                    assert len(extra_ref_image_latents) == len(extra_ref_image_scales) == len(extra_ref_image_embeds)
                    for extra_ind in range(len(extra_ref_image_latents)):
                        ref_image_embed_ = extra_ref_image_embeds[extra_ind]
                        ref_image_latent_ = extra_ref_image_latents[extra_ind]
                        extra_guidance_scale_ = extra_ref_image_scales[extra_ind]
                        
                        noise_pred_text_0, noise_pred_uncond, _ = \
                        model.generate_epsilons(input_img = ref_image_latent_,
                                                input_img_clip_embed = ref_image_embed_, time = t,
                                                promptnet_cond = cond_latents, prompt = prompt,
                                                ref_prompt = ref_prompt,
                                                latent_model_input = latent_model_input,
                                                negative_prompt = negative_prompt,
                                                negative_prompt_embeds = negative_prompt_embeds,
                                                num_images_per_prompt = num_images_per_prompt, device = device,
                                                cross_attention_kwargs = cross_attention_kwargs,
                                               )
                        s_1_ = s_1 * extra_guidance_scale_ / guidance_scale
                        noise_pred += s_1_ * (noise_pred_text_0 - noise_pred_uncond)
                   
                latents = model.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample
                latents = controller.step_callback(latents)
                
                
    with torch.no_grad():
        if output_type == "latent":
            image = latents
            has_nsfw_concept = None
        elif output_type == "pil":
            # 8. post-processing
            image = model.decode_latents(latents)
            
            # 9. Run safety checker
            image, has_nsfw_concept = model.run_safety_checker(image, device, noise_pred_text_0.dtype)
            
            # 10. Convert to PIL
            image = model.numpy_to_pil(image)
            
        else:
            # 8. Post-processing
            image = model.decode_latents(latents)
            
            # 9. Run safety checker
            image, has_nsfw_concept = model.run_safety_checker(image, device, nosie_pred_text_0.dtype)
            
        # Offload last model to CPU
        if hasattr(model, "final_offload_hook") and model.final_offload_hook is not None:
            model.final_offload_hook.offload()
            
        if not return_dict:
            return (image, has_nsfw_concept)
        
    return image

def register_attention_control(model, controller):
    def ca_forward(self, place_in_unet):
        to_out = self.to_out
        if type(to_out) is torch.nn.modules.container.ModuleList:
            to_out = self.to_out[0]
        else:
            to_out = self.to_out

        def forward(hidden_state, encoder_hidden_states, attention_mask, **cross_attention_kwargs):
            batch_size, sequence_length, dim = hidden_state.shape
            h = self.heads
            q = self.to_q(hidden_state)
            is_cross = encoder_hidden_states is not None
            context = encoder_hidden_states if is_cross else hidden_state
            k = self.to_k(context)
            v = self.to_v(context)
            
            q = self.head_to_batch_dim(q)
            k = self.head_to_batch_dim(k)
            v = self.head_to_batch_dim(v)
            
            sim = torch.einsum("b i d, b j d -> b i j", q, k) * self.scale

            if attention_mask is not None:
                attention_mask = attention_mask.reshape(batch_size, -1)
                max_neg_value = -torch.finfo(sim.dtype).max
                mask = attention_mask[:, None, :].repeat(h, 1, 1)
                sim.masked_fill_(~mask, max_neg_value)

            # attention, what we cannot get enough of
            attn = sim.softmax(dim=-1)
            attn = controller(attn, is_cross, place_in_unet)
            out = torch.einsum("b i j, b j d -> b i d", attn, v)
            out = self.batch_to_head_dim(out)
            return to_out(out)

        return forward

    class DummyController:

        def __call__(self, *args):
            return args[0]

        def __init__(self):
            self.num_att_layers = 0

    if controller is None:
        controller = DummyController()

    def register_recr(net_, count, place_in_unet):
        if net_.__class__.__name__ == 'CrossAttention':
            net_.forward = ca_forward(net_, place_in_unet)
            return count + 1
        elif hasattr(net_, 'children'):
            for net__ in net_.children():
                count = register_recr(net__, count, place_in_unet)
        return count

    cross_att_count = 0
    sub_nets = model.unet.named_children()
    for net in sub_nets:
        if "down" in net[0]:
            cross_att_count += register_recr(net[1], 0, "down")
        elif "up" in net[0]:
            cross_att_count += register_recr(net[1], 0, "up")
        elif "mid" in net[0]:
            cross_att_count += register_recr(net[1], 0, "mid")

    controller.num_att_layers = cross_att_count

    
def get_word_inds(text: str, word_place: int, tokenizer):
    split_text = text.split(" ")
    if type(word_place) is str:
        word_place = [i for i, word in enumerate(split_text) if word_place == word]
    elif type(word_place) is int:
        word_place = [word_place]
    out = []
    if len(word_place) > 0:
        words_encode = [tokenizer.decode([item]).strip("#") for item in tokenizer.encode(text)][1:-1]
        cur_len, ptr = 0, 0

        for i in range(len(words_encode)):
            cur_len += len(words_encode[i])
            if ptr in word_place:
                out.append(i + 1)
            if cur_len >= len(split_text[ptr]):
                ptr += 1
                cur_len = 0
    return np.array(out)


def update_alpha_time_word(alpha, bounds: Union[float, Tuple[float, float]], prompt_ind: int,
                           word_inds: Optional[torch.Tensor]=None):
    if type(bounds) is float:
        bounds = 0, bounds
    start, end = int(bounds[0] * alpha.shape[0]), int(bounds[1] * alpha.shape[0])
    if word_inds is None:
        word_inds = torch.arange(alpha.shape[2])
    alpha[: start, prompt_ind, word_inds] = 0
    alpha[start: end, prompt_ind, word_inds] = 1
    alpha[end:, prompt_ind, word_inds] = 0
    return alpha


def get_time_words_attention_alpha(prompts, num_steps,
                                   cross_replace_steps: Union[float, Dict[str, Tuple[float, float]]],
                                   tokenizer, max_num_words=77):
    if type(cross_replace_steps) is not dict:
        cross_replace_steps = {"default_": cross_replace_steps}
    if "default_" not in cross_replace_steps:
        cross_replace_steps["default_"] = (0., 1.)
    alpha_time_words = torch.zeros(num_steps + 1, len(prompts) - 1, max_num_words)
    for i in range(len(prompts) - 1):
        alpha_time_words = update_alpha_time_word(alpha_time_words, cross_replace_steps["default_"],
                                                  i)
    for key, item in cross_replace_steps.items():
        if key != "default_":
            inds = [get_word_inds(prompts[i], key, tokenizer) for i in range(1, len(prompts))]
            for i, ind in enumerate(inds):
                 if len(ind) > 0:
                    alpha_time_words = update_alpha_time_word(alpha_time_words, item, i, ind)
    alpha_time_words = alpha_time_words.reshape(num_steps + 1, len(prompts) - 1, 1, 1, max_num_words)
    return alpha_time_words

import torch
import os
import folder_paths
from transformers import CLIPTextModelWithProjection, CLIPTokenizer, T5EncoderModel, T5TokenizerFast
from diffusers import StableDiffusion3InstructPix2PixPipeline, AutoencoderKL, FlowMatchEulerDiscreteScheduler, SD3Transformer2DModel
import numpy as np
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"

folder_paths.folder_names_and_paths["ultraedit"] = ([os.path.join(folder_paths.models_dir, "ultraedit")], folder_paths.supported_pt_extensions)


def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)


class UltraEdit_ModelLoader_Zho:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_model": ("STRING", {"default": "BleachNick/SD3_UltraEdit_w_mask"}),
            }
        }

    RETURN_TYPES = ("UEMODEL",)
    RETURN_NAMES = ("pipe",)
    FUNCTION = "load_model"
    CATEGORY = "🏕️UltraEdit"
  
    def load_model(self, base_model):
        pipe = StableDiffusion3InstructPix2PixPipeline.from_pretrained(
            base_model,
            torch_dtype=torch.float16,
        ).to(device)
        return [pipe]


class UltraEdit_ModelLoader_local_Zho:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_model": (folder_paths.get_filename_list("ultraedit"), ),
            }
        }

    RETURN_TYPES = ("UEMODEL",)
    RETURN_NAMES = ("pipe",)
    FUNCTION = "load_model"
    CATEGORY = "🏕️UltraEdit"
  
    def load_model(self, base_model):
        if not base_model:
            raise ValueError("Please provide the aurasr_model parameter with the name of the model file.")

        ultraedit_path = folder_paths.get_full_path("ultraedit", base_model)
        print(ultraedit_path)

        # 获取当前工作目录
        current_dir = os.path.dirname(os.path.abspath(__file__))

        # 绝对路径加载 text_encoder
        text_encoder_path = os.path.join(current_dir, "../../models/ultraedit/text_encoder")
        text_encoder_2_path = os.path.join(current_dir, "../../models/ultraedit/text_encoder_2")
        text_encoder_3_path = os.path.join(current_dir, "../../models/ultraedit/text_encoder_3")

        text_encoder = CLIPTextModelWithProjection.from_pretrained(text_encoder_path)
        text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(text_encoder_2_path)
        text_encoder_3 = T5EncoderModel.from_pretrained(text_encoder_3_path)

        # 绝对路径加载 vae
        vae_path = os.path.join(current_dir, "../../models/ultraedit/vae")
        vae = AutoencoderKL.from_pretrained(vae_path)
        
        # 绝对路径加载 transformer
        transformer_path = os.path.join(current_dir, "../../models/ultraedit/transformer")
        transformer = SD3Transformer2DModel.from_pretrained(transformer_path)

        # 绝对路径加载 tokenizer
        tokenizer_path = os.path.join(current_dir, "../../models/ultraedit/tokenizer")
        tokenizer_2_path = os.path.join(current_dir, "../../models/ultraedit/tokenizer_2")
        tokenizer_3_path = os.path.join(current_dir, "../../models/ultraedit/tokenizer_3")

        tokenizer = CLIPTokenizer.from_pretrained(tokenizer_path)
        tokenizer_2 = CLIPTokenizer.from_pretrained(tokenizer_2_path)
        tokenizer_3 = T5TokenizerFast.from_pretrained(tokenizer_3_path)

        # 绝对路径加载 scheduler
        scheduler_path = os.path.join(current_dir, "../../models/ultraedit/scheduler")
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(scheduler_path)

        
        pipe = StableDiffusion3InstructPix2PixPipeline.from_single_file(
            ultraedit_path,
            transformer=transformer,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            text_encoder_3=text_encoder_3,
            vae=vae,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            tokenizer_3=tokenizer_3,
            scheduler=scheduler,
            torch_dtype=torch.float16,
        ).to(device)
        return [pipe]


class UltraEdit_Generation_Zho:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipe": ("UEMODEL",),
                "image": ("IMAGE",), 
                "positive": ("STRING", {"default": "cat", "multiline": True}),
                "negative": ("STRING", {"default": "worst quality, low quality", "multiline": True}), 
                "steps": ("INT", {"default": 50, "min": 1, "max": 100, "step": 1}),
                "image_guidance_scale": ("FLOAT", {"default": 1.5, "min": 0, "max": 2.5}),
                "text_guidance_scale": ("FLOAT", {"default": 7.5, "min": 0, "max": 12.5}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "mask": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate_image"
    CATEGORY = "🏕️UltraEdit"
                       
    def generate_image(self, pipe, image, positive, negative, steps, seed, image_guidance_scale, text_guidance_scale, mask=None):

        generator = torch.Generator(device=device).manual_seed(seed)
        
        image_t=tensor2pil(image).resize((512, 512))
        
        if mask is None:
            mask_t = Image.new("RGB", image_t.size, (255, 255, 255))
        else:
            mask_t = tensor2pil(mask).resize((512, 512))
        
        output = pipe(
            prompt=positive,
            negative_prompt=negative,
            image=image_t,
            mask_img=mask_t,
            num_inference_steps=steps,
            image_guidance_scale=image_guidance_scale,
            guidance_scale=text_guidance_scale,
            generator=generator,
        )[0]
        
        output_t = pil2tensor(output)
        output_t = output_t.squeeze(0)
        print(output_t.shape)
        
        return (output_t,)


NODE_CLASS_MAPPINGS = {
    "UltraEdit_ModelLoader_Zho": UltraEdit_ModelLoader_Zho,
    "UltraEdit_ModelLoader_local_Zho": UltraEdit_ModelLoader_local_Zho,
    "UltraEdit_Generation_Zho": UltraEdit_Generation_Zho
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "UltraEdit_ModelLoader_Zho": "🏕️UltraEdit Model(auto) Zho",
    "UltraEdit_ModelLoader_local_Zho": "🏕️UltraEdit Model(local) Zho",
    "UltraEdit_Generation_Zho": "🏕️UltraEdit Generation Zho"
}

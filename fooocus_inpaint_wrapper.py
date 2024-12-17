import os
import os.path
import sys
import json
import requests
from tqdm import tqdm
from huggingface_hub import snapshot_download

class FooocusInpaintWrapper:
	def __init__(self):
		self.node_dir = os.path.normpath(os.path.dirname(os.path.realpath(__file__)))
		self.comfyui_dir = os.path.normpath(self.node_dir.split("custom_nodes")[0])
		self.fooocus_dir = os.path.normpath(self.node_dir + "/Fooocus")

		if not os.path.isfile(os.path.normpath(self.fooocus_dir + "/config.txt")):
			config = {
				"path_checkpoints": [
					os.path.normpath(self.comfyui_dir + "/models/checkpoints")
				],
				"path_loras": [
					os.path.normpath(self.comfyui_dir + "/models/loras")
				],
				"path_embeddings": os.path.normpath(self.comfyui_dir + "/models/embeddings"),
				"path_vae_approx": os.path.normpath(self.comfyui_dir + "/models/vae_approx"),
				"path_vae": os.path.normpath(self.comfyui_dir + "/models/vae"),
				"path_upscale_models": os.path.normpath(self.comfyui_dir + "/models/upscale_/models"),
				"path_inpaint": os.path.normpath(self.comfyui_dir + "/models/inpaint"),
				"path_controlnet": os.path.normpath(self.comfyui_dir + "/models/controlnet"),
				"path_clip_vision": os.path.normpath(self.comfyui_dir + "/models/clip_vision"),
				"path_fooocus_expansion": os.path.normpath(self.comfyui_dir + "/models/prompt_expansion/fooocus_expansion"),
				"path_wildcards": os.path.normpath(self.comfyui_dir + "/wildcards"),
				"path_safety_checker": os.path.normpath(self.comfyui_dir + "/models/safety_checker"),
				"path_sam": os.path.normpath(self.comfyui_dir + "/models/sam"),
				"path_outputs": os.path.normpath(self.fooocus_dir + "/outputs")
			}

			with open(os.path.normpath(self.fooocus_dir + "/config.txt"), "w") as f:
				json.dump(config, f, indent=4)

		self.find_replace(os.path.normpath(self.fooocus_dir + "/modules/config.py"), './config.txt', os.path.normpath(self.fooocus_dir + "/config.txt"))
		self.find_replace(os.path.normpath(self.fooocus_dir + "/modules/config.py"), './presets/default.json', os.path.normpath(self.fooocus_dir + "/presets/default.json"))
		self.find_replace(os.path.normpath(self.fooocus_dir + "/args_manager.py"), 'args_parser.args = args_parser.parser.parse_args()', 'args_parser.args, unknown = args_parser.parser.parse_known_args()')

		snapshot_download(repo_id="LykosAI/GPT-Prompt-Expansion-Fooocus-v2", local_dir = os.path.normpath(self.comfyui_dir + '/models/prompt_expansion/fooocus_expansion'))

		self.file_download('https://huggingface.co/lllyasviel/misc/resolve/main/xlvaeapp.pth?download=true', os.path.normpath(self.comfyui_dir + '/models/vae_approx/xlvaeapp.pth'))
		self.file_download('https://huggingface.co/lllyasviel/fav_models/resolve/main/fav/juggernautXL_v8Rundiffusion.safetensors?download=true', os.path.normpath(self.comfyui_dir + '/models/checkpoints/juggernautXL_v8Rundiffusion.safetensors'))
		self.file_download('https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_offset_example-lora_1.0.safetensors?download=true', os.path.normpath(self.comfyui_dir + '/models/loras/sd_xl_offset_example-lora_1.0.safetensors'))

	def file_download(self, url, save_path):
		if os.path.isfile(save_path):
			return True

		response = requests.get(url, stream=True)

		if response.status_code == 200:
		    total_size = int(response.headers.get('content-length', 0))
		    
		    # Usa tqdm per mostrare la barra di avanzamento
		    with open(save_path, 'wb') as file, tqdm(
		        desc=save_path.split("models")[-1],
		        total=total_size,
		        unit='B',
		        unit_scale=True,
		    ) as bar:
		        for chunk in response.iter_content(chunk_size=1024):
		            file.write(chunk)
		            bar.update(len(chunk))
		else:
		    print(f"Error: {response.status_code}")

	def find_replace(self, file_path, find_text, replace_text):
		with open(file_path, 'r') as file:
			content = file.read()

		new_content = content.replace(find_text, replace_text.replace("\\","\\\\"))

		with open(file_path, 'w') as file:
			file.write(new_content)
	
	@classmethod
	def INPUT_TYPES(s):
		def get_files(folder):
			checkpoint_dir = os.path.normpath(os.path.dirname(os.path.realpath(__file__)).split("custom_nodes")[0] + "/models/" + folder)
			if os.path.exists(checkpoint_dir):
				files = [
					f for f in os.listdir(checkpoint_dir)
					if f.endswith(".ckpt") or f.endswith(".safetensors")
				]
			else:
				files = []
			return files if files else ["No checkpoints found"]
		
		checkpoints = get_files("checkpoints")
		loras = get_files("loras")

		loras.append("None")

		return {
			"required": {
				"image": ("IMAGE",),
				"mask": ("MASK",),
				"performance": (["Quality", "Speed", "Extreme Speed", "Lightning", "Hyper-SD"], {"default": "Speed"}),
				"checkpoint": (checkpoints, {"default": checkpoints[0]}),
				"prompt": ("STRING", {"multiline": True}),
				"guidance_scale": ("FLOAT", {"default": 4, "min": 0, "max": 30.0}),
				"image_sharpness": ("FLOAT", {"default": 2, "min": 0, "max": 30.0}),
				"seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "tooltip": "The random seed used for creating the noise."}),
			},
			"optional": {
				"negative_prompt": ("STRING", {"multiline": True}),
				"method": (["", "Improve Detail", "Modify content"],),
				"inpaint_additional_prompt": ("STRING", {"multiline": True}),
				"lora1": (loras, {"default": "None"}),
				"lora1_weight": ("FLOAT", {"default": 0.1, "min": -2, "max": 2}),
				"lora2": (loras, {"default": "None"}),
				"lora2_weight": ("FLOAT", {"default": 0.1, "min": -2, "max": 2}),
				"lora3": (loras, {"default": "None"}),
				"lora3_weight": ("FLOAT", {"default": 0.1, "min": -2, "max": 2}),
				"lora4": (loras, {"default": "None"}),
				"lora4_weight": ("FLOAT", {"default": 0.1, "min": -2, "max": 2}),
				"lora5": (loras, {"default": "None"}),
				"lora5_weight": ("FLOAT", {"default": 0.1, "min": -2, "max": 2}),
			},
		}
	 
	RETURN_TYPES = ("IMAGE",)
	RETURN_NAMES = ("image",)
 
	FUNCTION = "start_inpaint"
 
	#OUTPUT_NODE = False
 
	CATEGORY = "inpaint"
 
	def start_inpaint(self, image, mask, performance, checkpoint, prompt, guidance_scale, image_sharpness, seed, negative_prompt, method, inpaint_additional_prompt, lora1, lora1_weight, lora2, lora2_weight, lora3, lora3_weight, lora4, lora4_weight, lora5, lora5_weight):
		sys.path.append(self.fooocus_dir)
		from launch import fooocusinpaintlaunch
		new_image = fooocusinpaintlaunch(self.fooocus_dir, image, mask, performance, checkpoint, prompt, negative_prompt, guidance_scale, image_sharpness, seed, method, inpaint_additional_prompt, lora1, lora1_weight, lora2, lora2_weight, lora3, lora3_weight, lora4, lora4_weight, lora5, lora5_weight)
		return (new_image)
 
 
# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
	"FooocusInpaintWrapper": FooocusInpaintWrapper
}
 
# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
	"FirstNode": "Fooocus Inpaint Wrapper"
}
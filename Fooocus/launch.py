import os
import sys
import numpy as np
import torch
from modules import async_worker as worker
from PIL import Image
from io import BytesIO
from torchvision import transforms
import shutil

def fooocusinpaint_clear_folder(directory):
    for elemento in os.listdir(directory):
        percorso_elemento = os.path.join(directory, elemento)
        if os.path.isdir(percorso_elemento):
            shutil.rmtree(percorso_elemento)

def fooocusinpaint_image_to_tensor(image_path):
	img = Image.open(image_path)
	img = np.array(img).astype(np.float32) / 255.0
	img = torch.from_numpy(img)[None,]
	return ([img])

def fooocusinpaint_findpng(folder_path):
	for root, dirs, files in os.walk(folder_path):
		for file in files:
			if file.lower().endswith(".png"):
				return os.path.join(root, file)  
	return None  

def fooocusinpaint_tensor2np(image_array, save_format='png'):
	# Converti il tensore in un'immagine PIL
	image_array = Image.fromarray(np.clip(255. * image_array.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

	# Se l'immagine è in modalità 'LA' (luminanza e alfa), possiamo separare i canali
	if image_array.mode == 'LA':
		# Estrai il canale di luminanza (canale 0)
		luminance = image_array.getchannel(0)
		
		# Crea un'immagine RGB usando lo stesso canale di luminanza per R, G, B
		image_array = Image.merge('RGB', (luminance, luminance, luminance))
	elif image_array.mode == 'L':
		# Se l'immagine è già in scala di grigi ('L'), duplicala in tre canali
		image_array = image_array.convert('RGB')

	# Se il formato di salvataggio è specificato
	if save_format == 'jpg':
		# Salva l'immagine in formato JPG (RGB)
		with BytesIO() as img_byte_array:
			image_array.save(img_byte_array, format='JPEG')
			img_byte_array.seek(0)  # Torna all'inizio del byte stream
			# Leggi l'immagine direttamente da memoria
			image_array = np.array(Image.open(img_byte_array))
	elif save_format == 'png':
		# Se l'immagine ha un canale alfa, usala
		if image_array.mode == 'RGBA':
			with BytesIO() as img_byte_array:
				image_array.save(img_byte_array, format='PNG')
				img_byte_array.seek(0)  # Torna all'inizio del byte stream
				# Leggi l'immagine direttamente da memoria
				image_array = np.array(Image.open(img_byte_array))
		else:
			# Converti in formato RGBA per PNG
			image_array = image_array.convert('RGBA')
			with BytesIO() as img_byte_array:
				image_array.save(img_byte_array, format='PNG')
				img_byte_array.seek(0)  # Torna all'inizio del byte stream
				# Leggi l'immagine direttamente da memoria
				image_array = np.array(Image.open(img_byte_array))

	return image_array

def fooocusinpaintlaunch(fooocus_dir, image_array, mask_array, performance, checkpoint, positive_prompt, negative_prompt, guidance_scale, image_sharpness, seed, method, inpaint_additional_prompt, lora1, lora1_weight, lora2, lora2_weight, lora3, lora3_weight, lora4, lora4_weight, lora5, lora5_weight):
	sys.path.insert(0, fooocus_dir)

	image_array = fooocusinpaint_tensor2np(image_array)
	mask_array = fooocusinpaint_tensor2np(mask_array)

	#checkpoint = 'juggernautXL_v8Rundiffusion.safetensors'

	method_setting = [False, 'v2.6', 1, 0.618]
	if method == "Improve Detail":
		method_setting = [False, 'None', 0.5, 0]
	if method == "Modify content":
		method_setting = [True, 'v2.6', 1, 0]

	args = [False, positive_prompt, negative_prompt, ['Fooocus V2', 'Fooocus Enhance', 'Fooocus Negative'], performance, '1152×896 <span style="color: grey;"> ∣ 9:7</span>', 1, 'png', seed, False, image_sharpness, guidance_scale, checkpoint, 'None', 0.5, True, lora1, lora1_weight, True, lora2, lora2_weight, True, lora3, lora3_weight, True, lora4, lora4_weight, True, lora5, lora5_weight, True, 'inpaint', 'Disabled', None, [], {'image': image_array, 'mask': mask_array}, inpaint_additional_prompt, None, False, False, False, False, 1.5, 0.8, 0.3, 7, 2, 'dpmpp_2m_sde_gpu', 'karras', 'Default (model)', -1, -1, -1, -1, -1, -1, False, False, False, False, 64, 128, 'joint', 0.25, False, 1.01, 1.02, 0.99, 0.95, False, method_setting[0], method_setting[1], method_setting[2], method_setting[3], False, False, 0, False, False, 'fooocus', None, 0.5, 0.6, 'ImagePrompt', None, 0.5, 0.6, 'ImagePrompt', None, 0.5, 0.6, 'ImagePrompt', None, 0.5, 0.6, 'ImagePrompt', False, 0, False, None, False, 'Disabled', 'Before First Enhancement', 'Original Prompts', False, '', '', '', 'sam', 'full', 'vit_b', 0.25, 0.3, 0, False, 'v2.6', 1, 0.618, 0, False, False, '', '', '', 'sam', 'full', 'vit_b', 0.25, 0.3, 0, False, 'v2.6', 1, 0.618, 0, False, False, '', '', '', 'sam', 'full', 'vit_b', 0.25, 0.3, 0, False, 'v2.6', 1, 0.618, 0, False]

	task = worker.AsyncTask(args=args)

	worker.async_tasks.append(task)
	worker.worker()

	final_image = fooocusinpaint_findpng(os.path.normpath(fooocus_dir+"/outputs"))
	tensor_image = fooocusinpaint_image_to_tensor(final_image)
	fooocusinpaint_clear_folder(os.path.normpath(fooocus_dir+"/outputs"))
	return tensor_image
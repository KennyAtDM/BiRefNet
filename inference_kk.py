import os
import argparse
from glob import glob
from tqdm import tqdm
import cv2
import torch
from torch import nn
from dataset import MyData
from models.birefnet import BiRefNet
from utils import save_tensor_img, check_state_dict , path_to_image
import requests
from PIL import Image
from io import BytesIO
from IPython.display import display
from config import Config


config = Config()

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:32'


model = BiRefNet(bb_pretrained=False)
state_dict = torch.load("BiRefNet-massive-epoch_240.pth", map_location='cpu')
state_dict = check_state_dict(state_dict)

# 逐层加载权重
model_dict = model.state_dict()
pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict and model_dict[k].size() == v.size()}
model_dict.update(pretrained_dict)
model.load_state_dict(model_dict)
model = model.to('cuda')

model.eval()
from torchvision import transforms


transform_image = transforms.Compose([
    transforms.Resize((1024,1024)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def tensor_to_pil(tenor_im):
    im = tenor_im.cpu().clone()
    im = im.squeeze(0)
    tensor2pil = transforms.ToPILImage()
    im = tensor2pil(im)
    return im

def singleImageoutput(image):
    secondimage = image.copy()
    input_images = transform_image(image).unsqueeze(0).to('cuda')
    with torch.no_grad():
        scaled_preds = model(input_images)[-1].sigmoid()
    for idx_sample in range(scaled_preds.shape[0]):
        res = nn.functional.interpolate(
            scaled_preds[idx_sample].unsqueeze(0),
            size=secondimage.size,
            mode='bilinear',
            align_corners=True
        )
    output=tensor_to_pil(res)   # test set dir + file name

    secondimage.putalpha(output)
    return secondimage

# imagepath = '/home/dm/KelingYaoDM/Blurry_image_classifier/blur_dataset/sharp/154_IPHONE-7_S.jpeg'
# image = Image.open(imagepath)
# output = singleImageoutput(image.resize((1024,1024))).resize(image.size)
# display(output)
# display(image)
# display(output.split()[3])


# Search for 'images' directories under the benchmark directory
image_dirs = glob("/mnt/data/siyan/benchmark/*/images")
output_dir = "benchmark"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Process all images found in each 'images' directory
for dir_path in tqdm(image_dirs):
    relative_path = os.path.relpath(dir_path, "/mnt/data/siyan/benchmark")  # Get the relative path from base
    output_path = os.path.join(output_dir, relative_path)  # Construct the output path
    
    for filename in os.listdir(dir_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(dir_path, filename)
            image = Image.open(image_path)
            output_image = singleImageoutput(image.resize((1024, 1024)))
            output_image = output_image.resize(image.size).split()[3]  # Resize back to original size if needed

            # Save the output image
            # Prepare save directory
            if not os.path.exists(output_path):
                os.makedirs(output_path)

            # Save the output image
            save_filename = os.path.join(output_path, filename)
            output_image.save(save_filename)

print("All images have been processed and saved.")
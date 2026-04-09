import os
import torch
from PIL import Image
import pandas as pd
import json
from torchvision.transforms.functional import InterpolationMode
import torchvision.transforms as torch_transforms

# Hard coded function to filter dataset, need to be removed
def filter_dataset(pd_data, model_name_or_path, cache_dir):
    from transformers import CLIPTokenizer
    tokenizer = CLIPTokenizer.from_pretrained(
        model_name_or_path, subfolder="tokenizer", cache_dir=cache_dir
    )
    ignore = []
    for i in range(len(pd_data)):
        prompt = pd_data.iloc[i].prompt
        if len(tokenizer(prompt)['input_ids']) > 60:
            ignore.append(i)
    return ignore

def _convert_image_to_rgb(image):
    return image.convert("RGB")

def get_transform(interpolation=InterpolationMode.BICUBIC, size=512):
    transform = torch_transforms.Compose([
        torch_transforms.Resize(size, interpolation=interpolation),
        torch_transforms.CenterCrop(size),
        _convert_image_to_rgb,
        torch_transforms.ToTensor(),
        torch_transforms.Normalize([0.5], [0.5])
    ])
    return transform

class PNGImageDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None, model_name_or_path=None, cache_dir='.cache'):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = []
        prompts_df = pd.read_csv(os.path.join(self.root_dir,'prompts.csv'))
        self.data = prompts_df[['prompt', 'evaluation_seed', 'evaluation_guidance']] if 'evaluation_seed' in prompts_df.columns else prompts_df[['prompt']]
        if os.path.exists(os.path.join(self.root_dir,'ignore.json')):
            with open(os.path.join(self.root_dir,'ignore.json')) as f:
                ignore = json.load(f)
        else:
            m = model_name_or_path or os.environ.get('DIFFUSION_MODEL_NAME_OR_PATH', './stable-diffusion-v1-4')
            c = cache_dir if cache_dir is not None else '.cache'
            ignore = filter_dataset(self.data, m, c)
            with open(os.path.join(self.root_dir,'ignore.json'), 'w') as f:
                json.dump(ignore, f)
        self.idxs = [i for i in range(len(self.data)) if i not in ignore]
        subdir = 'imgs'
        dir_path = os.path.join(self.root_dir, subdir)
        self.image_files.extend([(f,subdir) for f in os.listdir(dir_path) if f.endswith('.png')])
        self.image_files.sort(key=lambda x: int(x[0].split('_')[0]))

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, idx):
        idx = self.idxs[idx]
        img_path = self.image_files[idx][0]
        subdir = self.image_files[idx][1]
        img_path = os.path.join(self.root_dir,subdir, img_path)
        image = Image.open(img_path).convert("RGB")  # Reads image and returns a CHW tensor
        # image = TF.to_tensor(image)
        prompt = self.data.iloc[idx].prompt
        seed = self.data.iloc[idx].evaluation_seed if 'evaluation_seed' in self.data.columns else None
        guidance_scale = self.data.iloc[idx].evaluation_guidance if 'evaluation_guidance' in self.data.columns else 7.5  
        if self.transform:
            image = self.transform(image)
        
        return image, prompt, seed, guidance_scale

def get(root_dir, model_name_or_path=None, cache_dir='.cache'):
    return PNGImageDataset(
        root_dir=root_dir,
        transform=get_transform(),
        model_name_or_path=model_name_or_path,
        cache_dir=cache_dir,
    )
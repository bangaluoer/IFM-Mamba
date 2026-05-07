import argparse
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
import os
import glob
from tqdm import tqdm
import random
import numpy as np

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from basicsr.archs.mambairv2light_arch import MambaIRv2Light 

def load_model(weight_path, device='cuda'):
    model = MambaIRv2Light(
        upscale=1,
        in_chans=3,
        img_size=96,
        img_range=1.,
        embed_dim=180,
        d_state=16,
        depths=[6, 6, 6, 6, 6, 6],
        num_heads=[6, 6, 6, 6, 6, 6],
        window_size=16,
        inner_rank=64,
        num_tokens=128,
        convffn_kernel_size=5,
        mlp_ratio=4.0
    )
    
    checkpoint = torch.load(weight_path, map_location=device)
    state_dict = checkpoint['params'] if 'params' in checkpoint else checkpoint
    
    model.load_state_dict(state_dict, strict=True)
    model.eval().to(device)
    print(f"Model loaded successfully from {weight_path}")
    return model

def batch_inference():
    parser = argparse.ArgumentParser(description='Image Denoising Testing')
    parser.add_argument('--dataset_dir', type=str, required=True, help='Path to the clean dataset')
    parser.add_argument('--noise_level', type=float, default=15.0, help='Noise level (e.g., 15)')
    parser.add_argument('--weights', type=str, required=True, help='Path to the pre-trained weights')
    parser.add_argument('--save_dir', type=str, default='./results', help='Path to save the denoised images')
    args = parser.parse_args()

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    WINDOW_SIZE = 16 

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)


    model = load_model(args.weights, DEVICE)
    

    img_list = sorted(glob.glob(os.path.join(args.dataset_dir, '*.png')))
    if len(img_list) == 0:
        print(f"Error: No PNG images found in {args.dataset_dir}")
        return
        
    print(f"Found {len(img_list)} images. Starting inference with noise level {args.noise_level}...")

    with torch.no_grad():
        for img_path in tqdm(img_list):
            img_name = os.path.basename(img_path)
 
            img = Image.open(img_path).convert('RGB')
            clean_tensor = TF.to_tensor(img).unsqueeze(0).to(DEVICE)


            noise = torch.randn_like(clean_tensor) * (args.noise_level / 255.0)
            noisy_tensor = clean_tensor + noise

            _, _, h, w = noisy_tensor.size()
            mod_pad_h = (WINDOW_SIZE - h % WINDOW_SIZE) % WINDOW_SIZE
            mod_pad_w = (WINDOW_SIZE - w % WINDOW_SIZE) % WINDOW_SIZE
            

            img_padded = F.pad(noisy_tensor, (0, mod_pad_w, 0, mod_pad_h), 'reflect')

  
            output_padded = model(img_padded)

            output_tensor = output_padded[:, :, 0:h, 0:w]
            output_tensor = output_tensor.squeeze(0).clamp(0, 1)

            save_path = os.path.join(args.save_dir, img_name)
            TF.to_pil_image(output_tensor).save(save_path)
            
    print(f"Inference complete! Denoised images are saved in {args.save_dir}")

if __name__ == '__main__':
    batch_inference()

import os
import numpy as np
from PIL import Image
import random
import torchvision.transforms as transforms
from torchvision.transforms import functional as TF

input_dir = '...'
output_dir = 'aug/...'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def add_horizontal_gaussian_noise(image, noise_factor=0.01):
    """Add horizontal Gaussian noise to the image."""
    image_np = np.array(image)
    noise = np.random.randn(*image_np.shape) * noise_factor
    noisy_image = np.clip(image_np + noise, 0, 255).astype(np.uint8)
    return TF.to_pil_image(noisy_image)

aug_list = [
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.Lambda(lambda img: add_horizontal_gaussian_noise(img, noise_factor=0.01))
]

def mixup(image1, image2, alpha=0.2):
    """Mix two images together using Mixup augmentation."""
    lambda_param = np.random.beta(alpha, alpha)
    image1_resized = image1.resize((256, 256))
    image2_resized = image2.resize((256, 256))
    
    img1_array = np.array(image1_resized, dtype=np.float32)
    img2_array = np.array(image2_resized, dtype=np.float32)
    
    mixed_img_array = (lambda_param * img1_array + (1 - lambda_param) * img2_array)
    return Image.fromarray(np.uint8(np.clip(mixed_img_array, 0, 255)))

def augmix(image, width=3, depth=-1, alpha=1.0):
    ws = np.float32(np.random.dirichlet([alpha] * width))
    m = np.float32(np.random.beta(alpha, alpha))

    mix = np.zeros_like(np.array(image), dtype=np.float32)
    for i in range(width):
        image_aug = image.copy()
        d = depth if depth > 0 else np.random.randint(1, 4)
        for _ in range(d):
            op = np.random.choice(aug_list)
            image_aug = op(image_aug)
        mix += ws[i] * np.array(image_aug, dtype=np.float32)

    mixed = (1 - m) * np.array(image, dtype=np.float32) + m * mix
    return Image.fromarray(np.uint8(mixed))

def augment_images(input_dir, output_dir, num_augments=5):
    image_filenames = [f for f in os.listdir(input_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]

    for img_file in image_filenames:
        img_path = os.path.join(input_dir, img_file)
        img = Image.open(img_path).convert('RGB')
        
        for i in range(num_augments):
            aug_img = augmix(img)

            if len(image_filenames) > 1:
                other_img_file = random.choice(image_filenames)
                other_img_path = os.path.join(input_dir, other_img_file)
                other_img = Image.open(other_img_path).convert('RGB')

                mixup_img = mixup(aug_img, other_img)

                output_img_path = os.path.join(output_dir, f'{os.path.splitext(img_file)[0]}_aug_{i+1}.png')
                mixup_img.save(output_img_path, format='PNG')

augment_images(input_dir, output_dir, num_augments=196)
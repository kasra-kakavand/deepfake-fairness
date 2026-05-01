"""
Dataset module for fairness-aware deepfake detection.

This module provides dataset creation utilities including synthetic face generation,
manipulation simulation, demographic annotation, and bias introduction.
"""

import os
import glob
import random
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
import torch
from torch.utils.data import Dataset
from torchvision import transforms


# Demographic categories
SKIN_TONES = ['light', 'medium-light', 'medium', 'medium-dark', 'dark']
GENDERS = ['male', 'female']

# Skin tone color palette (RGB)
SKIN_TONE_COLORS = {
    'light': (255, 219, 172),
    'medium-light': (224, 172, 105),
    'medium': (198, 134, 66),
    'medium-dark': (141, 85, 36),
    'dark': (92, 51, 23)
}


def create_realistic_face(skin_tone='medium', size=224):
    """
    Create a synthetic face image with specified skin tone.
    
    Args:
        skin_tone (str): One of SKIN_TONES
        size (int): Image dimensions (size x size)
    
    Returns:
        PIL.Image: Generated face image
    """
    skin_color = SKIN_TONE_COLORS[skin_tone]
    img = Image.new('RGB', (size, size), color=skin_color)
    draw = ImageDraw.Draw(img)
    
    # Face oval
    draw.ellipse(
        [size//6, size//8, 5*size//6, 7*size//8],
        fill=skin_color, outline=(0, 0, 0), width=2
    )
    
    # Eyes
    eye_y = size // 3
    draw.ellipse([size//3-15, eye_y-10, size//3+15, eye_y+10], 
                 fill='white', outline=(0, 0, 0))
    draw.ellipse([2*size//3-15, eye_y-10, 2*size//3+15, eye_y+10], 
                 fill='white', outline=(0, 0, 0))
    draw.ellipse([size//3-5, eye_y-5, size//3+5, eye_y+5], fill='black')
    draw.ellipse([2*size//3-5, eye_y-5, 2*size//3+5, eye_y+5], fill='black')
    
    # Nose
    draw.line([size//2, size//2, size//2, 2*size//3], fill=(0, 0, 0), width=2)
    
    # Mouth
    draw.arc([size//3, 2*size//3-10, 2*size//3, 2*size//3+30], 
             0, 180, fill=(200, 0, 0), width=3)
    
    # Add noise for realism
    img_array = np.array(img)
    noise = np.random.normal(0, 10, img_array.shape).astype(np.int16)
    img_array = np.clip(img_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    return Image.fromarray(img_array)


def create_manipulated_face(real_img):
    """
    Apply manipulation artifacts to simulate deepfake.
    
    Args:
        real_img (PIL.Image): Original real image
    
    Returns:
        PIL.Image: Manipulated image with deepfake artifacts
    """
    img_array = np.array(real_img).astype(np.float32)
    
    # Random manipulation type
    manipulation = random.choice(['blend', 'compress', 'blur_edges', 'color_shift'])
    
    if manipulation == 'blend':
        # Blending artifacts
        mask = np.ones_like(img_array)
        mask[::2, ::2] *= 0.8
        img_array = img_array * mask
        
    elif manipulation == 'compress':
        # JPEG compression artifacts
        img = Image.fromarray(img_array.astype(np.uint8))
        temp_path = '/tmp/temp_compress.jpg'
        img.save(temp_path, quality=15)
        img = Image.open(temp_path)
        img_array = np.array(img).astype(np.float32)
        
    elif manipulation == 'blur_edges':
        # Boundary blur
        img = Image.fromarray(img_array.astype(np.uint8))
        img = img.filter(ImageFilter.GaussianBlur(radius=3))
        img_array = np.array(img).astype(np.float32)
        
    elif manipulation == 'color_shift':
        # Color channel inconsistency
        img_array[:, :, 0] *= 1.1  # Red channel
        img_array[:, :, 2] *= 0.9  # Blue channel
    
    img_array = np.clip(img_array, 0, 255).astype(np.uint8)
    return Image.fromarray(img_array)


def add_degradation(img, severity='high'):
    """
    Apply quality degradation to simulate underrepresented data.
    
    Args:
        img (PIL.Image): Input image
        severity (str): 'medium' or 'high'
    
    Returns:
        PIL.Image: Degraded image
    """
    img_array = np.array(img).astype(np.float32)
    
    if severity == 'high':
        noise = np.random.normal(0, 25, img_array.shape)
        blur_radius = 2.5
    else:
        noise = np.random.normal(0, 15, img_array.shape)
        blur_radius = 1.5
    
    img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(img_array)
    img = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    
    return img


class DemographicDeepfakeDataset(Dataset):
    """
    Custom dataset for deepfake detection with demographic annotations.
    
    Each sample includes:
    - image: Preprocessed face image tensor
    - label: 0 (real) or 1 (fake)
    - skin_tone: Categorical demographic attribute
    - gender: Categorical demographic attribute
    """
    
    def __init__(self, annotations_csv, transform=None):
        """
        Args:
            annotations_csv (str): Path to CSV with image paths and demographic labels
            transform (callable, optional): Image transformation pipeline
        """
        self.df = pd.read_csv(annotations_csv)
        self.transform = transform
        
        # Create mappings for categorical variables
        self.skin_tone_map = {tone: idx for idx, tone in enumerate(SKIN_TONES)}
        self.gender_map = {gender: idx for idx, gender in enumerate(GENDERS)}
        
        print(f"Loaded {len(self.df)} images")
        print(f"Real: {(self.df['label'] == 0).sum()}, "
              f"Fake: {(self.df['label'] == 1).sum()}")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Load and convert image
        image = Image.open(row['image_path']).convert('RGB')
        
        # Get labels
        label = row['label']
        skin_tone = self.skin_tone_map[row['skin_tone']]
        gender = self.gender_map[row['gender']]
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return {
            'image': image,
            'label': label,
            'skin_tone': skin_tone,
            'gender': gender,
            'skin_tone_text': row['skin_tone'],
            'gender_text': row['gender']
        }


def get_default_transform():
    """Get standard image preprocessing pipeline."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def generate_dataset(output_dir, n_train=100, n_test=30, seed=42):
    """
    Generate synthetic dataset with demographic annotations.
    
    Args:
        output_dir (str): Directory to save dataset
        n_train (int): Number of train pairs (real+fake)
        n_test (int): Number of test pairs
        seed (int): Random seed for reproducibility
    
    Returns:
        tuple: Paths to train and test annotation CSVs
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # Create directory structure
    train_real = os.path.join(output_dir, 'train/real')
    train_fake = os.path.join(output_dir, 'train/fake')
    test_real = os.path.join(output_dir, 'test/real')
    test_fake = os.path.join(output_dir, 'test/fake')
    
    for folder in [train_real, train_fake, test_real, test_fake]:
        os.makedirs(folder, exist_ok=True)
    
    # Generate training data
    print(f"Generating {n_train} training pairs...")
    for i in range(n_train):
        skin_tone = SKIN_TONES[i % len(SKIN_TONES)]
        real_img = create_realistic_face(skin_tone=skin_tone)
        real_img.save(f"{train_real}/real_{i:03d}.jpg")
        
        fake_img = create_manipulated_face(real_img)
        fake_img.save(f"{train_fake}/fake_{i:03d}.jpg")
    
    # Generate test data
    print(f"Generating {n_test} test pairs...")
    for i in range(n_test):
        skin_tone = SKIN_TONES[i % len(SKIN_TONES)]
        real_img = create_realistic_face(skin_tone=skin_tone)
        real_img.save(f"{test_real}/real_{i:03d}.jpg")
        
        fake_img = create_manipulated_face(real_img)
        fake_img.save(f"{test_fake}/fake_{i:03d}.jpg")
    
    # Create annotations
    train_csv = _create_annotations(train_real, train_fake, output_dir, 'train')
    test_csv = _create_annotations(test_real, test_fake, output_dir, 'test')
    
    print(f"Dataset generated successfully at {output_dir}")
    return train_csv, test_csv


def _create_annotations(real_dir, fake_dir, output_dir, split_name):
    """Create CSV annotations with demographic labels."""
    annotations = []
    
    for label_type in ['real', 'fake']:
        folder = real_dir if label_type == 'real' else fake_dir
        images = sorted(glob.glob(os.path.join(folder, '*.jpg')))
        
        for img_path in images:
            img_num = int(os.path.basename(img_path).split('_')[1].split('.')[0])
            skin_tone = SKIN_TONES[img_num % len(SKIN_TONES)]
            gender = GENDERS[img_num % len(GENDERS)]
            
            annotations.append({
                'image_path': img_path,
                'filename': os.path.basename(img_path),
                'label': 0 if label_type == 'real' else 1,
                'label_text': label_type,
                'skin_tone': skin_tone,
                'gender': gender
            })
    
    df = pd.DataFrame(annotations)
    csv_path = os.path.join(output_dir, f'{split_name}_annotations.csv')
    df.to_csv(csv_path, index=False)
    
    return csv_path


if __name__ == '__main__':
    # Example usage
    train_csv, test_csv = generate_dataset(
        output_dir='./data',
        n_train=100,
        n_test=30
    )
    print(f"Train annotations: {train_csv}")
    print(f"Test annotations: {test_csv}")

import random
import torch
from torchvision.transforms import v2

resize_x = 256
resize_y = 256

def add_noise_gaussian(tensor):
    """
    Parameters:
    - tensor: PyTorch tensor data type without noise (input)

    Returns:
    - tensor + noise: PyTorch tensor data type with noise (output)
    """

    constant = 0.2

    mean = tensor.mean() * constant
    std = tensor.std() * constant

    noise = torch.randn(tensor.size()) * std + mean
    return tensor + noise

def add_noise_salt_pepper(tensor, salt_prob = 0.01, pepper_prob = 0.01):
    """
    Parameters:
    - tensor: PyTorch tensor data type without noise (input)
    - salt_prob: Probability that salt noise is added (full white)
    - pepper_prob: Probability that pepper noise is added (full black)

    Returns:
    - tensor + salt_mask = pepper_mask: PyTorch tensor data type with noise (output)
    that ensured to be between 0 and 1
    """

    salt_mask = (torch.rand_like(tensor) < salt_prob).float() # * 0.35 * X_std
    pepper_mask = (torch.rand_like(tensor) < pepper_prob).float() # * 0.35 * X_std

    # return torch.clamp((tensor + salt_mask - pepper_mask), 0, 1)
    return torch.clamp((tensor + salt_mask - pepper_mask), 0, 1)

def set_resize_x_y(resize_x, resize_y):
    """
    Parameters:
    - resize_x: New x dimension
    - resize_y: New y dimension

    Returns:
    - None
    """

    if(resize_x < 0 or resize_y < 0):
        raise ValueError("resize_x and resize_y must be positive integers")
    
    if(resize_x != resize_y):
        raise ValueError("resize_x and resize_y must be equal")

    resize_x = resize_x
    resize_y = resize_y

transforms = {
    'toTensor': v2.Compose([
                        v2.ToImage(), 
                        v2.ToDtype(torch.float32, scale=True)
                        ]),
    'scale': v2.Compose([
                        v2.ToImage(), 
                        v2.ToDtype(torch.float32, scale=True),
                        v2.Resize((resize_x, resize_y), antialias=True)
                        ]),
    'noise_gaussian': v2.Compose([
                        v2.ToImage(), 
                        v2.ToDtype(torch.float32, scale=True),
                        v2.Lambda(lambda x: add_noise_gaussian(x))
                        ]),
    'noise_salt_pepper': v2.Compose([
                        v2.ToImage(), 
                        v2.ToDtype(torch.float32, scale=True),
                        v2.Lambda(lambda x: add_noise_salt_pepper(x))
                        ]),
    'all_gaussian':     v2.Compose([
                        v2.ToImage(), 
                        v2.ToDtype(torch.float32, scale=True),
                        v2.Resize((resize_x, resize_y), antialias=True),
                        v2.Lambda(lambda x: add_noise_gaussian(x))
                        ]),
    'all_salt_pepper':  v2.Compose([
                        v2.ToImage(), 
                        v2.ToDtype(torch.float32, scale=True),
                        v2.Resize((resize_x, resize_y), antialias=True),
                        v2.Lambda(lambda x: add_noise_salt_pepper(x))
                        ])
}
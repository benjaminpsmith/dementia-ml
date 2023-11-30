import random
import torch
from torchvision.transforms import v2

resize_x = 256
resize_y = 256

gl_mean = 0
gl_std = 1

noise_constant = 0

def add_noise_gaussian(tensor, mean=gl_mean*noise_constant , std=gl_std*noise_constant):
    """
    Parameters:
    - tensor: PyTorch tensor data type without noise (input)
    - mean: Mean of the Gaussian distribution
    - std: Standard deviation of the Gaussian distribution

    Returns:
    - tensor + noise: PyTorch tensor data type with noise (output)
    """
    std = std * (tensor.max() - tensor.min())
    mean = mean * (tensor.max() - tensor.min())
    
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

def set_noise_constant(constant):
    """
    Parameters:
    - constant: Noise constant

    Returns:
    - None
    """
    global noise_constant
    noise_constant = constant

def set_mean_std(mean, std):
    """
    Parameters:
    - mean: Mean of the Gaussian distribution
    - std: Standard deviation of the Gaussian distribution

    Returns:
    - None
    """
    global gl_mean
    global gl_std
    gl_mean = mean
    gl_std = std

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
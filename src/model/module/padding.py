import torch

def padding_with_method(x, padding_size, method='zero'):
    """
    Apply padding to a tensor using the specified method.
    
    Args:
    - x (torch.Tensor): The tensor to be padded.
    - padding_size (int): The target size after padding.
    - method (str): The padding method. Can be 'zero', 'gaussian', or 'mean_std'. Default is 'zero'.
    
    Returns:
    - torch.Tensor: The padded tensor.
    """
    b, _, d = x.shape
    if method == 'zero':
        padding_values = torch.zeros(b, padding_size, d)
    elif method == 'gaussian':
        mean = torch.mean(x, dim=(0, 1))
        std = torch.std(x, dim=(0, 1))
        padding_values = torch.normal(mean=mean, std=std, size=(b, padding_size, d))
    elif method == 'mean_std':
        mean = torch.mean(x, dim=(0, 1))
        std = torch.std(x, dim=(0, 1))
        padding_values = mean.unsqueeze(0).unsqueeze(1).expand(b, padding_size, d) + \
                         std.unsqueeze(0).unsqueeze(1).expand(b, padding_size, d) * \
                         torch.randn(b, padding_size, d).to(x.device)
    else:
        raise ValueError(f"Unsupported padding method: {method}. Supported methods are 'zero', 'gaussian', and 'mean_std'.")
    
    padded_tensor = torch.cat([x, padding_values], dim=1)
    return padded_tensor

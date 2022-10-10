from typing import List
import numpy as np
import cv2
import torch
import PIL.Image


def visualize_feature(x, size = 256):
    if isinstance(x, torch.Tensor):
        npx = x.detach().to('cpu').numpy()
        ans = visualize_feature(npx, size)
        return torch.from_numpy(ans).to(x.device)

    if len(x.shape) > 4 or len(x.shape) < 2:
        raise NotImplemented()
    elif len(x.shape) == 4:
        assert x.shape[1] == 1
        cams = []
        for i in range(x.shape[0]):
            ki = visualize_feature(x[i][0], size)
            cams.append(ki)
        return np.stack(cams, axis=0)
    elif len(x.shape) == 3:
        cams = []
        for i in range(x.shape[0]):
            ki = visualize_feature(x[i], size)
            cams.append(ki)
        return np.stack(cams, axis=0)

    x = x - np.min(x)
    cam_img = x / (np.max(x) + 1e-6)
    cam_img = np.uint8(255 * cam_img)
    cam_img = cv2.resize(cam_img, (size, size), interpolation=2)
    cam_img = cv2.applyColorMap(cam_img, cv2.COLORMAP_JET).transpose(2, 0, 1)
    return cam_img / 127.5 - 1


def image_blend_normal(img1: torch.Tensor, img2: torch.Tensor, alpha_a: float = 0.5):
    return img1 * alpha_a + img2 * (1 - alpha_a)


def image_grid(imgs: List[torch.Tensor], grid_width: int):
    assert len(imgs) > 0
    shape = imgs[0].shape
    assert len(shape) == 4

    device = imgs[0].device
    dtype = imgs[0].dtype
    _, C, H, W = shape
    for i in range(len(imgs)):
        img = imgs[i]
        assert img.shape == shape
        imgs[i] = img.to(device).detach()
    
    grid_width = min(grid_width, shape[0])
    if shape[0] % grid_width != 0:
        extra_shape = (grid_width - shape[0] % grid_width, C, H, W)
        for i in range(len(imgs)):
            img = imgs[i]
            img = torch.cat([img, torch.zeros(extra_shape, dtype=dtype, device=device)], dim=0)
            imgs[i] = img
    shape = imgs[0].shape
    grid_height = shape[0] // grid_width
        
    for i in range(len(imgs)):
        img = imgs[i]
        img = img.view(-1, grid_width, C, H, W)
        imgs[i] = img
    ans = torch.zeros([grid_height * len(imgs), grid_width, C, H, W], dtype=dtype, device=device)
    for h in range(grid_height):
        for i in range(len(imgs)):
            img = imgs[i]
            ans[h * len(imgs) + i] = img[h]
    ans = ans.permute(2, 0, 3, 1, 4).contiguous()
    ans = ans.view(C, H * grid_height * len(imgs), W * grid_width)
    return ans


def save_image(img: torch.Tensor, path: str):
    img = (img.permute(1, 2, 0) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    PIL.Image.fromarray(img.cpu().numpy(), 'RGB').save(path, quality=100, subsampling=0)
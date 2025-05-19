from typing import Any, Tuple, TypeVar, Union
import matplotlib.pyplot as plt

import torch
import numpy as np
from pytorch_lightning import Callback

from multi_object_datasets_torch import adjusted_rand_index

import wandb

Tensor = TypeVar("torch.tensor")
T = TypeVar("T")
TK = TypeVar("TK")
TV = TypeVar("TV")


def conv_transpose_out_shape(in_size, stride, padding, kernel_size, out_padding, dilation=1):
    return (in_size - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + out_padding + 1


def assert_shape(actual: Union[torch.Size, Tuple[int, ...]], expected: Tuple[int, ...], message: str = ""):
    assert actual == expected, f"Expected shape: {expected} but passed shape: {actual}. {message}"


def build_grid(resolution):
    ranges = [torch.linspace(0.0, 1.0, steps=res) for res in resolution]
    grid = torch.meshgrid(*ranges)
    grid = torch.stack(grid, dim=-1)
    grid = torch.reshape(grid, [resolution[0], resolution[1], -1])
    grid = grid.unsqueeze(0)
    return torch.cat([grid, 1.0 - grid], dim=-1)


def rescale(x: Tensor) -> Tensor:
    return x * 2 - 1


def compact(l: Any) -> Any:
    return list(filter(None, l))


def first(x):
    return next(iter(x))


def only(x):
    materialized_x = list(x)
    assert len(materialized_x) == 1
    return materialized_x[0]


class ImageLogCallback(Callback):
    def on_validation_epoch_end(self, trainer, pl_module):
        """Called when the train epoch ends."""

        if trainer.logger:
            with torch.no_grad():
                pl_module.eval()
                images = pl_module.sample_images()
                trainer.logger.experiment.log({"images": [wandb.Image(images)]}, commit=False)


def to_rgb_from_tensor(x: Tensor):
    return (x * 0.5 + 0.5).clamp(0, 1)


def visualize_masks(image, true_masks, pred_masks):
    """
    Visualize original image, ground truth masks, and predicted masks
    
    Args:
        image: Tensor of shape [3, H, W]
        true_masks: Tensor of shape [N, H, W]
        pred_masks: Tensor of shape [K, 1, H, W]
    """
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axs[0].imshow(image.permute(1, 2, 0).cpu().numpy())
    axs[0].set_title('Original Image')
    axs[0].axis('off')
    
    # Ground truth masks (colored by instance)
    true_mask_vis = np.zeros((*true_masks.shape[1:], 3))
    for i in range(true_masks.shape[0]):
        # Skip background (usually index 0)
        if i == 0:
            continue
        color = np.random.rand(3)
        mask = true_masks[i].cpu().numpy()
        for c in range(3):
            true_mask_vis[..., c] += mask * color[c]
    
    axs[1].imshow(true_mask_vis)
    axs[1].set_title('Ground Truth Masks')
    axs[1].axis('off')
    
    # Predicted masks (colored by slot)
    pred_mask_vis = np.zeros((*pred_masks.shape[2:], 3))
    for i in range(pred_masks.shape[0]):
        color = np.random.rand(3)
        mask = pred_masks[i, 0].cpu().numpy()
        for c in range(3):
            pred_mask_vis[..., c] += mask * color[c]
    
    axs[2].imshow(pred_mask_vis)
    axs[2].set_title('Predicted Masks')
    axs[2].axis('off')
    
    plt.tight_layout()
    return fig


def compute_ari(true_masks, pred_masks):
    """
    Compute Adjusted Rand Index between true masks and predicted masks
    
    Args:
        true_masks: Tensor of shape [batch_size, n_true_groups, height, width]
        pred_masks: Tensor of shape [batch_size, n_pred_groups, 1, height, width]
    
    Returns:
        Tensor of shape [batch_size] containing ARI scores
    """
    batch_size, n_groups, _, height, width = true_masks.shape
    batch_size, _, _, _, _ = pred_masks.shape

    # Reshape masks to [batch_size, n_points, n_groups]
    true_masks_flat = true_masks.squeeze(2).permute(0, 2, 3, 1).reshape(batch_size, height * width, n_groups)
    pred_masks_flat = pred_masks.squeeze(2).permute(0, 2, 3, 1).reshape(batch_size, height * width, n_groups)
    
    # Compute ARI
    ari_scores = adjusted_rand_index(true_masks_flat, pred_masks_flat)
    
    return ari_scores
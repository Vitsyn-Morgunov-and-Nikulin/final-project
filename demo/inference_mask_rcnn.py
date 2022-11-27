from collections.abc import Callable
from pathlib import Path

import click
import numpy as np
import skimage.io as io
import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn
import albumentations as A
from albumentations.pytorch import ToTensorV2
from skimage.color import gray2rgb

import numpy as np
import skimage.io as io
import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn
import albumentations as A
from albumentations.pytorch import ToTensorV2
from skimage.color import gray2rgb
from torchvision.ops import nms
from torchvision.utils import draw_segmentation_masks



def load_image(image_path: str):
    image = io.imread(str(image_path))
    return image


def get_masks_from_image(image: torch.Tensor, model: torch.nn.Module) -> torch.Tensor:
    score_threshold = 0.0  # All predictions would be counted, even with low score
    nms_threshold = 0.2  # Overlapping instances will be dropped, lower - lower overlap is permitted
    mask_threshold = 0.5  # Cut masks by the threshold
    
    with torch.no_grad():
        output = model.forward([image])[0]

    scores = output['scores'].detach().cpu()
    masks = output['masks'].squeeze().detach().cpu()
    boxes = output['boxes'].detach().cpu()

    masks = (masks >= mask_threshold).int()

    indices = torch.as_tensor([torch.sum(mask) > 0 for mask in masks])
    masks, boxes, scores = masks[indices], boxes[indices], scores[indices]

    indices = scores >= score_threshold
    masks, boxes, scores = masks[indices], boxes[indices], scores

    indices = nms(boxes, scores, nms_threshold)
    masks, boxes, scores = masks[indices], boxes[indices], scores[indices]

    return masks


def to_rgb_image(im: torch.Tensor):
    return np.transpose(im.numpy(), axes=(1, 2, 0))


def draw_segmentation_mask(image, masks):
    gt_image = gray2rgb(image.squeeze())
    gt_image = torch.as_tensor(np.transpose(gt_image, axes=(2, 0, 1)), dtype=torch.uint8)
    gt_masks = torch.as_tensor(masks == 1)

    segm = draw_segmentation_masks(
        image=gt_image,
        masks=gt_masks,
        alpha=0.5
    )
    return to_rgb_image(segm)


def inference_mask_rcnn(weights_path: str, image_path: str) -> np.ndarray:
    """
    Parameters:
    -----------
    weights_path: str - path to the .ckpt file with Mask R-CNN trained weights
    image_path: str - path to the image, for which one needs inference

    Returns:
    -----------
    np.ndarray of shape [IMAGE_HEIGHT, IMAGE_WIDTH, 3]
    """
    weights_dir = Path(weights_path)

    preprocess_image = A.Compose([
        A.Normalize(mean=(0.485,), std=(0.229,)),
        ToTensorV2(),
    ])
    
    model = maskrcnn_resnet50_fpn(progress=False, num_classes=4)
    model.load_state_dict(torch.load(weights_dir, map_location=torch.device("cpu")))
    model = model.eval()

    image = load_image(image_path)
    torchified_image = preprocess_image(image=image)['image']
    predicted_masks = get_masks_from_image(torchified_image, model)
    predicted_image = draw_segmentation_mask(image, predicted_masks)
    return predicted_image



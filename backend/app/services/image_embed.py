"""
Image embedding service.

Tries to use PyTorch + torchvision (ResNet50) to compute image embeddings.
If torch is not installed, raises ImportError so caller can return a helpful message.

The embedding is the ResNet50 avgpool feature (2048-D), L2-normalized.
This is a small, reliable image feature to use for image similarity search.
"""

from typing import List
from io import BytesIO

# PIL is already in requirements; used for image open
from PIL import Image
import numpy as np

# Lazy imports for torch/torchvision
_torch_available = False
_torch = None
_torchvision = None
_resnet = None
_transforms = None

def _ensure_torch():
    global _torch_available, _torch, _resnet, _transforms, _torchvision
    if _torch_available:
        return
    try:
        import torch
        import torchvision
        from torchvision import transforms, models
        _torch = torch
        _torchvision = torchvision
        _transforms = transforms
        # load pretrained resnet50 and remove final fc
        model = models.resnet50(pretrained=True)
        # remove classifier: take everything up to avgpool
        model.eval()
        # remove gradients for speed
        for p in model.parameters():
            p.requires_grad = False
        # keep model in eval mode
        _resnet = model
        _torch_available = True
    except Exception as e:
        _torch_available = False
        raise

def embed_image_bytes(img_bytes: bytes) -> List[float]:
    """
    Compute embedding from raw image bytes.
    Returns L2-normalized vector (list of floats).
    Raises ImportError if torch/torchvision not installed.
    """
    # ensure torch is available, otherwise raise
    try:
        _ensure_torch()
    except Exception as e:
        # re-raise with a clearer message
        raise ImportError("PyTorch / torchvision not available. Install torch to use image search.") from e

    # open image with PIL
    im = Image.open(BytesIO(img_bytes)).convert("RGB")

    # build transform consistent with ResNet pretrained
    transform = _transforms.Compose([
        _transforms.Resize(256),
        _transforms.CenterCrop(224),
        _transforms.ToTensor(),                 # converts to [0,1]
        _transforms.Normalize(                  # imagenet normalize
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    inp = transform(im).unsqueeze(0)  # shape (1, 3, 224, 224)
    # run through model up to avgpool
    with _torch.no_grad():
        x = inp
        # ResNet forward but capture layer outputs till avgpool
        # Equivalent to: features = model.conv1..layer4..avgpool
        # We can call _resnet.forward and pull out the penultimate layer by creating a small forward hook-less pass:
        # Simpler: run full forward then take model.fc removed -- use model until avgpool:
        # The easiest safe approach: replicate parts:
        # Use torchvision.models._utils.IntermediateLayerGetter would be ideal but keep it direct:
        # We'll run forward and stop before fc by popping fc and flattening avgpool output.
        # Use the model's children:
        modules = list(_resnet.children())[:-1]  # everything except the final fc
        feat_extractor = _torch.nn.Sequential(*modules)
        feats = feat_extractor(x)               # shape (1, 2048, 1, 1)
        feats = feats.reshape(feats.shape[0], -1)  # (1, 2048)
        vec = feats.cpu().numpy().reshape(-1).astype(float)
    # normalize
    norm = np.linalg.norm(vec)
    if norm == 0:
        norm = 1.0
    vec = (vec / norm).tolist()
    return vec

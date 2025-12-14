"""
En este script definimos los extractores de características para CBIR.
Cada extractor recibe una imagen (PIL) y devuelve un vector NumPy float32 con forma (1, D).
"""

import numpy as np
import matplotlib.colors as mcolors

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T

device = torch.device("cpu")


# 1) RGB histogram

def extractor_rgb_hist(pil_img, bins_per_channel=16):

    """
    Descriptor global de color en RGB:
    - Redimensiona a 256x256
    - Calcula histogramas por canal (R,G,B) con 'bins_per_channel' bins
    - Concatena los 3 histogramas->vector (1, 3*bins)
    """

    img = pil_img.resize((256, 256)).convert("RGB")
    arr = np.array(img)

    hr, _ = np.histogram(arr[:, :, 0], bins=bins_per_channel, range=(0, 256), density=True)
    hg, _ = np.histogram(arr[:, :, 1], bins=bins_per_channel, range=(0, 256), density=True)
    hb, _ = np.histogram(arr[:, :, 2], bins=bins_per_channel, range=(0, 256), density=True)

    feat = np.concatenate([hr, hg, hb]).astype("float32")
    return feat.reshape(1, -1)


# 2) HSV histogram

def extractor_hsv_hist(pil_img, bins_per_channel=16):

    """
    Descriptor global de color en HSV:
    - Redimensiona a 256x256 y pasa a RGB
    - Escala a [0,1] y convierte a HSV
    - Histogramas en H, S y V (16 bins por defecto)
    - Concatena -> vector (1, 3*bins)
    """

    img = pil_img.resize((256, 256)).convert("RGB")
    arr = np.array(img) / 255.0
    hsv = mcolors.rgb_to_hsv(arr)

    hh, _ = np.histogram(hsv[:, :, 0], bins=bins_per_channel, range=(0, 1), density=True)
    hs, _ = np.histogram(hsv[:, :, 1], bins=bins_per_channel, range=(0, 1), density=True)
    hv, _ = np.histogram(hsv[:, :, 2], bins=bins_per_channel, range=(0, 1), density=True)

    feat = np.concatenate([hh, hs, hv]).astype("float32")
    return feat.reshape(1, -1)


# 3) ResNet18 (CNN)

# Cargamos ResNet18 preentrenada en ImageNet
_resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

_resnet = nn.Sequential(*list(_resnet.children())[:-1])  # Quitamos la capa final de clasificación para quedarnos con el embedding
_resnet.eval().to(device)

# Transformaciones estándar para CNNs preentrenadas en ImageNet
_transform_resnet = T.Compose(
    [
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

@torch.no_grad()
def extractor_resnet18(pil_img):

    """
    Extrae un embedding (1, 512) usando ResNet18 sin la capa final.
    """

    img = pil_img.convert("RGB")
    x = _transform_resnet(img).unsqueeze(0).to(device)
    feat = _resnet(x).view(1, -1).cpu().numpy().astype("float32")  # (1,512)
    return feat


# 4) EfficientNet-B0 (CNN)

_efficientnet = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
_efficientnet.classifier = nn.Identity() # Quitamos la parte de clasificación para obtener directamente el embedding
_efficientnet.eval().to(device)

_transform_eff = T.Compose(
    [
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

@torch.no_grad()
def extractor_efficientnet_b0(pil_img):

    """
    Extrae un embedding profundo usando EfficientNet-B0 (sin clasificador).
    """

    img = pil_img.convert("RGB")
    x = _transform_eff(img).unsqueeze(0).to(device)
    feat = _efficientnet(x).view(1, -1).cpu().numpy().astype("float32")
    return feat


# 5) VGG19 (CNN) -> 512 dims

_vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
_vgg_features = _vgg.features
_vgg_features.eval().to(device)

# Pooling para convertir (1,512,7,7)->(1,512,1,1) y dejar vector fijo (1,512)
_vgg_pool = nn.AdaptiveAvgPool2d((1, 1)).eval().to(device)

_transform_vgg = T.Compose(
    [
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

@torch.no_grad()
def extractor_vgg19(pil_img):

    """
    Extrae un embedding (1, 512) usando VGG19:
    - Solo bloque convolucional (features)
    - Pooling adaptativo a (1,1)
    """

    img = pil_img.convert("RGB")
    x = _transform_vgg(img).unsqueeze(0).to(device)        # (1,3,224,224)
    f = _vgg_features(x)                                   # (1,512,7,7)
    f = _vgg_pool(f).view(1, -1)                           # (1,512)
    return f.cpu().numpy().astype("float32")


# 6) SIFT 

# Requiere: pip install opencv-contrib-python
# SIFT devuelve un nº variable de descriptores, aquí lo convertimos a vector fijo (256)

try:
    import cv2

    _sift = cv2.SIFT_create()

    def extractor_sift(pil_img, max_size=256):

        """
        Extrae características con SIFT y las vectoriza a tamaño fijo:
        - Convierte a escala de grises
        - Reescala si la imagen es grande 
        - Calcula SIFT (desc: Nx128)
        - Devuelve concat(mean, std) -> (256,)
        - Si no hay keypoints, devuelve vector de ceros
        """

        img = pil_img.convert("L")
        arr = np.array(img)

        h, w = arr.shape[:2]
        scale = min(max_size / max(h, w), 1.0)
        if scale < 1.0:
            arr = cv2.resize(arr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

        _, desc = _sift.detectAndCompute(arr, None)

        if desc is None or len(desc) == 0:
            feat = np.zeros((256,), dtype=np.float32)
            return feat.reshape(1, -1)

        mean = desc.mean(axis=0)
        std = desc.std(axis=0)

        feat = np.concatenate([mean, std]).astype(np.float32)  # (256,)
        return feat.reshape(1, -1)

except Exception:
    extractor_sift = None

# Diccionario final de extractores 
EXTRACTORS = {
    "rgb_hist": extractor_rgb_hist,
    "hsv_hist": extractor_hsv_hist,
    "resnet18": extractor_resnet18,
    "efficientnet_b0": extractor_efficientnet_b0,
    "vgg19": extractor_vgg19,
    "sift": extractor_sift
}


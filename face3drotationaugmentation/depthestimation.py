import torch
import cv2

_midas = None
_transform = None
_device = None

initialized = False


def init():
    global _midas
    global _transform
    global _device
    global initialized
    #model_type = "DPT_Large"     # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
    model_type = "DPT_Hybrid"   # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
    #model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)

    # This repo looks legit. I put in the current commit hash. So not much could go wrong here.
    repository = "intel-isl/MiDaS:1645b7e1675301fdfac03640738fe5a6531e17d6"

    _midas = torch.hub.load(repository, model_type, trust_repo=True)

    _device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    _midas.to(_device)
    _midas.eval()

    midas_transforms = torch.hub.load(repository, "transforms", trust_repo=True)
    if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
        _transform = midas_transforms.dpt_transform
    else:
        _transform = midas_transforms.small_transform
    initialized = True


def inference(img):
    input_batch = _transform(img).to(_device)

    with torch.no_grad():
        prediction = _midas(input_batch)

        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    output = prediction.cpu().numpy()
    return output
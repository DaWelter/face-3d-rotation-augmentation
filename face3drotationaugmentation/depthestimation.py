import torch
from typing import Any, cast

_midas : torch.nn.Module | None = None
_transform : Any | None = None
_device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

initialized = False


def init():
    global _midas
    global _transform
    global initialized

    # Using https://pytorch.org/hub/intelisl_midas_v2/

    #model_type = "DPT_Large"     # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
    model_type = "DPT_Hybrid"   # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
    #model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)

    # This repo looks legit. I put in the current commit hash. So not much could go wrong here.
    repository = "intel-isl/MiDaS:1645b7e1675301fdfac03640738fe5a6531e17d6"

    _midas = cast(torch.nn.Module, torch.hub.load(repository, model_type, trust_repo=True))
    assert isinstance(_midas, torch.nn.Module)

    _midas.to(_device)
    _midas.eval()

    # https://github.com/isl-org/MiDaS/blob/master/hubconf.py#L303
    midas_transforms : Any = torch.hub.load(repository, "transforms", trust_repo=True)
    if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
        _transform = midas_transforms.dpt_transform
    else:
        _transform = midas_transforms.small_transform
    initialized = True


def inference(img):
    if _transform is None or _midas is None:
        msg = "MiDaS model is not initialized. Call `init()` first."
        raise RuntimeError(msg)

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
import torch
import cv2

midas = None
transform = None
device = None


def init():
    global midas
    global transform
    global device
    #model_type = "DPT_Large"     # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
    model_type = "DPT_Hybrid"   # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
    #model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)

    # This repo looks legit. I put in the current commit hash. So not much could go wrong here.
    repository = "intel-isl/MiDaS:1645b7e1675301fdfac03640738fe5a6531e17d6"

    midas = torch.hub.load(repository, model_type, trust_repo=True)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    midas.to(device)
    midas.eval()

    midas_transforms = torch.hub.load(repository, "transforms", trust_repo=True)
    if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
        transform = midas_transforms.dpt_transform
    else:
        transform = midas_transforms.small_transform


def inference(img):
    input_batch = transform(img).to(device)

    with torch.no_grad():
        prediction = midas(input_batch)

        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    output = prediction.cpu().numpy()
    return output
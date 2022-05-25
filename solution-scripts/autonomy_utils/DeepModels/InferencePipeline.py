from pathlib import Path
import time
from autonomy_utils.Logger import Logger
from autonomy_utils.ambf_utils import ImageSaver
import cv2
import numpy as np
from autonomy_utils.DeepModels.Model import UNet
from autonomy_utils.DeepModels.Dice import DiceLoss, DiceScore, DiceBCELoss
import torch
import torch.nn as nn
from torchvision import transforms
from torch.autograd import Variable
import torch.nn.functional as functional
from torch.utils.data import Dataset, DataLoader, random_split
from skimage import io, transform
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2 as ToTensor

import rospy


log = Logger("autonomy_utils").log


def format_image(img):
    img = np.array(np.transpose(img, (1, 2, 0)))
    mean = np.array((0.485, 0.456, 0.406))
    std = np.array((0.229, 0.224, 0.225))
    img = std * img + mean
    img = img * 255
    img = img.astype(np.uint8)
    return img


def test_transform():
    return A.Compose(
        [A.Resize(1080, 1920), A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), ToTensor()]
    )


class InferencePipe:
    def __init__(self, model_path: Path, device: str = "cuda"):

        self.device = torch.device(device)
        self.model = UNet(1)

        # Load weigths
        best_model_cp = torch.load(model_path, map_location=device)
        self.model.load_state_dict(best_model_cp["model_state_dict"])

        if self.device.type == "cuda":
            log.info("Running model on GPU")
            self.model = self.model.cuda()

        self.convert_tensor = test_transform()

    def segmented_image(self, img: np.ndarray) -> np.ndarray:

        augmented = self.convert_tensor(image=img)
        img = augmented["image"]

        with torch.no_grad():
            if self.device.type == "cuda":
                input = torch.autograd.Variable(img.unsqueeze(0)).cuda()
            else:
                input = torch.autograd.Variable(img.unsqueeze(0))

            o = self.model(input)

        tm = o[0][0].data.cpu().numpy()
        predict_test_t = ((tm > 0.5) * 255).astype(np.uint8)
        img = input[0].data.cpu()
        img = format_image(img)

        return predict_test_t


if __name__ == "__main__":

    # Read image from test folder
    # img_pt = Path("./Media/test_img/test_image01.jpeg")
    # if not img_pt.exists():
    #     log.error("test image not found")
    #     exit(0)
    # test_img = cv2.imread(str(img_pt))

    # Read image from topics
    rospy.init_node("main_node")
    img_saver = ImageSaver()
    time.sleep(0.03)
    test_img = img_saver.get_current_frame("left")

    model_path = Path("./Resources/segmentation_weights/best_model_512.pth")
    if not model_path.exists():
        log.error("model weights not found")
        exit(0)

    test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB).astype("float32")

    tic = time.time()
    inference_model = InferencePipe(model_path, device="cuda")
    toc = time.time()
    log.info(f"Time loading segmentation network {toc-tic}")
    tic = time.time()
    segmented_img = inference_model.segmented_image(test_img)
    toc = time.time()
    log.info(f"Inference time {toc-tic}")

    segmented_img = cv2.cvtColor(segmented_img, cv2.COLOR_GRAY2BGR)

    # Combine left and right into a single frame to display
    test_img = cv2.cvtColor(test_img, cv2.COLOR_RGB2BGR).astype("uint8")
    test_img = cv2.resize(test_img, (640, 480), interpolation=cv2.INTER_AREA)
    segmented_img = cv2.resize(segmented_img, (640, 480), interpolation=cv2.INTER_AREA)
    final = np.hstack((test_img, segmented_img))
    cv2.imshow("final", final)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # cv2.imshow("final", segmented_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

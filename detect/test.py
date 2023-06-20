import os

import cv2
import torch.cuda

if torch.cuda.is_available():
    print(torch.cuda.device_count())
    print(torch.cuda.get_device_name())


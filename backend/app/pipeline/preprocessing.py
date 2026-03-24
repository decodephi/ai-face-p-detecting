from dataclasses import dataclass

import cv2
import numpy as np

from app.core.config import FIXED_IMAGE_SIZE


@dataclass
class PreprocessedImage:
    original_bgr: np.ndarray
    resized_bgr: np.ndarray
    normalized_rgb: np.ndarray
    hsv: np.ndarray
    lab: np.ndarray


def preprocess_image(image_bgr: np.ndarray) -> PreprocessedImage:
    resized = cv2.resize(image_bgr, FIXED_IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    denoised = cv2.fastNlMeansDenoisingColored(resized, None, 5, 5, 7, 21)
    lab_image = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab_image)
    l_channel = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8)).apply(l_channel)
    enhanced_bgr = cv2.cvtColor(cv2.merge((l_channel, a_channel, b_channel)), cv2.COLOR_LAB2BGR)
    sharpened = cv2.addWeighted(enhanced_bgr, 1.18, cv2.GaussianBlur(enhanced_bgr, (0, 0), 2.0), -0.18, 0)
    rgb = cv2.cvtColor(sharpened, cv2.COLOR_BGR2RGB)
    normalized_rgb = rgb.astype(np.float32) / 255.0
    hsv = cv2.cvtColor(sharpened, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(sharpened, cv2.COLOR_BGR2LAB)
    return PreprocessedImage(
        original_bgr=image_bgr,
        resized_bgr=sharpened,
        normalized_rgb=normalized_rgb,
        hsv=hsv,
        lab=lab,
    )

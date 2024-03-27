import cv2
import numpy as np
import torch
from einops import rearrange


def preprocess_live(cap, num_frames_to_process=64):
    """
    Capture and preprocess video frames from the live feed.
    """
    frames = []

    for _ in range(num_frames_to_process):
        ret, img = cap.read()
        if not ret:
            return None  # If a frame can't be captured, return None

        img = cv2.resize(img, (224, 224))  # Resize to match model input
        img = (img / 255.) * 2 - 1  # Normalize
        frames.append(img)

    frames = np.asarray(frames, dtype=np.float32)
    frames = torch.tensor(frames)  # Convert to tensor
    frames = rearrange(frames, 't h w c -> 1 c t h w')  # Rearrange to (1, C, T, H, W)
    return frames
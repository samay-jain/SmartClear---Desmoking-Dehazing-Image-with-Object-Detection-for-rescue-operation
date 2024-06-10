from skimage.metrics import structural_similarity as ssim
from PIL import Image
import numpy as np

#Dehazed Image output accuracy calculator
def calculate_accuracy(dehazed_path, ground_truth_path):
    dehazed_img = Image.open(dehazed_path).convert('RGB')
    ground_truth_img = Image.open(ground_truth_path).convert('RGB')

    dehazed_np = np.array(dehazed_img)
    ground_truth_np = np.array(ground_truth_img)
    window_size = 3
    accuracy = ssim(ground_truth_np, dehazed_np, win_size=window_size, multichannel=True)
    return accuracy
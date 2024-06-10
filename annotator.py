import torch
import cv2
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms import functional as FF
from torchvision.transforms.functional import to_pil_image


#converting cv2 image to pil image
def cv2_to_pil(cv2_img):
    if len(cv2_img.shape) == 2:
        return Image.fromarray(cv2_img)
    elif len(cv2_img.shape) == 3:
        return Image.fromarray(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB))
    else:
        raise ValueError("Unsupported image format")
    

def AnnotatorAndGridMaker(original, dehazed, detected):
    smoked_label = " Hazed Image - "
    dehazed_label = " De-Hazed Image - "
    detection = " Living Being Detection - "

    dehazed = to_pil_image(dehazed)
    detected = cv2_to_pil(detected)

    draw = ImageDraw.Draw(original)
    font_size = 20 
    font = ImageFont.truetype("comicbd.ttf", font_size)
    draw.text((5, 5), smoked_label, fill="darkgreen", font=font)

    draw = ImageDraw.Draw(dehazed)
    draw.text((5, 5), dehazed_label, fill="lightgreen", font=font)

    draw = ImageDraw.Draw(detected)
    draw.text((5, 5), detection, fill="lightgreen", font=font)

    original = FF.to_tensor(original)
    dehazed = FF.to_tensor(dehazed)
    detected = FF.to_tensor(detected)

    image_grid = torch.cat((original, dehazed, detected), -1)
    return image_grid
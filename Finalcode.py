import os
import torch
import torch.nn.functional as F
import torchvision.transforms as tfs
import torchvision.utils as vutils
import warnings
import numpy as np
import cv2
import matplotlib.pyplot as plt

from torch import nn
from torchvision.utils import make_grid
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms import functional as FF
from torchvision.transforms.functional import to_tensor
from torchvision.transforms.functional import to_pil_image
from skimage.metrics import structural_similarity as ssim
from ultralytics import YOLO
warnings.filterwarnings('ignore')

import numpy as np
from guided_filter import HazeRemoval
import os

#Initialising GPU
if torch.cuda.is_available():
    device = 'cuda'
    print(f"CUDA version: {torch.version.cuda}")
    cuda_id = torch.cuda.current_device()
    print(f"ID of current CUDA device: {torch.cuda.current_device()}")
    print(f"Name of current CUDA device: {torch.cuda.get_device_name(cuda_id)}", "\n")
else:
    device = 'cpu'
    print("CUDA is not available. Using CPU.")


#Num residual_groups
gps = 3
#Num residual_blocks
#blocks = 19
blocks = 19

#Initialising input,output and model directories
img_dir = 'D:/Major Project/De-Smoking or De-Hazing Module/Input/Input_Images/'
#img_dir = 'C:/Users/Admin/Desktop/imgezzzzz/imgezzzzz/'
pretrained_model_dir = 'D:/Major Project/De-Smoking or De-Hazing Module/weights/' + f'model_{gps}_{blocks}_20000.pk'
output_dir = 'D:/Major Project/De-Smoking or De-Hazing Module/Output/'

if not os.path.exists(output_dir):
    os.mkdir(output_dir)


#FFA-Net dehazing/desmoking
def tensorShow(tensors,titles=None):
    '''t:BCWH'''
    fig=plt.figure()
    for tensor, title, i in zip(tensors, titles, range(len(tensors))):
        img = make_grid(tensor)
        npimg = img.numpy()
        ax = fig.add_subplot(211+i)
        ax.imshow(np.transpose(npimg, (1, 2, 0)))
        ax.set_title(title)
    plt.show()


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size//2), bias=bias)
    

class PALayer(nn.Module):
    def __init__(self, channel):
        super(PALayer, self).__init__()
        self.pa = nn.Sequential(
                nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 8, 1, 1, padding=0, bias=True),
                nn.Sigmoid()
        )
    def forward(self, x):
        y = self.pa(x)
        return x * y


class CALayer(nn.Module):
    def __init__(self, channel):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
                nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 8, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.ca(y)
        return x * y

    
class Block(nn.Module):
    def __init__(self, conv, dim, kernel_size,):
        super(Block, self).__init__()
        self.conv1 = conv(dim, dim, kernel_size, bias=True)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = conv(dim, dim, kernel_size, bias=True)
        self.calayer = CALayer(dim)
        self.palayer = PALayer(dim)

    def forward(self, x):
        res = self.act1(self.conv1(x))
        res = res+x 
        res = self.conv2(res)
        res = self.calayer(res)
        res = self.palayer(res)
        res += x 
        return res

    
class Group(nn.Module):
    def __init__(self, conv, dim, kernel_size, blocks):
        super(Group, self).__init__()
        modules = [Block(conv, dim, kernel_size)  for _ in range(blocks)]
        modules.append(conv(dim, dim, kernel_size))
        self.gp = nn.Sequential(*modules)

    def forward(self, x):
        res = self.gp(x)
        res += x
        return res

    
class FFA(nn.Module):
    def __init__(self,gps,blocks,conv=default_conv):
        super(FFA, self).__init__()
        self.gps = gps
        self.dim = 64
        kernel_size = 3
        pre_process = [conv(3, self.dim, kernel_size)]
        assert self.gps==3
        self.g1 = Group(conv, self.dim, kernel_size,blocks=blocks)
        self.g2 = Group(conv, self.dim, kernel_size,blocks=blocks)
        self.g3 = Group(conv, self.dim, kernel_size,blocks=blocks)
        self.ca = nn.Sequential(*[
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.dim*self.gps,self.dim//16,1,padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.dim//16, self.dim*self.gps, 1, padding=0, bias=True),
            nn.Sigmoid()
            ])
        self.palayer = PALayer(self.dim)

        post_process = [
            conv(self.dim, self.dim, kernel_size),
            conv(self.dim, 3, kernel_size)]

        self.pre = nn.Sequential(*pre_process)
        self.post = nn.Sequential(*post_process)


    def forward(self, x1):
        x = self.pre(x1)
        res1 = self.g1(x)
        res2 = self.g2(res1)
        res3 = self.g3(res2)
        w = self.ca(torch.cat([res1,res2,res3],dim=1))
        w = w.view(-1,self.gps, self.dim)[:,:,:,None,None]
        out = w[:,0,::] * res1 + w[:,1,::] * res2+w[:,2,::] * res3
        out = self.palayer(out)
        x = self.post(out)
        return x + x1


#Human Detection initialisation part
class_list = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
              'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
              'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
              'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
              'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
              'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
              'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
              'teddy bear', 'hair drier', 'toothbrush']

#Generate random colors for class list
#detection_colors = [(np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)) for _ in range(len(class_list))]
#Load a YOLOv8n model
model = YOLO("weights/yolov8-med-25.9params.pt","v8")
#model = YOLO("D:/Major Project/De-Smoking or De-Hazing Module/weights/yolov8nmod.pt","v8")


#Dehazed Image output accuracy calculator
def calculate_accuracy(dehazed_path, ground_truth_path):
    dehazed_img = Image.open(dehazed_path).convert('RGB')
    ground_truth_img = Image.open(ground_truth_path).convert('RGB')

    dehazed_np = np.array(dehazed_img)
    ground_truth_np = np.array(ground_truth_img)
    window_size = 3
    accuracy = ssim(ground_truth_np, dehazed_np, win_size=window_size, multichannel=True)
    return accuracy


#tensor image to cv2 image
def tensor_to_cv2(tensor_image):
    numpy_image = tensor_image.permute(1, 2, 0).numpy()
    numpy_image = (numpy_image * 255).astype(np.uint8)
    cv2_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
    return cv2_image


#cv2 image to tensor image
def cv2_to_tensor(cv2_image):
    rgb_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
    tensor_image = to_tensor(rgb_image)
    return tensor_image


#cv2 image to PIL image
def cv2_to_pil(cv2_img):
    if len(cv2_img.shape) == 2:
        return Image.fromarray(cv2_img)
    elif len(cv2_img.shape) == 3:
        return Image.fromarray(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB))
    else:
        raise ValueError("Unsupported image format")


def dehazingImg(haze):
    haze1 = tfs.Compose([
        tfs.ToTensor(),
        tfs.Normalize(mean=[0.64, 0.6, 0.58], std=[0.14, 0.15, 0.152])
    ])(haze)[None, ::]
    haze_no = tfs.ToTensor()(haze)[None, ::]
    with torch.no_grad():
        pred = net(haze1)
    ts = torch.squeeze(pred.clamp(0, 1).cpu())

    haze_no = make_grid(haze_no, nrow=1, normalize=True)
    ts = make_grid(ts, nrow=1, normalize=True)
    return ts


def livingDetection(img):
    frame = tensor_to_cv2(img)
    if frame is None:
        print("Error: Could not open or read the image.")
        exit()

    detect_params = model.predict(source=[frame], conf=0.3, save=False)  # Adjust confidence threshold
    DP = detect_params[0].cpu().numpy()

    if len(DP) != 0:
        # Iterate over detected objects
        for i in range(len(detect_params[0])):
            boxes = detect_params[0].boxes
            box = boxes[i]
            clsID = box.cls.cpu().numpy()[0]
            conf = box.conf.cpu().numpy()[0]
            bb = box.xyxy.cpu().numpy()[0]

            # Check if the detected object belongs to living beings
            class_name = class_list[int(clsID)]
            living_being_label = classify_living_being(class_name)

            if living_being_label == 'person':
                # Draw bounding box for persons
                cv2.rectangle(
                    frame,
                    (int(bb[0]), int(bb[1])),
                    (int(bb[2]), int(bb[3])),
                    (255, 0, 0),  # Color for persons
                    3,
                )

                # Display class name and confidence
                font = cv2.FONT_HERSHEY_COMPLEX
                cv2.putText(
                    frame,
                    "Person",
                    (int(bb[0]), int(bb[1]) - 10),
                    font,
                    1,
                    (255, 0, 0),
                    2,
                )

            elif living_being_label == 'animal':
                # Draw bounding box for animals
                cv2.rectangle(
                    frame,
                    (int(bb[0]), int(bb[1])),
                    (int(bb[2]), int(bb[3])),
                    (0, 0, 255),  # Color for animals
                    3,
                )

                # Display class name and confidence
                font = cv2.FONT_HERSHEY_COMPLEX
                cv2.putText(
                    frame,
                    "Animal",
                    (int(bb[0]), int(bb[1]) - 10),
                    font,
                    1,
                    (0, 0, 255),
                    2,
                )

            elif living_being_label == 'bird':

                # Draw bounding box for birds
                cv2.rectangle(
                    frame,
                    (int(bb[0]), int(bb[1])),
                    (int(bb[2]), int(bb[3])),
                    (1, 50, 32),  # Color for birds
                    3,
                )

                # Display class name and confidence
                font = cv2.FONT_HERSHEY_COMPLEX
                cv2.putText(
                    frame,
                    "Bird",
                    (int(bb[0]), int(bb[1]) - 10),
                    font,
                    1,
                    (1, 50, 32),
                    2,
                )
    return frame

def classify_living_being(class_name):
    if class_name == 'person':
        return 'person'
    elif class_name in ['cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe']:
        return 'animal'
    elif class_name == 'bird':
        return 'bird'
    else:
        return None    
            

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

#Loading dehazing desmoking model
ckp = torch.load(pretrained_model_dir, map_location=device)
net = FFA(gps=gps, blocks=blocks)
net = nn.DataParallel(net)
net.load_state_dict(ckp['model'])
net = net.to(device)
net.eval()
        
#Operations -
hr = HazeRemoval()

print("1. Image\n2. Video\n3. Folder path\n")
ch = int(input("Enter you choice: "))

if(ch==1):
    imgpath = input("Enter Image path: ")
    original = Image.open(imgpath)

    hr.process_image(original)
    dehazed = hr.get_processed_image()
    
    livingbeing = livingDetection(cv2_to_tensor(dehazed))
    livingbeing = cv2.cvtColor(livingbeing, cv2.COLOR_BGR2RGB)

    outputImg = AnnotatorAndGridMaker(original, dehazed, livingbeing)
    vutils.save_image(outputImg, output_dir + os.path.basename(imgpath) + '_dehazed_img.png')

elif(ch==2):
    videopath = input("Enter Video path: ")
    option = 1
    cap = cv2.VideoCapture(videopath)
    output_video_path = output_dir + os.path.basename(videopath) + '_dehazed_video.mp4'

    #Getting video information
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 10, (frame_width, frame_height))

    frame_counter = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_counter += 1

        if frame_counter % 15 == 0:

            if option==1:
                original = cv2_to_pil(frame)
                hr.process_image(original)
                dehazed = hr.get_processed_image()
    
                livingbeing = livingDetection(cv2_to_tensor(dehazed))
                livingbeing = cv2.cvtColor(livingbeing, cv2.COLOR_BGR2RGB)

                outputImg = AnnotatorAndGridMaker(original, dehazed, livingbeing)
                #cv2.imshow('Video', livingbeing)
                vutils.save_image(outputImg, output_dir + str(frame_counter) + '_dehazed_img.png')
                out.write(livingbeing)

            elif option==2:
                original = cv2_to_pil(frame)
                dehazed = dehazingImg(original)
                livingbeing = livingDetection(dehazed)

                outputImg = AnnotatorAndGridMaker(original, dehazed, livingbeing)
                #cv2.imshow('Video', livingbeing)
                vutils.save_image(outputImg, output_dir + str(frame_counter) + '_dehazed_img.png')
                out.write(livingbeing)

    cap.release()
    out.release()
    print(f"Video saved at: {output_video_path}")

elif(ch==3):
    folderpath = input("Enter Folder path: ")
    option = 1
    img_paths = sorted(os.listdir(folderpath))

    output_dir = 'D:/Major Project/De-Smoking or De-Hazing Module/Output/Output_Images2/'

    if option == 1:
        for img_path in img_paths:
            img_path = os.path.join(folderpath, img_path)
            original = Image.open(img_path)
            dehazed = dehazingImg(original)
            livingbeing = livingDetection(dehazed)

            #livingbeing = cv2.cvtColor(livingbeing, cv2.COLOR_BGR2RGB)
            outputImg = AnnotatorAndGridMaker(original, dehazed, livingbeing)
            vutils.save_image(outputImg, output_dir + os.path.basename(img_path) + '_dehazed_img.png')

    elif option == 2:
        for img_path in img_paths:
            img_path = os.path.join(folderpath, img_path)
            original = Image.open(img_path)
            hr.process_image(original)
            dehazed = hr.get_processed_image()
    
            livingbeing = livingDetection(cv2_to_tensor(dehazed))
            livingbeing = cv2.cvtColor(livingbeing, cv2.COLOR_BGR2RGB)

            outputImg = AnnotatorAndGridMaker(original, dehazed, livingbeing)
            vutils.save_image(outputImg, output_dir + os.path.basename(img_path) + '_dehazed_img.png')

else:
    print("Invalid choice")

import os
import torch
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
from skimage.metrics import structural_similarity as ssim
from ultralytics import YOLO
warnings.filterwarnings('ignore')

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
img_dir = 'D:/Major Project/De-Smoking or De-Hazing Module/Input_Images/'
pretrained_model_dir = 'D:/Major Project/De-Smoking or De-Hazing Module/weights/' + f'model_{gps}_{blocks}_20000.pk'
output_dir = 'D:/Major Project/De-Smoking or De-Hazing Module/Output_Images/'

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
detection_colors = [(np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)) for _ in range(len(class_list))]
#Load a pretrained YOLOv8n model
model = YOLO("weights/yolov8n.pt", "v8")


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
    # Convert PyTorch tensor to numpy array
    numpy_image = tensor_image.permute(1, 2, 0).numpy()
    # Convert from float to uint8 and scale to [0, 255]
    numpy_image = (numpy_image * 255).astype(np.uint8)
    # Convert to OpenCV format (BGR)
    cv2_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
    return cv2_image


#cv2 image to tensor image
def cv2_to_tensor(cv2_image):
    # Convert from BGR to RGB
    rgb_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
    # Convert from numpy array to PyTorch tensor
    tensor_image = to_tensor(rgb_image)
    return tensor_image


#cv2 image to PIL image
def cv2_to_pil(cv2_img):
    if len(cv2_img.shape) == 2:
        # Grayscale image
        return Image.fromarray(cv2_img)
    elif len(cv2_img.shape) == 3:
        # Colored image
        return Image.fromarray(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB))
    else:
        raise ValueError("Unsupported image format")


#Operations -

#Loading dehazing desmoking model
ckp = torch.load(pretrained_model_dir, map_location=device)
net = FFA(gps=gps, blocks=blocks)
net = nn.DataParallel(net)
net.load_state_dict(ckp['model'])
net.eval()

img_paths = sorted(os.listdir(img_dir))
img_paths = [img_path for img_path in img_paths if '_1.' in img_path]

for im in img_paths:
    haze = Image.open(img_dir + im)
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

    #saving dehazed image alone for calculating accuracy of dehazing model
    #vutils.save_image(ts,'D:/Major Project/De-Smoking or De-Hazing Module/accuracy operations/dehazedimg/' + im.split('.')[0] + '.png')


    #Human Detection Operation
    frame = tensor_to_cv2(ts)
    if frame is None:
        print("Error: Could not open or read the image.")
        exit()

    #Predict objects in the frame
    detect_params = model.predict(source=[frame], conf=0.45, save=False)
    DP = detect_params[0].cpu().numpy()

    if len(DP) != 0:
        #Iterate over detected objects
        for i in range(len(detect_params[0])):
            boxes = detect_params[0].boxes
            box = boxes[i]
            clsID = box.cls.cpu().numpy()[0]
            conf = box.conf.cpu().numpy()[0]
            bb = box.xyxy.cpu().numpy()[0]

            #Check if the detected object is a person
            if class_list[int(clsID)] == 'person':
                #Draw bounding box
                cv2.rectangle(
                    frame,
                    (int(bb[0]), int(bb[1])),
                    (int(bb[2]), int(bb[3])),
                    detection_colors[int(clsID)],
                    3,
                )

                #Display class name and confidence
                font = cv2.FONT_HERSHEY_COMPLEX
                cv2.putText(
                    frame,
                    class_list[int(clsID)] + " " + str(round(conf, 3)) + "%",
                    (int(bb[0]), int(bb[1]) - 10),
                    font,
                    1,
                    (255, 255, 255),
                    2,
                )

    smoked_label = " Hazed Image - "
    dehazed_label = " De-Hazed Image - "


    #Convert tensor image and cv2 image to PIL images for annotation
    haze_no_pil = FF.to_pil_image(haze_no)
    ts_pil = cv2_to_pil(frame)


    #Add text labels to the images with adjusted font size
    draw = ImageDraw.Draw(haze_no_pil)
    font_size = 20 
    font = ImageFont.truetype("comicbd.ttf", font_size)
    draw.text((5, 5), smoked_label, fill="darkgreen", font=font)
    draw = ImageDraw.Draw(ts_pil)
    draw.text((5, 5), dehazed_label, fill="lightgreen", font=font)


    #Convert back to tensors image
    haze_no = FF.to_tensor(haze_no_pil)
    ts = FF.to_tensor(ts_pil)


    #Calculating the accuracy
    #accuracy_ground_truth_path = 'D:/Major Project/De-Smoking or De-Hazing Module/accuracy operations/truthimg/'+im.split('.')[0]+'.png'
    #dehazed_image_path = 'D:/Major Project/De-Smoking or De-Hazing Module/accuracy operations/dehazedimg/' + im.split('.')[0] + '.png'
    #accuracy = calculate_accuracy(dehazed_image_path, accuracy_ground_truth_path)
    #accuracy_label = f"{accuracy:.2%}"
    #print('Accuracy of Image '+ im.split('.')[0] + '.png is: '+accuracy_label)
    #print('\n')


    #Concatenate images
    image_grid = torch.cat((haze_no, ts), -1)


    #Saving the Final Image
    vutils.save_image(image_grid, output_dir + im.split('.')[0] + '_dehazed_img.png')

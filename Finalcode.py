import os
import torch
from torch import nn
import torchvision.transforms as tfs
import torchvision.utils as vutils
from torchvision.utils import make_grid
from torchvision.transforms.functional import to_tensor
import cv2
from PIL import Image


from darkChannelPrior import HazeRemoval
from ffaNet import FFA
from livingBeingDetection import livingDetection
from annotator import AnnotatorAndGridMaker
from accuracyCalculator import calculate_accuracy

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
output_dir = 'D:/Major Project/De-Smoking or De-Hazing Module/project_files/output/'

if not os.path.exists(output_dir):
    os.mkdir(output_dir)






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

            

#initializing FFA-Net model
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
    option = 2
    img_paths = sorted(os.listdir(folderpath))

    output_dir = 'D:/Major Project/De-Smoking or De-Hazing Module/project_files/output/'

    #Using FFA-Net model
    if option == 1:
        for img_path in img_paths:
            img_path = os.path.join(folderpath, img_path)
            original = Image.open(img_path)
            dehazed = dehazingImg(original)
            livingbeing = livingDetection(dehazed)

            #livingbeing = cv2.cvtColor(livingbeing, cv2.COLOR_BGR2RGB)
            outputImg = AnnotatorAndGridMaker(original, dehazed, livingbeing)
            vutils.save_image(outputImg, output_dir + os.path.basename(img_path) + '_dehazed_img.png')

    #using Dark Channel Prior technique
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

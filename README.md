# SmartClear---Desmoking-Dehazing-Image-with-Object-Detection-for-rescue-operation


## Overview
This GitHub repository contains the source code and documentation for our final year B.Tech project in Computer Science and Engineering. The project focuses on creating a Python GUI-based application that performs image dehazing/desmoking and living being detection. The goal is to contribute to efficient rescue operations during indoor fire disasters by clearing smoke and haze from images and detecting the presence of living beings at the disaster site.

Training Code of Feature Fusion Attention Network Architecture Model can be accessed through - https://drive.google.com/file/d/1KhRlRJyCslhM-T-0BWtZhv3StSDawKBl/view?usp=drive_link
### Output images - 

![hab3 png_dehazed_img](https://github.com/samay-jain/SmartClear---Desmoking-Dehazing-Image-with-Object-Detection-for-rescue-operation/assets/116068471/39ca3e87-a1da-4f96-8f72-5aed0f937f2f)

![hab6 png_dehazed_img](https://github.com/samay-jain/SmartClear---Desmoking-Dehazing-Image-with-Object-Detection-for-rescue-operation/assets/116068471/19980974-2111-4969-b26c-a664076bbc83)

![833_1_dehazed_img](https://github.com/samay-jain/SmartClear---Desmoking-Dehazing-Image-with-Object-Detection-for-rescue-operation/assets/116068471/841e730c-d06d-4e88-b80b-d8b46da02792)

![234_1_dehazed_img](https://github.com/samay-jain/SmartClear---Desmoking-Dehazing-Image-with-Object-Detection-for-rescue-operation/assets/116068471/f0e3aa92-4496-4ebd-8734-339a8e7ccc09)

![236_1 png_dehazed_img](https://github.com/samay-jain/SmartClear---Desmoking-Dehazing-Image-with-Object-Detection-for-rescue-operation/assets/116068471/ceb00b14-03d7-432c-bdf4-e89bc2a9dc60)


## Features
Dehazing/Desmoking Model: Utilizes the FFA-Net architecture, a Feature Fusion Attention Network, to effectively dehaze/smoke images. The model has been trained on a labeled dataset consisting of 30,000 indoor hazed images.

Living being Detection: Implements YOLOv8, a deep learning object detection model, for the detection of living beings in images.

Graphical User Interface (GUI): The application is equipped with a user-friendly GUI for easy interaction.

Image Grid Output: Displays the image before and after dehazing/desmoking and living being detection in the form of a grid. The original image is shown on the left, and the processed image is shown on the right.

## Execution
System Requirements: The project runs comfortably on systems with high processing capacity. It may be time-consuming on systems with limited resources, executing primarily on CPU.

Dependencies: Ensure that the required dependencies, including Torchvision, PIL, Torch, OpenCV, and Ultralytics YOLO, are installed on your system.

Weights for dehazing/desmoking and living being detection (Humans, Animals and Birds) can be downloaded from the link - https://drive.google.com/drive/folders/1SnwXBWQ-5dLs8wrsyJWJ71oXvXv0zWHs?usp=drive_link

## Execution Steps:

Clone the repository to your local machine.
Set up the necessary directories for input images, pretrained models, and output images.
Run the Python script finalcode.py to execute the dehazing and detection operations.
## Usage
Launch the application.
Load input images for dehazing and detection.
View the processed images in the GUI.
Save the final output images, showing the impact of dehazing and the results of human and animal detection.
## Contributing
If you'd like to contribute to the project, feel free to fork the repository and submit pull requests.

## Acknowledgments
We express our gratitude to the open-source community for providing valuable tools and frameworks that have contributed to the success of this project.

## License
This project is licensed under the MIT License. Feel free to use, modify, and distribute it as per the license terms.

## Contact
For any inquiries or suggestions, please contact Samay Jain at mr.samayjain@gmail.com We welcome your feedback and contributions.

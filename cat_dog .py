# Step 1: Install Requirements

#clone YOLOv5 and 
!git clone https://github.com/ultralytics/yolov5  # clone repo

#run the following command on CMD
%cd yolov5
%pip install -qr requirements.txt # install dependencies
%pip install -q roboflow

# importing the required library
import torch
import os
from IPython.display import Image, clear_output  # to display images
from roboflow import Roboflow

# Step 2: Download the annotated dataset from the Roboflow.

# set up environment
os.environ["DATASET_DIRECTORY"] = "/content/datasets"

# Copied from roboflow.
rf = Roboflow(api_key="KeNlv1lzrWiD9Ankyu5G")
project = rf.workspace().project("dog-or-cat-detection_yolo")
dataset = project.version(1).download("yolov5")

"""
# Step 3: Train Our Custom YOLOv5 model

Here, we are able to pass a number of arguments:
- **img:** define input image size
- **batch:** determine batch size
- **epochs:** define the number of training epochs.
- **data:** Our dataset locaiton is saved in the `dataset.location`
- **weights:** specify a path to weights to start transfer learning from. Here we choose the generic COCO pretrained checkpoint.
- **cache:** cache images for faster training
"""

!python train.py --img 416 --batch 16 --epochs 150 --data {dataset.location}/data.yaml --weights yolov5s.pt --cache

"""# Evaluate Custom YOLOv5 Detector Performance
Training losses and performance metrics are saved to Tensorboard and also to a logfile."""


# logs save in the folder "runs"
# To run the tensorboard run the following command on the cmd
%load_ext tensorboard
%tensorboard --logdir runs


# Run this command to detect all of the test images class.
# All the dected test image saved on the yolov5/runs/detect/exp directory.
python detect.py --weights runs/train/exp/weights/best.pt --img 416 --conf 0.1 --source {dataset.location}/test/images


#display inference on ALL test images
import glob
from IPython.display import Image, display

for imageName in glob.glob('/content/yolov5/runs/detect/exp/*.jpg'): #assuming JPG
    display(Image(filename=imageName))
    print("\n")

#export your model's weights for future use
from google.colab import files
files.download('./runs/train/exp/weights/best.pt')

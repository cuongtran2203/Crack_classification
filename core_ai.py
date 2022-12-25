import cv2 
import numpy as np
import numpy as np
import cv2
import threading
import time
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, models, transforms
from model.mobilenetv3 import MobileNetV3
from PIL import Image
count=0
count2=0
model =MobileNetV3()
count3=0
def pre_process(img):
    data_transforms=transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    return data_transforms(img)
lin=model.classifier
classnames=["NG","OK"]
new_lin=nn.Sequential(
        nn.Dropout(p=0.8),    # refer to paper section 6
        nn.Linear(1280,1000),
        nn.Linear(1000,2),
        nn.Softmax(dim=1)
    )
model.classifier=new_lin
model.load_state_dict(torch.load("output/best_weights.pth",map_location="cpu"),strict=False)
model.eval()
def circles_detection(img,video_path):
    global count,count2,count3
    img_copy=img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    count3+=1
    blurred = cv2.medianBlur(gray, 9) #cv2.bilateralFilter(gray,10,50,50)
    height,width=img.shape[0],img.shape[1]
    minDist = 120
    param1 =40 
    param2 = 60  #smaller value-> more false circles
    minRadius = 50
    maxRadius = 250 #10
    Radius=118
    S=False
    M=False
    # docstring of HoughCircles: HoughCircles(image, method, dp, minDist[, circles[, param1[, param2[, minRadius[, maxRadius]]]]]) -> circles
    cv2.line(img,(int(width/2),0),(int(width/2),height),(0,0,255),2)
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, minDist, param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius)
    
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0,:]:
            # draw square around circle
            coord_A=(abs(i[0]-i[2]),i[1]+i[2])
            coord_B=(i[0]+i[2],abs(i[1]-i[2]))
            mask2 = np.zeros_like(img)
            # print("i :",i[0])
            # print("line :",int(width/2))
            # print("radius :",i[2])
            cv2.circle(img, (i[0], i[1]), i[2], (0, 0,255), 2)
            try:
                img_detect=img_copy[abs(i[1]-i[2]):(i[1]+i[2]),abs(i[0]-i[2]):(i[0]+i[2])]
                img_ls = cv2.cvtColor(img_detect, cv2.COLOR_BGR2RGB)
                im_pil = Image.fromarray(img_ls)
                x=pre_process(im_pil)
                t1=time.time()
                outputs=model(x.unsqueeze(0))
                print("Time pred : ",(time.time()-t1))
                _, preds = torch.max(outputs, 1)
                print(outputs)
                cv2.putText(img,classnames[preds],(20,60),cv2.FONT_HERSHEY_COMPLEX,2,(255,25,13),2)
            except :
                print("error detected")
           
            if i[0]-int(width/2)<13 and i[0]-int(width/2)>0 :
                if abs(i[2]-Radius)>6 :
                    cv2.putText(img,"Error",(i[0]-i[2],i[1]-i[2]),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,0,255),1)
                    count2=1
                elif   abs(i[2]-Radius)<6 and classnames[preds] =="OK":
                    cv2.putText(img,"FIT",(i[0]-i[2],i[1]-i[2]),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,0,255),1)
                    count=1
                elif classnames[preds] =="NG" :
                    count2=1
               
            else :
                count=0
                count2=0
    return img,count,count2

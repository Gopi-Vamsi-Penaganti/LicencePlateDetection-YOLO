import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import random

#settings for yolo input
INPUT_WIDTH = 640
INPUT_HEIGHT =640



# Load Model
net = cv2.dnn.readNetFromONNX('Saved_Model/Model2/weights/best.onnx')
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Convert image to yolo format
def get_detections(img,net):
    image = img.copy()
    row,col,d = image.shape

    max_rc = max(row,col)
    input_image = np.zeros((max_rc,max_rc,d),dtype=np.uint8)
    input_image[0:row,0:col] = image

    # Show input to yolo
    # cv2.namedWindow('input to yolo',cv2.WINDOW_KEEPRATIO)
    # cv2.imshow('input to yolo',input_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Get Predictions
    blob = cv2.dnn.blobFromImage(input_image,1/255,(INPUT_WIDTH,INPUT_HEIGHT),swapRB=True,crop=False)
    net.setInput(blob)
    preds = net.forward()
    detections = preds[0]

    return input_image, detections

# Filter Detections based on Confidence and Probability score
# center_x,center_y,w,h,conf,prob
def non_maximum_supression(input_image,detections):
    boxes = []
    confidence = []

    image_w, image_h = input_image.shape[:2]

    x_factor = image_w/INPUT_WIDTH
    y_factor = image_h/INPUT_HEIGHT

    for i in range(len(detections)):
        row = detections[i]
        conf = row[4]
        if conf>0.4:
            class_score = row[5]
            if class_score>0.25:
                cx,cy,w,h = row[0:4]
                
                left = int((cx-0.5*w)*x_factor)
                top = int((cy-0.5*h)*y_factor)
                width = int(w*x_factor)
                height = int(h*y_factor)

                box = np.array([left,top,width,height])

                confidence.append(conf)
                boxes.append(box)

    # convert to list format
    boxes_np = np.array(boxes).tolist()
    confidence_np = np.array(confidence).tolist()
    # Non maximum supression
    #index = cv2.dnn.NMSBoxes(boxes_np,confidence_np,0.25,0.45).flatten()
    index = cv2.dnn.NMSBoxes(boxes_np,confidence_np,0.25,0.45)
    index = np.array(index).flatten()
    return boxes_np, confidence_np, index

# Drawing
def drawings(image,boxes_np,confidence_np,index):
    for ind in index:
        x,y,w,h = boxes_np[ind]
        bb_conf = confidence_np[ind]
        cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,255),2)
        cv2.rectangle(image,(x,y-30),(x+w,y),(255,0,255),-1)
        text = 'Plate : {:.0f}%'.format(bb_conf*100)
        cv2.putText(image,text,(x,y-10),cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,0.7,(255,255,255),1)
    return image


##########
#Predictions
##########
def yolo_predictions(img,net):
    # Get detections
    input_image, detections = get_detections(img,net)
    # Non Max supression
    boxes_np, confidence_np, index = non_maximum_supression(input_image,detections)
    # Draw boxes
    result_img = drawings(img,boxes_np,confidence_np,index)
    return result_img


def show(text,img):
    cv2.namedWindow(text,cv2.WINDOW_KEEPRATIO)
    cv2.imshow(text,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

## subplot of test images      
folder = 'testing'
files = [folder+'/'+x for x in os.listdir(folder)]
subs = []
fig, axes = plt.subplots(nrows=3, ncols=4)
for row in range(3):
    for col in range(4):
        img = cv2.imread(files[row*4+col])
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        result_img = yolo_predictions(img,net)
        axes[row,col].imshow(result_img)
        axes[row,col].axis('off')
plt.tight_layout()
plt.savefig('subplots.png')
plt.show()

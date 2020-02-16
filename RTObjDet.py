import cv2
import cvlib as cv
import numpy as np
from cvlib.object_detection import draw_bbox

cap = cv2.VideoCapture('D:\\Learn\\Py\\5.mp4')

fps = cap.get(cv2.CAP_PROP_FPS)
frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

traffic_score = 0

CurrentFrame = 0

while (CurrentFrame < frames):
    ret, im = cap.read()
#   im = im[400:826, 610:1520]
    
#    To deal with illumination changes, the saturation of the images (RGB)
#    are augmented with a uniform random variable.   
    
#    hsv = cv2.cvtColor(im,cv2.COLOR_BGR2HSV)

#    hsv[...,1] = hsv[...,1]*np.random.uniform(0,2)
#    Given that this illumination factor is random, it also has regularization 
#    properties, making it harder for the network to overfit the training data.
        
#    rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
    
    bbox, label, conf = cv.detect_common_objects(im)
    output_image = draw_bbox(im, bbox, label, conf)
    
#   count the number of desired labels in the list per frame 
    ctr = label.count('car') + label.count('truck')
    traffic_score = traffic_score + ctr
    
    cv2.imshow('frame2',output_image)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    CurrentFrame = CurrentFrame + 1    

cap.release()
cv2.destroyAllWindows()

avg_Tr = traffic_score/frames
print(avg_Tr)
import cv2
import numpy as np

# Playing video from file:
cap = cv2.VideoCapture('1.mp4')

ret, frame1 = cap.read()
frame1 = frame1[400:826, 610:1520] # for 1.mp4 
hsv = cv2.cvtColor(frame1,cv2.COLOR_BGR2HSV)
hsv[...,1] = hsv[...,1]*np.random.uniform(0,2)
prvs = cv2.cvtColor(frame1,cv2.COLOR_HSV2BGR)
prvs = cv2.cvtColor(prvs,cv2.COLOR_BGR2GRAY)

g = cap.get(cv2.CAP_PROP_FRAME_COUNT)
CurrentFrame = 1

while (CurrentFrame < g):
    ret, frame2 = cap.read()
    
    frame2 = frame2[400:826, 610:1520] # for 1.mp4 
    hsv = cv2.cvtColor(frame2,cv2.COLOR_BGR2HSV)
    hsv[...,1] = hsv[...,1]*np.random.uniform(0,2)
    nex = cv2.cvtColor(frame2,cv2.COLOR_HSV2BGR)
    nex = cv2.cvtColor(nex,cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(prvs,nex, None, 0.5, 3, 15, 3, 5, 1.2, cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
    
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
    
    cv2.imshow('frame2',rgb)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    elif k == ord('s'):
        cv2.imwrite('opticalfb.png',frame2)
        cv2.imwrite('opticalhsv.png',rgb)
    prvs = nex
    CurrentFrame = CurrentFrame + 1
    
cap.release()
cv2.destroyAllWindows()
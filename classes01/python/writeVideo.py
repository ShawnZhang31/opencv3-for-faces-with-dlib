import cv2
import numpy as np

# 打开摄像头
cap=cv2.VideoCapture(0)
if(cap==None):
    print("打开摄像头失败!")
else:
    print("打开摄像头成功!")

frame_width=int(cap.get(3))
frame_height=int(cap.get(4))

video=cv2.VideoWriter("output.avi",cv2.VideoWriter_fourcc('M','J','P','G'),15,(frame_width,frame_height))

while(cap):
    ret,frame=cap.read()
    if ret==True:
        video.write(frame)
        cv2.imshow("Frame",frame)

        if(cv2.waitKey(25)& 0xFF==27):
            break

    else:
        break

cap.release()
video.release()
import cv2
import numpy as np

cap =cv2.VideoCapture("../../data/videos/chaplin.mp4")

if(cap.isOpened()==False):
    print("视频文件打开失败!")

while(cap.isOpened()):
    ret,frame=cap.read()
    if ret==True:
        cv2.imshow("Frame",frame)

        if cv2.waitKey(25) & 0xFF==27:
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
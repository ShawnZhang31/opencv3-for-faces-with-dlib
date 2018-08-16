import cv2

cap = cv2.VideoCapture(0)
k=0
while(True):
    ret,frame=cap.read()
    if(k==27):
        break
    
    if(k==101 or k==69):
        cv2.putText(frame,"E is pressed, press Z of ESC",(100,180),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
    
    if(k==90 or k==122):
        cv2.putText(frame,"Z is pressed",(100,180),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
    
    cv2.imshow("Image", frame)
    k=cv2.waitKey(1000) & 0xFF

cap.release()
cv2.destroyAllWindows()
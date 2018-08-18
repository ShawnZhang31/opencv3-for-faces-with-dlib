import cv2
import numpy as np

def mouseHanler(evnet, x, y, flags, data):
    if evnet == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(data['im'],(x,y),2,(0,255,255),3,cv2.LINE_AA)
        cv2.imshow("Destination",data['im'])
        if len(data['points'])<4:
            data['points'].append([x,y])

def get_four_h_points(im):
    data={}
    data['im']=im
    data['points']=[]
    cv2.imshow("Destination",im)
    cv2.setMouseCallback("Destination",mouseHanler,data)
    cv2.waitKey(0)

    return np.vstack(data['points']).astype(float)


im_src=cv2.imread("../../data/images/first-image.jpg")
im_dst=cv2.imread("../../data/images/times-square.jpg")

print(im_src.shape)

pts_src=np.array(
                [
                    [0,0],
                    [im_src.shape[1]-1,0],
                    [im_src.shape[1]-1,im_src.shape[0]-1],
                    [0,im_src.shape[0]-1]
                ],dtype=float)

dst_temp = im_dst.copy()

pts_dst=get_four_h_points(dst_temp)

h,status=cv2.findHomography(pts_src,pts_dst)

dst_temp=cv2.warpPerspective(im_src,h,(dst_temp.shape[1],dst_temp.shape[0]))

cv2.fillConvexPoly(im_dst,pts_dst.astype(int),0,16)

im_dst=im_dst+dst_temp

cv2.imshow("Warped", im_dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
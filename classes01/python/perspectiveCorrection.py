import cv2
import numpy as np
from utils import get_four_points


im_src=cv2.imread("../../data/images/book1.jpg")
size=(300,400,3)

im_dst=np.zeros(size,np.uint8)

pts_dst= np.array([[0,0],[size[0]-1,0],[size[0]-1,size[1]-1],[0,size[1]-1]],dtype=float)

cv2.imshow("Image", im_src)
pts_src=get_four_points(im_src)

h,status=cv2.findHomography(pts_src,pts_dst)

print(size[0:2])
im_dst=cv2.warpPerspective(im_src,h,size[0:2])

cv2.imshow("Image", im_dst)
cv2.waitKey(0)
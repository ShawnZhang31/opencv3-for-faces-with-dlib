import cv2
import numpy as np

im_src=cv2.imread("../../data/images/book2.jpg")
pts_src=np.float32([[141, 131], [480, 159], [493, 630],[64, 601]])

im_dst=cv2.imread("../../data/images/book1.jpg")
pts_dst=np.float32([[318, 256],[534, 372],[316, 670],[73, 473]])

ho,status=cv2.findHomography(pts_src,pts_dst)

im_out=cv2.warpPerspective(im_src,ho,(im_src.shape[1],im_dst.shape[0]))

print(im_src.shape)

cv2.imshow("Source Image", im_src)
cv2.imshow("Destination", im_dst)
cv2.imshow("Warp", im_out)

cv2.waitKey(0)
cv2.destroyAllWindows()


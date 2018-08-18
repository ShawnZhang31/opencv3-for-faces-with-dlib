import cv2
import numpy as np

inp=np.float32([[50, 50], [100, 100], [200, 150]])

output=np.float32([[72, 51], [142, 101], [272, 136]])

output2=np.float32([[77, 76], [152, 151], [287, 236]])

warpMat=cv2.getAffineTransform(inp,output)
warpMat2=cv2.getAffineTransform(inp,output2)

print("Warp Matrix 1:\n {} \n".format(warpMat))
print("Warp Matrix 2:\n {} \n".format(warpMat2))
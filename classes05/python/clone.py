import cv2
import numpy as np

def seamlessCloningExample():
    src = cv2.imread("../data/images/airplane.jpg")
    dst = cv2.imread("../data/images/sky.jpg")

    srcMask = np.zeros(src.shape, src.dtype)

    poly = np.array([ [4,80], [30,54], [151,63], [254,37], [298,90], [272,134], [43,122] ], np.int32)
    cv2.fillPoly(srcMask, [poly], (255, 255, 255))
    center = (800,100)
    output = cv2.seamlessClone(src, dst, srcMask, center, cv2.NORMAL_CLONE)
    cv2.imshow("Seamless Cloning Example", output)
    # cv2.waitKey(0)

def normalVersusMixedCloningExample():
    im = cv2.imread("../data/images/wood-texture.jpg")
    obj= cv2.imread("../data/images/iloveyouticket.jpg")
    mask = 255 * np.ones(obj.shape, obj.dtype)
    width, height, channels = im.shape
    center = (int(height/2), int(width/2))
    normalClone = cv2.seamlessClone(obj, im, mask, center, cv2.NORMAL_CLONE)
    mixedClone = cv2.seamlessClone(obj, im, mask, center, cv2.MIXED_CLONE)
    cv2.imshow("NORMAL_CLONE Example", normalClone)
    cv2.imshow("MIXED_CLONE Example", mixedClone)
    cv2.waitKey(0)
if __name__ =="__main__":
    seamlessCloningExample()
    normalVersusMixedCloningExample()
# Copyright (c) 2015 Matthew Earl
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
# OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN
# NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
# USE OR OTHER DEALINGS IN THE SOFTWARE.

import cv2
import numpy as np

def correctColours(im1, im2, points):
    
  blurAmount = 0.5 * np.linalg.norm(np.array(points)[38] - np.array(points)[43])
  blurAmount = int(blurAmount)

  if blurAmount % 2 == 0:
    blurAmount += 1
  
  im1Blur = cv2.blur(im1, (blurAmount, blurAmount), 0)
  im2Blur = cv2.blur(im2, (blurAmount, blurAmount), 0)
  
  # Avoid divide-by-zero errors.
  im2Blur += (2 * (im2Blur <= 1)).astype(im2Blur.dtype)
  
  ret = np.uint8((im2.astype(np.float32) * im1Blur.astype(np.float32) /
                                              im2Blur.astype(np.float32)).clip(0,255))
  return ret
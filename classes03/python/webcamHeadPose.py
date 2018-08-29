#!/usr/bin/python
# Copyright 2017 BIG VISION LLC ALL RIGHTS RESERVED
#
# This code is made available to the students of
# the online course titled "Computer Vision for Faces"
# by Satya Mallick for personal non-commercial use.
#
# Sharing this code is strictly prohibited without written
# permission from Big Vision LLC.
#
# For licensing and other inquiries, please email
# spmallick@bigvisionllc.com
#
import sys
import cv2
import dlib
import numpy as np
from renderFace import renderFace

SKIP_FRAMES = 20
RESIZE_HEIGHT = 320
PREDICTOR_PATH = "../../common/resources/shape_predictor_68_face_landmarks.dat"


# 3D Model Points of selected landmarks in an arbitrary frame of reference
def get3dModelPoints():
  modelPoints = [[0.0, 0.0, 0.0],
                 [0.0, -330.0, -65.0],
                 [-225.0, 170.0, -135.0],
                 [225.0, 170.0, -135.0],
                 [-150.0, -150.0, -125.0],
                 [150.0, -150.0, -125.0]]
  return np.array(modelPoints, dtype=np.float64)


# 2D landmark points from all landmarks
def get2dImagePoints(shape):
  imagePoints = [[shape.part(30).x, shape.part(30).y],
                 [shape.part(8).x, shape.part(8).y],
                 [shape.part(36).x, shape.part(36).y],
                 [shape.part(45).x, shape.part(45).y],
                 [shape.part(48).x, shape.part(48).y],
                 [shape.part(54).x, shape.part(54).y]]
  return np.array(imagePoints, dtype=np.float64)


# Camera Matrix from focal length and focal center
def getCameraMatrix(focalLength, center):
  cameraMatrix = [[focalLength, 0, center[0]],
                  [0, focalLength, center[1]],
                  [0, 0, 1]]
  return np.array(cameraMatrix, dtype=np.float64)

try:
  # Create a VideoCapture object
  cap = cv2.VideoCapture(0)

  # Check if OpenCV is able to read feed from camera
  if (cap.isOpened() is False):
    print("Unable to connect to camera")
    sys.exit(0)

  # Just a place holder. Actual value calculated after 100 frames.
  fps = 30.0

  # Get first frame
  ret, im = cap.read()

  # We will use a fixed height image as input to face detector
  if ret == True:
    height = im.shape[0]
    # calculate resize scale
    RESIZE_SCALE = float(height)/RESIZE_HEIGHT
    size = im.shape[0:2]
  else:
    print("Unable to read frame")
    sys.exit(0)
  
  # Load face detection and pose estimation models
  detector = dlib.get_frontal_face_detector()
  predictor = dlib.shape_predictor(PREDICTOR_PATH)

  # initiate the tickCounter
  t = cv2.getTickCount()
  count = 0

  # Grab and process frames until the main window is closed by the user.
  while(True):

    # start tick counter if count is zero
    if count==0:
      t = cv2.getTickCount()

    # Grab a frame  
    ret, im = cap.read()

    # create imSmall by resizing image by resize scale
    imSmall= cv2.resize(im, None, fx = 1.0/RESIZE_SCALE, fy = 1.0/RESIZE_SCALE, interpolation = cv2.INTER_LINEAR)

    # Process frames at an interval of SKIP_FRAMES.
    # This value should be set depending on your system hardware
    # and camera fps.
    # To reduce computations, this value should be increased
    if (count % SKIP_FRAMES == 0):

      # Detect faces
      faces = detector(imSmall, 0)

    # get 3D model points
    modelPoints = get3dModelPoints()

    # Iterate over faces
    for face in faces:
      # Since we ran face detection on a resized image,
      # we will scale up coordinates of face rectangle
      newRect = dlib.rectangle(int(face.left() * RESIZE_SCALE),
                               int(face.top() * RESIZE_SCALE),
                               int(face.right() * RESIZE_SCALE),
                               int(face.bottom() * RESIZE_SCALE))
      
      # Find face landmarks by providing reactangle for each face
      shape = predictor(im, newRect)
      
      # Draw landmarks over face
      renderFace(im, shape)

      # get 2D landmarks from Dlib's shape object
      imagePoints = get2dImagePoints(shape)

      # Camera parameters
      rows, cols, ch = im.shape
      focalLength = cols
      cameraMatrix = getCameraMatrix(focalLength, (rows/2, cols/2))

      # Assume no lens distortion
      distCoeffs = np.zeros((4, 1), dtype=np.float64)

      # calculate rotation and translation vector using solvePnP
      success, rotationVector, translationVector = cv2.solvePnP(modelPoints, imagePoints, cameraMatrix, distCoeffs)

      # Project a 3D point (0, 0, 1000.0) onto the image plane.
      # We use this to draw a line sticking out of the nose
      noseEndPoints3D = np.array([[0, 0, 1000.0]], dtype=np.float64)
      noseEndPoint2D, jacobian = cv2.projectPoints(noseEndPoints3D, rotationVector, translationVector, cameraMatrix, distCoeffs)

      # points to draw line
      p1 = (int(imagePoints[0, 0]), int(imagePoints[0, 1]))
      p2 = (int(noseEndPoint2D[0, 0, 0]), int(noseEndPoint2D[0, 0, 1]))

      # draw line using points P1 and P2
      cv2.line(im, p1, p2, (110, 220, 0), thickness=2, lineType=cv2.LINE_AA)
      # Print actual FPS
      cv2.putText(im, "fps: {}".format(fps), (50, size[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

      # Resize image for display
      imDisplay = cv2.resize(im, None, fx=0.5, fy=0.5)
      cv2.imshow("webcam Head Pose", imDisplay)

      # WaitKey slows down the runtime quite a lot
      # So check every 15 frames
      if (count % 15 == 0):
        key = cv2.waitKey(1) & 0xFF

        # Stop the program.
        if key==27:  # ESC
          # If ESC is pressed, exit.
          sys.exit()

      # Calculate actual fps
      # increment frame counter
      count = count + 1
      # calculate fps at an interval of 100 frames
      if (count == 100):
        t = (cv2.getTickCount() - t)/cv2.getTickFrequency()
        fps = 100.0/t
        count = 0

  cap.release()
  cv2.destroyAllWindows()
  
except Exception as e:
  print(e)

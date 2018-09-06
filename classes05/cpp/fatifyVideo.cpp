#include <iostream>
#include <dlib/opencv.h>
#include <opencv2/highgui/highgui.hpp>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing.h>
#include <opencv2/core/core.hpp>
#include "faceBlendCommon.hpp"
#include "mls.hpp"

using namespace cv;
using namespace std;
using namespace dlib;

// Variables for resizing to a standard height
#define RESIZE_HEIGHT 360
#define FACE_DOWNSAMPLE_RATIO 1.5
#define SKIP_FRAMES 1

int main(int argc, char** argv)
{
  // Get the face detector
  frontal_face_detector faceDetector = dlib::get_frontal_face_detector();
  
  // The landmark detector is implemented in the shape_predictor class
  shape_predictor landmarkDetector;
  
  // Load the landmark model
  deserialize("../../common/shape_predictor_68_face_landmarks.dat") >> landmarkDetector;
  
  // Amount of bulge to be given for fatify
  float offset = 1.5;

  // Points that should not move
  int anchorPoints[] = {1, 15, 30};
  std::vector<int> anchorPointsArray (anchorPoints, anchorPoints + sizeof(anchorPoints) / sizeof(int) );
  
  // Points that will be deformed
  int deformedPoints[] = {5, 6, 8, 10, 11};
  std::vector<int> deformedPointsArray (deformedPoints, deformedPoints + sizeof(deformedPoints) / sizeof(int) );

  // accept command line arguments for amount of fatify
  if (argc == 2)
  {   
    offset = atof(argv[1]);
  }
  
  // Setup the video stream
  // Change the argument to 0 to read from webcam
  cv::VideoCapture cap(0);
  cv::Mat src;

  // Read a frame initially to assign memory for the frame and calculate new height 
  cap >> src;
  int height = src.rows;
  float IMAGE_RESIZE = (float)height/RESIZE_HEIGHT;

    // Variables for Optical flow  calculation
  TermCriteria termcrit(TermCriteria::COUNT|TermCriteria::EPS,20,0.03);
  Size subPixWinSize(10,10), winSize(101,101);
  double eyeDistance, sigma;
  bool eyeDistanceNotCalculated = true;

  std::vector<Point2f> landmarksPrev ;
  std::vector<Point2f> landmarksNext ;
  
  Mat srcGray, srcGrayPrev;

  int count = 0;
  while(1)
  {
    double t = (double)cv::getTickCount();
    
    // Read an image and get the landmark points
    cap >> src;
    cv::resize(src, src, cv::Size(), 1.0/IMAGE_RESIZE, 1.0/IMAGE_RESIZE);

    std::vector<Point2f> landmarks;
    if (count % SKIP_FRAMES == 0)
    {
      landmarks = getLandmarks(faceDetector, landmarkDetector, src, (float)FACE_DOWNSAMPLE_RATIO);
      cout << "Face Detector" << endl;
    }
    if(landmarks.size() != 68)
    {   
      cout << "Points not detected" << endl;
      continue;
    }

    ////////// Calculation of Optical flow and Stabilization of Landmark points ////////////
    if(!landmarksPrev.size())
    {
      landmarksPrev = landmarks;
    }

    if ( eyeDistanceNotCalculated )
    {
      eyeDistance = norm(landmarks[36] - landmarks[45]);
      winSize = cv::Size(2 * int(eyeDistance/4) + 1,  2 * int(eyeDistance/4) + 1);
      eyeDistanceNotCalculated = false;
      sigma = eyeDistance * eyeDistance / 400;
    }

    cvtColor(src, srcGray, CV_BGR2GRAY);
    
    if(srcGrayPrev.empty())
      srcGrayPrev = srcGray.clone();
    
    std::vector<uchar> status;
    std::vector<float> err;

    // Calculate Optical Flow based estimate of the point in this frame
    calcOpticalFlowPyrLK(srcGrayPrev, srcGray, landmarksPrev, landmarksNext, status, err, winSize,
                         5, termcrit, 0, 0.001);     

    // Final landmark points are a weighted average of detected landmarks and tracked landmarks
    for (unsigned long k = 0; k < landmarks.size(); ++k)
    {
      double n = norm(landmarksNext[k] - landmarks[k]);
      double alpha = exp(-n*n/sigma);
      landmarks[k] = (1 - alpha) * landmarks[k] + alpha * landmarksNext[k];
      constrainPoint(landmarks[k], src.size());
    }

    // Update varibales for next pass
    landmarksPrev = landmarks;
    srcGrayPrev = srcGray.clone();        
    
    /////////// Finished Stabilization code   //////////////////////////////////

    // Set the center of face to be the nose tip
    Point2f center (landmarks[30]);

    // Variables for storing the original and deformed points
    std::vector<Point2f> srcPoints, dstPoints;

    // Adding the original and deformed points using the landmark points
    for( int i = 0; i < anchorPointsArray.size(); i++)
    {
      srcPoints.push_back(landmarks[anchorPointsArray[i]]);
      dstPoints.push_back(landmarks[anchorPointsArray[i]]);
    }
    for( int i = 0; i < deformedPointsArray.size(); i++)
    {
      srcPoints.push_back(landmarks[deformedPointsArray[i]]);
      Point2f pt( offset*(landmarks[deformedPointsArray[i]].x - center.x) + center.x, offset*(landmarks[deformedPointsArray[i]].y - center.y) + center.y);
      dstPoints.push_back(pt);
    }

    // Adding the boundary points to keep the image stable globally
    getEightBoundaryPoints(src.size(), srcPoints);
    getEightBoundaryPoints(src.size(), dstPoints);

    // Performing moving least squares deformation on the image using the points gathered above
    Mat dst = Mat::zeros(src.rows, src.cols, CV_8UC3);
    MLSWarpImage( src, srcPoints, dst, dstPoints, 0 );

    cout << "time taken " << ((double)cv::getTickCount() - t)/cv::getTickFrequency() << endl;

    imshow("Distorted",dst);
    int k = cv::waitKey(1);
    // Quit if  ESC is pressed
    if (k == 27)
    {
      break;
    }
    count++;
  }
  
  cap.release();  
  return 0;
}

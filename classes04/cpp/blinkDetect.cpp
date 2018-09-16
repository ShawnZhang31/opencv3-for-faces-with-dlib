/*
 Copyright 2017 BIG VISION LLC ALL RIGHTS RESERVED
 
 This program is distributed WITHOUT ANY WARRANTY to the
 Plus and Premium membership students of the online course
 titled "Computer Visionfor Faces" by Satya Mallick for
 personal non-commercial use.
 
 Sharing this code is strictly prohibited without written
 permission from Big Vision LLC.
 
 For licensing and other inquiries, please email
 spmallick@bigvisionllc.com
 
 */

#include "opencv2/objdetect.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/plot.hpp"

#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing.h>
#include <dlib/image_io.h>
#include <dlib/gui_widgets.h>
#include <dlib/opencv.h>

#include <iostream>
#include <fstream>

using namespace std;
using namespace cv;
using namespace dlib;

#define RESIZE_HEIGHT 320
#define FACE_DOWNSAMPLE_RATIO_DLIB 1

Mat frame, eye;

float blinkTime = 0.2;       // 200 ms
float drowsyTime = 1.0;      // 1000 ms
int drowsyLimit = 0;                
int falseBlinkLimit = 0;  

int state = 0;

float thresh = 0.43;
float normalizedCount = 0;
int eyeRegionCount = 0;

// Variables for dlib face landmark detector
frontal_face_detector detector = get_frontal_face_detector();
string modelPath = "../../common/resources/shape_predictor_68_face_landmarks.dat";
shape_predictor pose_model;

// dlib points for eyes
static int lefteye[] = {36, 37, 38, 39, 40, 41};
std::vector<int> lefteye_index (lefteye, lefteye + sizeof(lefteye) / sizeof(lefteye[0]) );
static int righteye[] = {42, 43, 44, 45, 46, 47};
std::vector<int> righteye_index (righteye, righteye + sizeof(righteye) / sizeof(righteye[0]) );

int checkEyeStatus( full_object_detection landmarks )
{
  // Create a black image to be used as a mask for the eyes
  Mat mask = Mat::zeros(frame.rows, frame.cols, frame.depth());

  // Create a convex hull using the points of the left and right eye
  std::vector<Point> hullLeftEye;
  for(int i = 0; i < lefteye_index.size(); i++)
  {
    Point pt( landmarks.part(lefteye_index[i]).x() , landmarks.part(lefteye_index[i]).y() ); 
    hullLeftEye.push_back(pt);
  }
  fillConvexPoly(mask, &hullLeftEye[0], hullLeftEye.size(), Scalar(255,255,255));

  std::vector<Point> hullRightEye;
  for(int i = 0; i < righteye_index.size(); i++)
  {
    Point pt( landmarks.part(righteye_index[i]).x() , landmarks.part(righteye_index[i]).y() ); 
    hullRightEye.push_back(pt);
  }
  fillConvexPoly(mask, &hullRightEye[0], hullRightEye.size(), Scalar(255,255,255));

  // imshow("mask", mask);
  // find the distance between the tips of eye
  int lenLeftEyeX = landmarks.part(lefteye_index[3]).x() - landmarks.part(lefteye_index[0]).x();
  int lenLeftEyeY = landmarks.part(lefteye_index[3]).y() - landmarks.part(lefteye_index[0]).y();
  float lenLeftEyeSquare = lenLeftEyeX*lenLeftEyeX + lenLeftEyeY*lenLeftEyeY;

  // find the area under the eye region
  eyeRegionCount = cv::countNonZero(mask == 255);   

  // normalize the area by the length of eye
  // The threshold will not work without the normalization
  // the same amount of eye opening will have more area if it is close to the camera
  normalizedCount = (float)eyeRegionCount/lenLeftEyeSquare; 

  eye = Mat::zeros(frame.rows, frame.cols, frame.depth());
  
  frame.copyTo(eye,mask);

  int eyeStatus = 1;          // 1 -> Open, 0 -> closed
  if (normalizedCount < thresh)
    eyeStatus = 0;

  return eyeStatus;

}


//simple finite state machine to keep track of the blinks. we can change the behaviour as needed.
int checkBlinkStatus(int eyeStatus, int& blinkCount, int& drowsy)
{
  //open state and false blink state
  if( state >=0 && state <= falseBlinkLimit)
  {
    // if eye is open then stay in this state
    if(eyeStatus)
      state = 0;
    // else go to next state
    else
      state++;
  }

  //closed state for (drowsyLimit - falseBlinkLimit) frames
  else if(state > falseBlinkLimit && state <= drowsyLimit)
  {
    if(eyeStatus)
    {
      state = 0;
      blinkCount++;
      return 1;
    }
    else
      state++;
  }

  // Extended closed state -- drowsy
  else
  {
    if(eyeStatus)
    {
      state = 0;
      blinkCount++;
      drowsy = 0;
      return 1;
    }
    else
      drowsy = 1;
  }
  return 0;
}

int main( int argc, char** argv )
{
  cout << "USAGE : ./blinkDetect <threshold (default " << thresh << " )>" << endl;
  if (argc == 2)
  {   
    thresh = atof(argv[1]);
  }

  VideoCapture capture;
  deserialize(modelPath) >> pose_model;
  std::vector<dlib::rectangle> faces ;

  capture.open( 0 );

  int blinkCount = 0;
  int drowsy = 0;
  double t = 0;

  ///////////////////////////////////////////////////////////////////////////////////////////////////////
  // FIND the FPS or SPF for your computer
  // Different computers will have relatively Different speeds
  // Since all operations are on frame basis
  // We want to find how many frames correspond to the blink and drowsy limit
  
  // Reading some frames to adjust the sensor to the lighting
  for( int i =0; i< 5 ;i ++)
    capture.read(frame);

  // Variables used for calculating FPS robustly
  float totalTime = 0.0;
  int validFrames = 0;
  int dummyFrames = 50;
  float spf = 0;
  double timeLandmarks = 0.0;

  while( validFrames < dummyFrames)
  {
    t = (double)cv::getTickCount();

    capture.read(frame);
    validFrames++;
    int height = frame.rows;
    float IMAGE_RESIZE = (float)height/RESIZE_HEIGHT;

    cv::resize(frame, frame, cv::Size(), 1.0/IMAGE_RESIZE, 1.0/IMAGE_RESIZE);
    cv::Size size = frame.size();
    cv::Mat frame_small;
    cv::resize(frame, frame_small, cv::Size(), 1.0/FACE_DOWNSAMPLE_RATIO_DLIB, 1.0/FACE_DOWNSAMPLE_RATIO_DLIB);

    cv_image<bgr_pixel> cimg(frame);
    cv_image<bgr_pixel> cimg_small(frame_small);
    // Detect face 
    faces = detector(cimg_small);

    timeLandmarks = ((double)cv::getTickCount() - t)/cv::getTickFrequency();

    // if face not detected then dont add this time to the calculation 
    if (!faces.size())
    {
      validFrames--;
      putText(frame, "Unable to detect face, Please check proper lighting", Point(10, 50), FONT_HERSHEY_COMPLEX, 0.4, Scalar(0, 0, 255), 1, LINE_AA);
      putText(frame, "Or Decrease FACE_DOWNSAMPLE_RATIO", Point(10, 150), FONT_HERSHEY_COMPLEX, 0.5, Scalar(0, 0, 255), 1, LINE_AA);
      imshow("Blink Detection Demo ",frame);
      if ( (waitKey(1) & 0xFF) == 27)
        return 0;
    }
    else
      totalTime += timeLandmarks;
  }

  spf = totalTime/dummyFrames ; 
  cout  << "SPF (seconds per frame) " << spf << endl;
  drowsyLimit = drowsyTime/spf;
  falseBlinkLimit = blinkTime/spf;
  cout << "drowsyLimit " << drowsyLimit << " frames ("<< drowsyLimit*spf*1000 << " ms )" << ", False blink limit : " << falseBlinkLimit << " frames (" << (falseBlinkLimit)*spf*1000 << " ms )" << endl;

  ///////////////////////////////////////////////////////////////////////////////////////////////////////

  if ( ! capture.isOpened() ) 
  { 
    cout << "--(!)Error opening video capture " << endl; 
    return -1; 
  }

  while ( capture.read(frame) )
  {
    if( frame.empty() )
    {
      cout << " --(!) No captured frame -- Break!" << endl;
      break;
    }

    int height = frame.rows;
    float IMAGE_RESIZE = (float)height/RESIZE_HEIGHT;
    cv::resize(frame, frame, cv::Size(), 1.0/IMAGE_RESIZE, 1.0/IMAGE_RESIZE);
    cv::Size size = frame.size();

    cv::Mat frame_small;
    cv::resize(frame, frame_small, cv::Size(), 1.0/FACE_DOWNSAMPLE_RATIO_DLIB, 1.0/FACE_DOWNSAMPLE_RATIO_DLIB);

    cv_image<bgr_pixel> cimg(frame);
    cv_image<bgr_pixel> cimg_small(frame_small);

    // Detect face 
    faces = detector(cimg_small);
    if (!faces.size())
    {
      putText(frame, "Unable to detect face, Please check proper lighting", Point(10, 50), FONT_HERSHEY_COMPLEX, 0.5, Scalar(0, 0, 255), 1, LINE_AA);
      putText(frame, "Or Decrease FACE_DOWNSAMPLE_RATIO", Point(10, 150), FONT_HERSHEY_COMPLEX, 0.5, Scalar(0, 0, 255), 1, LINE_AA);
      imshow("Blink Detection Demo ",frame);
      if ( (waitKey(1) & 0xFF) == 27)
        return 0;
      continue;
    } 

    dlib::rectangle r(
                (long)(faces[0].left() * FACE_DOWNSAMPLE_RATIO_DLIB),
                (long)(faces[0].top() * FACE_DOWNSAMPLE_RATIO_DLIB),
                (long)(faces[0].right() * FACE_DOWNSAMPLE_RATIO_DLIB),
                (long)(faces[0].bottom() * FACE_DOWNSAMPLE_RATIO_DLIB)
                );

    full_object_detection landmarks;
    landmarks = pose_model(cimg, r);

    // check whether eye is open or close
    int eyeStatus = checkEyeStatus(landmarks);

    // pass the eyestatus to the state machine
    // to determine the blink status by checking history
    int blinkStatus = checkBlinkStatus(eyeStatus,blinkCount,drowsy);
    
    // Plot the eyepoints on the face for showing
    for (int i = 0; i < righteye_index.size() ; i++)
    {
      Point pt( landmarks.part(righteye_index[i]).x() , landmarks.part(righteye_index[i]).y() );
      circle(frame, pt, 1, Scalar(0,0,255), 1, 8);
    }
    for (int i = 0; i < lefteye_index.size() ; i++)
    {
      Point pt( landmarks.part(lefteye_index[i]).x() , landmarks.part(lefteye_index[i]).y() );
      circle(frame, pt, 1, Scalar(0,0,255), 1, 8);
    }

    // Overlay the blink count on the frame
    if(drowsy)
    {   
      putText(eye, cv::format("state: %d, blinks: %d",state,blinkCount), cv::Point(10, 50), cv::FONT_HERSHEY_COMPLEX, .9, cv::Scalar(255, 255, 255), 2);
      putText(frame, cv::format("!!! DROWSY !!! "), cv::Point(50, 100), cv::FONT_HERSHEY_COMPLEX, 1.5, cv::Scalar(0, 0, 255), 3);
    }
    else
    {
      putText(eye, cv::format("state: %d, blinks: %d",state,blinkCount), cv::Point(10, 50), cv::FONT_HERSHEY_COMPLEX, .9, cv::Scalar(255, 255, 255), 2);
      putText(frame, cv::format("Blinks : %d",blinkCount), cv::Point(50, 50), cv::FONT_HERSHEY_COMPLEX, .9, cv::Scalar(255, 255, 255), 3);
    }
    
    imshow("Blink Detection Demo ",frame);

    char c = (char)waitKey(1);
    if( c == 27 )
      break;  // escape
  }
  
  capture.release();
  destroyAllWindows();
  return 0;
}

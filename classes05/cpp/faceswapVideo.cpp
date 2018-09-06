#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing.h>
#include <dlib/image_io.h>
#include <dlib/gui_widgets.h>
#include <opencv2/imgproc.hpp> 
#include <iostream>
#include <fstream>
#include <string> 
#include <dlib/opencv.h>
#include <stdlib.h>
#include "faceBlendCommon.hpp"
#include "colorCorrection.hpp"


using namespace cv;
using namespace std;
using namespace dlib;

#define RESIZE_HEIGHT 360
#define FACE_DOWNSAMPLE_RATIO 1.5
#define SKIP_FRAMES 2

int main( int argc, char** argv)
{   
  // Load face detection and pose estimation models.
  frontal_face_detector detector = get_frontal_face_detector();
  shape_predictor predictor;
  string modelPath = "../../common/shape_predictor_68_face_landmarks.dat";
  string filename = "../data/images/obama.jpg";

  // accept command line arguments for image file
  cout << "USAGE" << endl << "./FaceSwap <filename> " << endl;
  
  if (argc == 2)
  {   
    filename = argv[1];
  }

  deserialize(modelPath) >> predictor;

  //Read input image and find landmarks
  std::vector<Point2f> points1;
  Mat img1 = imread(filename);

  int height = img1.rows;
  float IMAGE_RESIZE = (float)height/RESIZE_HEIGHT;
  cv::resize(img1, img1, cv::Size(), 1.0/IMAGE_RESIZE, 1.0/IMAGE_RESIZE);

  points1 = getLandmarks(detector, predictor, img1, (float)FACE_DOWNSAMPLE_RATIO);
  img1.convertTo(img1, CV_32F);
  
  // Create a convex hull from the facial points
  std::vector<int> hullIndex;
  convexHull(points1, hullIndex, false, false);

  // Add the points on the mouth to the convex hull to create delaunay triangles
  for(int i=48; i<59;i++)
  {
    hullIndex.push_back(i);
  }
  
  // Create Delaunay triangles
  std::vector< std::vector<int> > dt;
  Rect rect(0, 0, img1.cols, img1.rows);
  std::vector<Point2f> hull1 ;

  for(int i = 0; i < hullIndex.size(); i++)
  {
    hull1.push_back(points1[hullIndex[i]]);
  }
  calculateDelaunayTriangles(rect, hull1, dt);
  
  cout << "processed input image";

  //process input from webcam or video file
  cv::VideoCapture cap("../data/videos/introduce.mp4");
  cv::Mat img2;

  // Read a frame initially to assign memory for the frame and calculate new height 
  cap >> img2;
  height = img2.rows;
  IMAGE_RESIZE = (float)height/RESIZE_HEIGHT;

  // Declare the variable for landmark points
  std::vector<Point2f> points2;

  // Some variables for tracking time
  int count = 0;
  double t = (double)cv::getTickCount();
  double fps = 30.0;

  // Variables for Optical flow  calculation
  TermCriteria termcrit(TermCriteria::COUNT|TermCriteria::EPS,20,0.03);
  Size subPixWinSize(10,10), winSize(101,101);
  double eyeDistance, sigma;
  bool eyeDistanceNotCalculated = true;

  std::vector<Point2f> hull2Prev ;
  std::vector<Point2f> hull2Next ;
  
  Mat img2Gray, img2GrayPrev;

  Mat result, output, img1Warped;
  
  namedWindow("After Blending");

  // Main Loop
  while(cap.read(img2))
  {
    if ( count == 0 )
      t = (double)cv::getTickCount();

    double time_detector = (double)cv::getTickCount();

    cv::resize(img2, img2, cv::Size(), 1.0/IMAGE_RESIZE, 1.0/IMAGE_RESIZE);

    // Detect face 
    if (count % SKIP_FRAMES == 0)
    {
      points2 = getLandmarks(detector, predictor, img2, (float)FACE_DOWNSAMPLE_RATIO);
      cout << "Face Detector" << endl;
    }
    if(points2.size() != 68)
    {   
      cout << "Points not detected" << endl;
      continue;
    }

    //convert Mat to float data type
    img1Warped = img2.clone();
    img1Warped.convertTo(img1Warped, CV_32F);

    // Find convex hull
    std::vector<Point2f> hull2 ;

    for(int i = 0; i < hullIndex.size(); i++)
    {
      hull2.push_back(points2[hullIndex[i]]);
    }

    ////////// Calculation of Optical flow and Stabilization of Landmark points ////////////
    if(!hull2Prev.size())
    {
      hull2Prev = hull2;
    }

    double t1 = (double)cv::getTickCount();

    if ( eyeDistanceNotCalculated )
    {
      eyeDistance = norm(points2[36] - points2[45]);
      winSize = cv::Size(2 * int(eyeDistance/4) + 1,  2 * int(eyeDistance/4) + 1);
      eyeDistanceNotCalculated = false;
      sigma = eyeDistance * eyeDistance / 400;
    }

    cvtColor(img2, img2Gray, CV_BGR2GRAY);
    
    if(img2GrayPrev.empty())
      img2GrayPrev = img2Gray.clone();
    
    std::vector<uchar> status;
    std::vector<float> err;

    // Calculate Optical Flow based estimate of the point in this frame
    calcOpticalFlowPyrLK(img2GrayPrev, img2Gray, hull2Prev, hull2Next, status, err, winSize,
                         5, termcrit, 0, 0.001);     

    // Final landmark points are a weighted average of detected landmarks and tracked landmarks
    for (unsigned long k = 0; k < hull2.size(); ++k)
    {
      double n = norm(hull2Next[k] - hull2[k]);
      double alpha = exp(-n*n/sigma);
      hull2[k] = (1 - alpha) * hull2[k] + alpha * hull2Next[k];
      constrainPoint(hull2[k], img2.size());
    }

    // Update varibales for next pass
    hull2Prev = hull2;
    img2GrayPrev = img2Gray.clone();        
    
    /////////// Finished Stabilization code   //////////////////////////////////

    // Apply affine transformation to Delaunay triangles
    for(size_t i = 0; i < dt.size(); i++)
    {    
      std::vector<Point2f> t1, t2;

      // Get points for img1, img2 corresponding to the triangles
      for(size_t j = 0; j < 3; j++)
      {
        t1.push_back(hull1[dt[i][j]]);
        t2.push_back(hull2[dt[i][j]]);
      }
      warpTriangle(img1, img1Warped, t1, t2);
    }       
    
    cout << "Stabilize and Warp time" << ((double)cv::getTickCount() - t1)/cv::getTickFrequency() << endl;

/////////////////////////   Blending   /////////////////////////////////////////////////////////////

    img1Warped.convertTo(img1Warped, CV_8UC3);

    // Color Correction of the warped image so that the source color matches that of the destination
    output = correctColours(img2, img1Warped, points2);   
    
    // imshow("Before Blending", output);

    // Create a Mask around the face
    Rect re = boundingRect(hull2);
    Point center = (re.tl() + re.br()) / 2;
    std::vector<Point> hull3;
    
    for(int i = 0; i < hull2.size()-12; i++)
    {
      //Take the points just inside of the convex hull
      Point pt1( 0.95*(hull2[i].x - center.x) + center.x, 0.95*(hull2[i].y - center.y) + center.y);
      hull3.push_back(pt1);
    }
    Mat mask1 = Mat::zeros(img2.rows, img2.cols, img2.type());
    
    fillConvexPoly(mask1,&hull3[0], hull3.size(), Scalar(255,255,255));

    // Blur the mask before blending
    cv::GaussianBlur(mask1,mask1, Size (21, 21),10);        
        
    Mat mask2 = Scalar(255,255,255) - mask1;
    // imshow("mask1",mask1);
    // imshow("mask2",mask2);

    // Perform alpha blending of the two images
    Mat temp1 = output.mul(mask1, 1.0/255);
    Mat temp2 = img2.mul(mask2,1.0/255);
    result =  temp1 + temp2;
    // imshow("temp1",temp1);
    // imshow("temp2",temp2);
    
//////////////////////////////////////////////////////////////////////////////////////////////////////
    cout << "Total time" << ((double)cv::getTickCount() - time_detector)/cv::getTickFrequency() << endl;
    imshow("After Blending", result);
    
    int k = cv::waitKey(1);
    // Quit if  ESC is pressed
    if (k == 27)
    {
      break;
    }

    count++;

    if ( count == 10)
    {
      fps = 10.0 * cv::getTickFrequency() / ((double)cv::getTickCount() - t);
      count = 0;
    }
    cout << "FPS " << fps << endl;
  }

  cap.release();
  return 1;
}

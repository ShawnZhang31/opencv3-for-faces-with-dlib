#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing.h>
#include <dlib/image_io.h>
#include <dlib/gui_widgets.h>
#include <opencv2/imgproc.hpp> // If you are using OpenCV 3
#include <iostream>
#include <fstream>
#include <string> 
#include <dlib/opencv.h>
#include <stdlib.h>
#include "faceBlendCommon.hpp"

using namespace cv;
using namespace std;
using namespace dlib;

#define RESIZE_HEIGHT 480
#define FACE_DOWNSAMPLE_RATIO 1

int selectedpoints[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 31, 32, 33, 34, 35, 55, 56, 57, 58, 59};
std::vector<int> selectedIndex (selectedpoints, selectedpoints + sizeof(selectedpoints) / sizeof(int) );

// Read points corresponding to beard, stored in text files
std::vector<Point2f> getSavedPoints(string pointsFileName)
{
  std::vector<Point2f> points;
  ifstream ifs(pointsFileName.c_str());
  float x, y;
  if (!ifs)
    cout << "Unable to open file" << endl;
  while(ifs >> x >> y)
  {
    points.push_back(Point2f(x,y));
  }
  return points;
}


int main(int argc, char** argv)
{
  string overlayFile = "../data/images/beard1.png";
  string imageFile = "../data/images/ted_cruz.jpg";

  // Load face detection and pose estimation models.
  frontal_face_detector detector = get_frontal_face_detector();
  shape_predictor predictor;
  string modelPath = "../../common/shape_predictor_68_face_landmarks.dat";
  deserialize(modelPath) >> predictor;

  if (argc == 2)
  {   
    imageFile = argv[1];
  }
  else if (argc == 3)
  {   
    imageFile = argv[1];
    overlayFile = argv[2];
  }

  // Read the beard image along with its alpha mask
  Mat beard, targetImage, beardAlphaMask;
  Mat imgWithMask = imread(overlayFile,CV_LOAD_IMAGE_UNCHANGED);
  std::vector<Mat> rgbaChannels(4);
  
  // Split into channels
  split(imgWithMask, rgbaChannels);
  
  // Extract the beard image
  std::vector<Mat> bgrchannels;
  bgrchannels.push_back(rgbaChannels[0]);
  bgrchannels.push_back(rgbaChannels[1]);
  bgrchannels.push_back(rgbaChannels[2]);

  merge(bgrchannels, beard);
  beard.convertTo(beard, CV_32F, 1.0/255.0);

  // Extract the beard mask
  std::vector<Mat> maskchannels;
  maskchannels.push_back(rgbaChannels[3]);
  maskchannels.push_back(rgbaChannels[3]);
  maskchannels.push_back(rgbaChannels[3]);

  merge(maskchannels, beardAlphaMask);
  beardAlphaMask.convertTo(beardAlphaMask, CV_32FC3);

  //Read points for beard from file
  std::vector<Point2f> featurePoints1 = getSavedPoints( overlayFile + ".txt");
  
  // Calculate Delaunay triangles
  Rect rect = boundingRect(featurePoints1);

  std::vector< std::vector<int> > dt;
  calculateDelaunayTriangles(rect, featurePoints1, dt);

  float time_detector = (double)cv::getTickCount();
  // Get the face image for putting the beard
  targetImage = imread(imageFile);
  int height = targetImage.rows;
  float IMAGE_RESIZE = (float)height/RESIZE_HEIGHT;
  cv::resize(targetImage, targetImage, cv::Size(), 1.0/IMAGE_RESIZE, 1.0/IMAGE_RESIZE);

  std::vector<Point2f> points2 = getLandmarks(detector, predictor, targetImage, (float)FACE_DOWNSAMPLE_RATIO);

  std::vector<Point2f> featurePoints2;
  for( int i = 0; i < selectedIndex.size(); i++)
  {
    featurePoints2.push_back(points2[selectedIndex[i]]);
    constrainPoint(featurePoints2[i], targetImage.size());
  }
  //convert Mat to float data type
  targetImage.convertTo(targetImage, CV_32F, 1.0/255.0);

  //empty warp image
  Mat beardWarped = Mat::zeros(targetImage.size(), beard.type());
  Mat beardAlphaMaskWarped = Mat::zeros(targetImage.size(), beardAlphaMask.type());

  // Apply affine transformation to Delaunay triangles
  for(size_t i = 0; i < dt.size(); i++)
  {
    std::vector<Point2f> t1, t2;
    // Get points for img1, targetImage corresponding to the triangles
    for(size_t j = 0; j < 3; j++)
    {
      t1.push_back(featurePoints1[dt[i][j]]);
      t2.push_back(featurePoints2[dt[i][j]]);
    }
    warpTriangle(beard, beardWarped, t1, t2);
    warpTriangle(beardAlphaMask, beardAlphaMaskWarped, t1, t2);
    // imshow("im1w",beardWarped);
    // imshow("im3w",beardAlphaMaskWarped);
    // waitKey(0);

  }
  imshow("beardWarped",beardWarped);
  imshow("beardAlphaMaskWarped",beardAlphaMaskWarped);

  Mat mask1;
  beardAlphaMaskWarped.convertTo(mask1, CV_32FC3, 1.0/255.0);
  // cv::GaussianBlur(mask1,mask1, Size (21, 21),10);        

  Mat mask2 = Scalar(1.0,1.0,1.0) - mask1;
  Mat temp1 = targetImage.mul(mask2);
  Mat temp2 = beardWarped.mul(mask1);

  imshow("mask1",mask1);
  imshow("mask2",mask2);
  imshow("temp1",temp1);
  imshow("temp2",temp2);

  Mat result = temp1 + temp2;
  imshow("result",result);

  cout << "Time taken  " << ((double)cv::getTickCount() - time_detector)/cv::getTickFrequency() << endl;
  char c = (char)waitKey(0);
  if( c == 's' )
    imwrite("warped_" + imageFile + "_" + overlayFile, result*255);

  return 0;
}

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

#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;

void seamlessCloningExample()
{
  // Read images : src image will be cloned into dst
  Mat src = imread("../data/images/airplane.jpg");
  Mat dst = imread("../data/images/sky.jpg");
  
  // Create a rough mask around the airplane.
  Mat srcMask = Mat::zeros(src.rows, src.cols, src.depth());
  
  // Define the mask as a closed polygon
  Point poly[1][7];
  poly[0][0] = Point(4, 80);
  poly[0][1] = Point(30, 54);
  poly[0][2] = Point(151,63);
  poly[0][3] = Point(254,37);
  poly[0][4] = Point(298,90);
  poly[0][5] = Point(272,134);
  poly[0][6] = Point(43,122);
  
  // fillPoly takes an array of polygons.
  // So we need to create an array even though
  // we have only one polygon
  const Point* polygons[1] = { poly[0] };
  int numPoints[] = { 7 };
  
  // Create mask by filling the polygon
  fillPoly(srcMask, polygons, numPoints, 1, Scalar(255,255,255));
  
  // The location of the center of the src in the dst
  Point center(800,100);
  
  // Seamlessly clone src into dst and put the results in output
  Mat output;
  seamlessClone(src, dst, srcMask, center, output, NORMAL_CLONE);
  
  // Display and save results
  namedWindow("Seamless Cloning Example");
  imshow("Seamless Cloning Example", output);
  
  imwrite("results/opencv-seamless-cloning-example.jpg", output);

}

void normalVersusMixedCloningExample()
{
  // Read images : src image will be cloned into dst
  Mat src = imread("../data/images/iloveyouticket.jpg");
  Mat dst = imread("../data/images/wood-texture.jpg");
  
  // Create an all white mask
  Mat srcMask = 255 * Mat::ones(src.rows, src.cols, src.depth());
  
  // The location of the center of the src in the dst
  Point center(dst.cols/2,dst.rows/2);
  
  // Seamlessly clone src into dst and put the results in output
  Mat normalClone;
  Mat mixedClone;
  
  seamlessClone(src, dst, srcMask, center, normalClone, NORMAL_CLONE);
  seamlessClone(src, dst, srcMask, center, mixedClone, MIXED_CLONE);
  
  // Display and save results
  namedWindow("NORMAL_CLONE Example");
  namedWindow("MIXED_CLONE Example");
  imshow("NORMAL_CLONE Example", normalClone);
  imshow("MIXED_CLONE Example", mixedClone);
  waitKey(0);
  
  imwrite("results/opencv-normal-clone-example.jpg", normalClone);
  imwrite("results/opencv-mixed-clone-example.jpg", mixedClone);
  
}


int main( int argc, char** argv )
{
  // A simple seamlessCloning example
  seamlessCloningExample();
  // Comparsion between NORMAL_CLONE vs MIXED_CLONE
  normalVersusMixedCloningExample();
  
}

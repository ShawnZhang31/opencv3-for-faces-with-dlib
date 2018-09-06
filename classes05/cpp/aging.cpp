#include "faceBlendCommon.hpp"

using namespace cv;
using namespace std;
using namespace dlib;

#define FACE_DOWNSAMPLE_RATIO 1

// Alpha blending using multiply and add functions
Mat& alphaBlend(Mat& alpha, Mat& foreground, Mat& background, Mat& outImage)
{
  Mat fore, back;
  multiply(alpha, foreground, fore, 1/255.0);
  multiply(Scalar::all(255)-alpha, background, back,1/255.0);
  add(fore, back, outImage);
  
  return outImage;
}

// Desaturate image
void desaturateImage(Mat &im, double scaleBy)
{
  // Convert input image to HSV
  Mat imgHSV;
  cv::cvtColor(im,imgHSV,COLOR_BGR2HSV);

  // Split HSV image into three channels.
  std::vector<Mat> channels(3);
  split(imgHSV,channels);
  
  // Multiple saturation by the scale.
  channels[1] = scaleBy * channels[1];
  
  // Merge back the three channels
  merge(channels,imgHSV);
  
  // Convert HSV to RGB
  cv::cvtColor(imgHSV,im,COLOR_HSV2BGR);
  
}

void removePolygonFromMask(Mat &mask, std::vector<Point2f> points, std::vector<int> pointsIndex)
{
  std::vector<Point> hullPoints;
  for(int i = 0; i < pointsIndex.size(); i++)
  {
    Point pt( points[pointsIndex[i]].x , points[pointsIndex[i]].y ); 
    hullPoints.push_back(pt);
  }
  fillConvexPoly(mask,&hullPoints[0], hullPoints.size(), Scalar(0,0,0));
}

void appendForeheadPoints(std::vector<Point2f> &points)
{
  
  double offsetScalp = 3.0;
  
  static int brows[] = {25, 23, 20, 18 };
  std::vector<int> browsIndex (brows, brows + sizeof(brows) / sizeof(brows[0]) );
  static int browsReference[] = {45, 47, 40, 36};
  std::vector<int> browsReferenceIndex (browsReference, browsReference + sizeof(browsReference) / sizeof(browsReference[0]) );
  
  for (unsigned long k = 0; k < browsIndex.size(); ++k)
  {
    Point2f foreheadPoint = offsetScalp * ( points[ browsIndex[k] ] - points[ browsReferenceIndex[k]]) + points[browsReferenceIndex[k]];
    points.push_back(foreheadPoint);
  }
  
}

Mat getFaceMask(Size size, std::vector<Point2f> points)
{
  
  // Left eye polygon
  static int leftEye[] = {36, 37, 38, 39, 40, 41};
  std::vector<int> leftEyeIndex (leftEye, leftEye + sizeof(leftEye) / sizeof(leftEye[0]) );
  
  // Right eye polygon
  static int rightEye[] = {42, 43, 44, 45, 46, 47};
  std::vector<int> rightEyeIndex (rightEye, rightEye + sizeof(rightEye) / sizeof(rightEye[0]) );
  
  // Mouth polygon
  static int mouth[] = {48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59};
  std::vector<int> mouthIndex (mouth, mouth + sizeof(mouth) / sizeof(mouth[0]) );
  
  // Nose polygon
  static int nose[] = {28, 31, 33, 35};
  std::vector<int> noseIndex (nose, nose + sizeof(nose) / sizeof(nose[0]) );
  
  // Find Convex hull of all points
  std::vector<Point2f> hull;
  convexHull(points, hull, false, true);
  
  // Convert to vector of Point2f to vector of Point
  std::vector<Point> hullInt;
  for(int i = 0; i < hull.size(); i++)
  {
    Point pt( hull[i].x , hull[i].y );
    hullInt.push_back(pt);
  }
  
  // Create mask such that convex hull is white.
  Mat mask = Mat::zeros(size.height, size.width, CV_8UC3);
  fillConvexPoly(mask,&hullInt[0], hullInt.size(), Scalar(255,255,255));
  
  // Remove eyes, mouth and nose from the mask.
  removePolygonFromMask(mask, points, leftEyeIndex);
  removePolygonFromMask(mask, points, rightEyeIndex);
  removePolygonFromMask(mask, points, noseIndex);
  removePolygonFromMask(mask, points, mouthIndex);
  
  return mask;
  
}


int main( int argc, char** argv)
{   
  
  // Load face detector
  frontal_face_detector faceDetector = get_frontal_face_detector();
  
  // Load landmark detector.
  shape_predictor landmarkDetector;
  deserialize("../../common/resources/shape_predictor_68_face_landmarks.dat") >> landmarkDetector;
  
  // File to copy wrinkles from
  string filename1 = "../data/images/wrinkle2.jpg";
  
  // File to apply aging
  string filename2 = "../data/images/zhanghanyu.jpeg";
  
  // Optionally load the two files
  cout << "USAGE" << endl << "./aging <wrinkle file> <original file>" << endl;
  if (argc == 2)
  {   
    filename1 = argv[1];
  }
  else if (argc == 3)
  {   
    filename1 = argv[1];
    filename2 = argv[2];
  }
  
  
  // Read images
  Mat img1 = imread(filename1);
  Mat img2 = imread(filename2);
  
  // Find landmarks.
  std::vector<Point2f> points1, points2;
  
  points1 = getLandmarks(faceDetector, landmarkDetector, img1, (float)FACE_DOWNSAMPLE_RATIO);
  points2 = getLandmarks(faceDetector, landmarkDetector, img2, (float)FACE_DOWNSAMPLE_RATIO);
  
  // Find forehead points.
  appendForeheadPoints(points1);
  appendForeheadPoints(points2);
  
  // Find Delaunay Triangulation
  std::vector< std::vector<int> > dt;
  Rect rect(0, 0, img1.cols, img1.rows);
  calculateDelaunayTriangles(rect, points1, dt);

  // Convert image for warping.
  img1.convertTo(img1, CV_32F);
  img2.convertTo(img2, CV_32F);
  
  // Warp wrinkle image to face image.
  Mat img1Warped = img2.clone();
  warpImage(img1,img1Warped, points1, points2, dt, true);
  img1Warped.convertTo(img1Warped, CV_8UC3);
  img2.convertTo(img2, CV_8UC3);

  // Calculate face mask for seamless cloning.
  Mat mask = getFaceMask(img2.size(), points2);
  
  // Seamlessly clone the wrinkle image onto original face
  Rect r1 = boundingRect(points2);
  Point center1 = (r1.tl() + r1.br()) / 2;
  Mat clonedOutput;
  seamlessClone(img1Warped,img2, mask, center1, clonedOutput, MIXED_CLONE);
  
  // Blurring face mask to alpha blend to hide seams
  Size size = mask.size();
  Mat maskSmall;
  resize(mask, maskSmall, Size(256, int((size.height) * 256.0/double(size.width))));
  erode(maskSmall, maskSmall, Mat(), Point(-1,-1), 5);
  GaussianBlur(maskSmall, maskSmall, Size(15,15), 0, 0);
  resize(maskSmall, mask, size);
  
  Mat agedImage = clonedOutput.clone();
  alphaBlend(mask, clonedOutput, img2, agedImage);
  
  // Desaturate output
  desaturateImage(agedImage, 0.8);
  
  // Display results
  Mat displayImage;
  hconcat(img2,agedImage,displayImage);
  namedWindow("Output", WINDOW_NORMAL);
  imshow("Output",displayImage);
  int k = cv::waitKey(0);
  return EXIT_SUCCESS;
}

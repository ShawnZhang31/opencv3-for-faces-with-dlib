#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;


/**
 * @brief 无缝拷贝示例
 * 
 */
void seamlessCloningExample()
{
    Mat src = imread("../data/images/airplane.jpg");
    Mat dst = imread("../data/images/sky.jpg");
    imshow("src", src);
    imshow("dst", dst);

    Mat srcMask = Mat::zeros(src.rows, src.cols, src.depth());
    Point poly[1][7];
    poly[0][0] = Point(4, 80);
    poly[0][1] = Point(30, 54);
    poly[0][2] = Point(151,63);
    poly[0][3] = Point(254,37);
    poly[0][4] = Point(298,90);
    poly[0][5] = Point(272,134);
    poly[0][6] = Point(43,122);

    const Point* polygons[1]={poly[0]};
    int numPoints[] ={7};
    fillPoly(srcMask, polygons, numPoints, 1, Scalar(255, 255, 255));
    imshow("srcMask", srcMask);

    Point center(800,100);
    Mat output=dst.clone();
    seamlessClone(src, dst, srcMask, center, output, NORMAL_CLONE);
    imshow("BLEND", output);
    
    waitKey(0);

    return;
}

void normalVersusMixedCloningExample()
{
  Mat src = imread("../data/images/iloveyouticket.jpg");
  Mat dst = imread("../data/images/wood-texture.jpg");
  Mat srcMask = 255 * Mat::ones(src.rows, src.cols, src.depth());
  Point center(dst.cols/2,dst.rows/2);
  Mat normalClone;
  Mat mixedClone;
  seamlessClone(src, dst, srcMask, center, normalClone, NORMAL_CLONE);
  seamlessClone(src, dst, srcMask, center, mixedClone, MIXED_CLONE);
  imshow("normalClone", normalClone);
  imshow("mixedClone", mixedClone);
    
  waitKey(0);

  return;
}

int main(int argc, char const *argv[])
{
  // 无缝拷贝
  seamlessCloningExample();
  normalVersusMixedCloningExample();
  return 0;
}

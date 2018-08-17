#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main(void)
{   
    // 原始图像
    Mat source = imread("../../data/images/sample.jpg");

    // 计算旋转变换的矩阵
    Mat M=getRotationMatrix2D(Point(source.cols/2, source.rows/2), 30,1.0);

    // 变换后的图像
    Mat dst;
    warpAffine(source,dst,M,Size(source.cols,source.rows));

    namedWindow("Original", WINDOW_AUTOSIZE);
    namedWindow("Rotation", WINDOW_AUTOSIZE);

    imshow("Original", source);
    imshow("Rotation", dst);

    waitKey(0);
    destroyAllWindows();

    

    return EXIT_SUCCESS;
}
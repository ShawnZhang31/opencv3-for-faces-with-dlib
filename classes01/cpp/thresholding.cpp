#include "opencv2/opencv.hpp"
 using namespace std;
 using namespace cv;

 int main(void)
 {
    //  首先读进来一张当通道的灰度图像
    Mat source=imread("../../data/images/threshold.png",IMREAD_GRAYSCALE);

    imshow("Original", source);
    
    int thresh=0;
    int maxValue=255;

    threshold(source,source,thresh,maxValue,THRESH_BINARY);
    // 显示二值化之后的图像
    imshow("Threshold", source);

    waitKey();
    destroyAllWindows();

     return EXIT_SUCCESS;
 }
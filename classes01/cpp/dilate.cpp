#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;

int main(void)
{
    Mat image;

    // 读取一张图片
    image=imread("../../data/images/truth.png", IMREAD_COLOR);

    // 创建一个核
    int dilationSize=6;
    Mat element=getStructuringElement(cv::MORPH_CROSS,cv::Size(2*dilationSize+1,2*dilationSize+1),cv::Point(dilationSize,dilationSize));

    // 定义个Mat用来装膨胀之后的图像
    Mat imageDilated;
    dilate(image,imageDilated,element);
    imshow("Original Image", image);
    imshow("Dilation",imageDilated);

    waitKey();
    
    return EXIT_SUCCESS;
}
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main(void)
{
    Mat source, scaleDown, scaleUp;

    // 读取原始图像
    source=imread("../../data/images/sample.jpg");

    // 缩放因子
    double scaleX=0.6;
    double scaleY=0.6;

    // 缩小为原来的0.6
    resize(source,scaleDown,Size(),scaleX,scaleY,INTER_LINEAR);
    // 发到到原来的1.8倍
    resize(source,scaleUp,Size(),scaleX*3,scaleY*3,INTER_LINEAR);

    // 裁切图像
    Mat crop = source(cv::Range(50,150), cv::Range(20, 200));

    // 创建显示图像的窗口
    namedWindow("Original",WINDOW_AUTOSIZE);
    namedWindow("Scaled Down",WINDOW_AUTOSIZE);
    namedWindow("Scaled Up",WINDOW_AUTOSIZE);
    namedWindow("Cropped Image",WINDOW_AUTOSIZE);

    // 显示图像
    imshow("Original",source);
    imshow("Scaled Down",scaleDown);
    imshow("Scaled Up",scaleUp);
    imshow("Cropped Image",crop);

    waitKey(0);


    return EXIT_SUCCESS;
}
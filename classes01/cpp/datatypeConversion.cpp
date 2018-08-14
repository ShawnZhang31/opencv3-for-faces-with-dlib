#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main(void)
{
    // 读取数据
    Mat source=imread("../../data/images/sample.jpg");

    // 在转变像值的时候的缩放因子
    double scale=1/255.0;
    double shift=0.0;

    // 将unsighed char转换为32位float
    source.convertTo(source,CV_32FC3,scale,shift);

    imshow("32bit",source);

    // 将float转换为unsigned char
    source.convertTo(source,CV_8UC3,1.0/scale,shift);

    imshow("unsigned char",source);

    waitKey(0);

    return EXIT_SUCCESS;
}
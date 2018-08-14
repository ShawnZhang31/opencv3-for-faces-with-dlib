#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

int main(void)
{
    // 读取图片
    Mat image = imread("../../data/images/sample.jpg");

    // 检查图片地址是否有效
    if(image.empty())
    {
        cout<<"无法读取图片文件"<<endl;
    }

    // 将图片转换成灰度图像
    Mat grayImage;
    cvtColor(image,grayImage,COLOR_BGR2GRAY);

    // 保存结果
    imwrite("imageGray.jpg",grayImage);

    // 创建一个窗口用于显示图像
    namedWindow("image",WINDOW_AUTOSIZE);
    namedWindow("gray image",WINDOW_AUTOSIZE);

    // 显示图像
    imshow("image",image);
    imshow("gray image",grayImage);

    // 只有当用户按下键的时候才销毁窗口
    waitKey(0);
    destroyAllWindows();


    return EXIT_SUCCESS;
}
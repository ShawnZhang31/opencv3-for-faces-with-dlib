#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;

int main(void)
{
    // 创建一个VideoCapture
    VideoCapture cap("../../data/videos/chaplin.mp4");

    // 检查文件是否打开成功
    if(!cap.isOpened())
    {
        cout<<"文件打开失败!"<<endl;
        return -1;
    }

    while(1)
    {
        Mat frame;
        cap >> frame;

        if(frame.empty())
            break;
        
        imshow("Frame",frame);

        char c=(char)waitKey(25);
        if(c==27)
            break;
    }

    //当使VIDEOCapture用完之后需要主动释放一下
    cap.release();

    // 关闭所有的帧
    destroyAllWindows();
    
    return EXIT_SUCCESS;
}
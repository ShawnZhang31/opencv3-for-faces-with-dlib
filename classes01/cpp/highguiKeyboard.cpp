#include <opencv2/videoio.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <string>
#include <iostream>

using namespace std;
using namespace cv;

int main(void)
{
    // 打开摄像头
    VideoCapture cap(0);
    Mat frame;
    // 监听键盘输入键
    int k=0;

    // 检查摄像头是否正常打开
    if(!cap.isOpened())
    {
        cout<<"打开摄像头失败!"<<endl;
        return -1;
    }
    else
    {
        while(1)
        {
            cap >> frame;
            if(k==27)
             break;
            
            if(k==101 || k==69)
              putText(frame,"E is pressed, press Z or ESC", Point(100,180),FONT_HERSHEY_COMPLEX,1,Scalar(0,255.0),2);
            
            if(k==90 || k==122)
              putText(frame,"Z is pressed",Point(100,180),FONT_HERSHEY_COMPLEX,1,Scalar(0,255,0),2);
            
            imshow("Image",frame);

            k=waitKey(1000) & 0xFF;
        }
    }

    cap.release();
    destroyAllWindows();

    return EXIT_SUCCESS;
}
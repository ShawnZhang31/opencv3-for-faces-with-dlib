#include <opencv2/opencv.hpp>
#include <string>

using namespace std;
using namespace cv;

int thresholdValue=150;
int thresholdType=3;
int const maxValue=255;
int const maxType=4;
int const max_BINARY_value=255;

Mat im, imGray, dst;

string windowName="Threshold Demo";
string trackbarType="Type: \n 0: Binary \n 1: Binary Inverted \n 2: Truncate \n 3: To Zero \n 4: To Zero Inverted";
string trackbarValue="Value";

/**
 * @brief 
 * 
 */
void thresholdDemo(int,void*);

int main()
{
    im=imread("../../data/images/threshold.png");

    cvtColor(im,imGray,COLOR_RGB2GRAY);

    namedWindow(windowName,CV_WINDOW_AUTOSIZE);

    createTrackbar(trackbarType, windowName, &thresholdType, maxType, thresholdDemo);

    thresholdDemo(0,0);

    while(true)
    {
        int c;
        c = waitKey(20);
        if(c==27)
         break;
        
    }

    return EXIT_SUCCESS;
}

void thresholdDemo(int,void*)
{
    threshold(imGray, dst, thresholdValue, max_BINARY_value, thresholdType);
    imshow(windowName, dst);
}
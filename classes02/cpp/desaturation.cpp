#include <opencv2/opencv.hpp>
#include <math.h>

using namespace cv;
using namespace std;

int main(void)
{
    Mat img=imread("../data/images/capsicum.jpg");

    float staturationScale=0.01;

    Mat hsvImage;
    cvtColor(img,hsvImage,COLOR_BGR2HSV);

    hsvImage.convertTo(hsvImage,CV_32F);

    vector<Mat> channels(3);
    split(hsvImage,channels);

    channels[1]=channels[1]*staturationScale;

    min(channels[1],255,channels[1]);
    max(channels[1],0,channels[1]);

    merge(channels,hsvImage);

    hsvImage.convertTo(hsvImage, CV_8UC3);

    Mat imSat;
    cvtColor(hsvImage, imSat, COLOR_HSV2BGR);

    Mat combined;
    hconcat(img,imSat,combined);

    imshow("Combined", combined);

    waitKey(0);
    destroyAllWindows();


    return EXIT_SUCCESS;
}
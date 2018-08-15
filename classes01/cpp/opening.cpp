#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main(void)
{
    Mat image=imread("../../data/images/opening.png",IMREAD_GRAYSCALE);

    int iterations=3;

    int erosionSize=3;
    Mat element=getStructuringElement(MORPH_ELLIPSE,Size(2*erosionSize+1,2*erosionSize+1),Point(erosionSize,erosionSize));

    Mat imageMorphOpened;
    morphologyEx(image,imageMorphOpened,MORPH_OPEN,element,Point(-1,-1),iterations);

    Mat imageDilation;
    dilate(image,imageDilation,element,Point(-1,-1),iterations);

    Mat imageErosion;
    erode(image,imageErosion,element,Point(-1,-1),iterations);

    imshow("Original", image);
    imshow("Dilation",imageDilation);
    imshow("Erosion",imageErosion);
    imshow("Opening",imageMorphOpened);

    waitKey(0);
    destroyAllWindows();

    return EXIT_SUCCESS;
}
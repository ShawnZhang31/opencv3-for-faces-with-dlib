#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main(void)
{
    Mat image = imread("../../data/images/closing.png",IMREAD_GRAYSCALE);

    int closingSize=10;
    Mat element = getStructuringElement(MORPH_ELLIPSE,Size(2*closingSize+1,2*closingSize+1),Point(closingSize,closingSize));

    Mat imageMorphClosed;
    morphologyEx(image,imageMorphClosed,MORPH_CLOSE,element);

    imshow("Original", image);
    imshow("Closed", imageMorphClosed);

    waitKey(0);
    destroyAllWindows();

    return EXIT_SUCCESS;
}
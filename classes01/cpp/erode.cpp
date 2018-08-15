#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main(void)
{
    Mat image;
    image=imread("../../data/images/truth.png",IMREAD_COLOR);
    
    int erosionSize=6;
    Mat element=getStructuringElement(MORPH_CROSS,Size(2*erosionSize+1,2*erosionSize+1),Point(erosionSize,erosionSize));

    Mat imageEroded;
    erode(image,imageEroded,element);

    imshow("Original",image);
    imshow("Eroded Image",imageEroded);

    waitKey(0);
    destroyAllWindows();

    return EXIT_SUCCESS;
}
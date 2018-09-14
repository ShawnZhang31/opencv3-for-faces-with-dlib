#include "faceBlendCommon.hpp"
#include <opencv2/highgui/highgui.hpp>


/**
 * @brief trackBar回调的携带的数据类型
 * 
 */
struct morphDatas{
    string winName; //window窗口名称
    cv::Mat imgNorm1; //归一化的图像1
    cv::Mat imgNorm2; //归一化的图像2
    std::vector<cv::Point2f> points1;   //归一化的特征点1
    std::vector<cv::Point2f> points2;   //归一化的特征点2
    std::vector<std::vector<int>> delaunayTri; //剖分后的三角形

    morphDatas(string _winName, cv::Mat& _imgNorm1, cv::Mat& _imgNorm2, 
                std::vector<cv::Point2f>& _points1, std::vector<cv::Point2f>& _points2,
                std::vector<std::vector<int>> _delaunayTri):
                winName(_winName), imgNorm1(_imgNorm1), imgNorm2(_imgNorm2),
                points1(_points1), points2(_points2), delaunayTri(_delaunayTri){}
};

/**
 * @brief trackbar回调参数处理
 * 
 * @param bar_val 当前的参数
 * @param userdata 携带的参数
 */
void trackerBarHandler(int bar_val, void* userdata)
{
    std::cout<<"bar_value:"<<bar_val<<std::endl;
    morphDatas data=*(morphDatas*)(userdata);

    double alpha =(double)bar_val/10.0;
    std::vector<cv::Point2f> points;
    for(int i=0;i<data.points1.size(); i++)
    {
        cv::Point2f pointMorph = (1-alpha)*data.points1[i]+alpha*data.points2[i];
        points.push_back(pointMorph);
    }

    cv::Mat imgOut1, imgOut2;
    warpImage(data.imgNorm1, imgOut1, data.points1, points, data.delaunayTri);
    warpImage(data.imgNorm2, imgOut2, data.points2, points, data.delaunayTri);

    cv::Mat imgMorph = (1-alpha)*imgOut1+alpha*imgOut2;

    std::vector<cv::Mat> imgIns;
    imgIns.push_back(data.imgNorm1);
    imgIns.push_back(data.imgNorm2);
    imgIns.push_back(imgMorph);

    cv::hconcat(imgIns, imgMorph);

    cv::imshow(data.winName, imgMorph);


}

int main(int argc, char const *argv[])
{
    // 1. 设置faceDetector和Shape Detector
    dlib::frontal_face_detector faceDetector = dlib::get_frontal_face_detector();
    dlib::shape_predictor landmarkDetector;
    dlib::deserialize("../../common/resources/zxm_shape_predictor_70_face_landmarks.dat") >> landmarkDetector;

    // 2. 加载混合图片
    cv::Mat img1 = cv::imread("../data/images/girls/hexiaoping.jpg");
    cv::Mat img2 = cv::imread("../data/images/girls/xiaohuizi.jpg");

    // 3. 检测图片中的关键特征点
    std::vector<cv::Point2f> points1 = getLandmarks(faceDetector, landmarkDetector, img1);
    std::vector<cv::Point2f> points2 = getLandmarks(faceDetector, landmarkDetector, img2);

    // 4. 转换图片的数据类型
    img1.convertTo(img1, CV_32FC3, 1.0/255);
    img2.convertTo(img2, CV_32FC3, 1.0/255);

    // 5. 定义输出图片的尺寸
    cv::Size size(300, 300);

    // 6. 调整图片的坐标系
    cv::Mat imgNorm1, imgNorm2;
    normalizeImagesAndLandmarks(size, img1, imgNorm1, points1, points1);
    normalizeImagesAndLandmarks(size, img2, imgNorm2, points2, points2);

    // 7. 计算关键特征点的平均位置，并用于Delaunay三角剖分
    std::vector<cv::Point2f> pointsAvg;
    for(int i=0;i<points1.size();i++)
    {
        pointsAvg.push_back((points1[i]+points2[i])/2.0);
    }

    // 8. 添加8个边缘点
    std::vector<cv::Point2f> boundaryPts;
    getEightBoundaryPoints(size, boundaryPts);
    for(int i=0; i<boundaryPts.size();i++)
    {
        pointsAvg.push_back(boundaryPts[i]);
        points1.push_back(boundaryPts[i]);
        points2.push_back(boundaryPts[i]);
    }

    // 9. 计算Delaunay三角形
    std::vector<vector<int>> delaunayTri;
    calculateDelaunayTriangles(cv::Rect(0,0,size.width,size.height), pointsAvg, delaunayTri);

    // 10. 创建混合图像

    // double alpha =0.5;
    // std::vector<cv::Point2f> points;
    // for(int i=0;i<points1.size(); i++)
    // {
    //     cv::Point2f pointMorph = (1-alpha)*points1[i]+alpha*points2[i];
    //     points.push_back(pointMorph);
    // }

    // cv::Mat imgOut1, imgOut2;
    // warpImage(imgNorm1, imgOut1, points1, points, delaunayTri);
    // warpImage(imgNorm2, imgOut2, points2, points, delaunayTri);

    // cv::Mat imgMorph = (1-alpha)*imgOut1+alpha*imgOut2;

    // std::vector<cv::Mat> imgIns;
    // imgIns.push_back(imgNorm1);
    // imgIns.push_back(imgNorm2);
    // imgIns.push_back(imgMorph);

    // cv::hconcat(imgIns, imgMorph);

    // cv::imshow("imgMorph", imgMorph);

    // 11. 替换10，使用滑竿通知alpha的值

    string winName="imgMorph";
    cv::namedWindow(winName, cv::WINDOW_AUTOSIZE);
    int value =1;
    morphDatas datas(winName, imgNorm1, imgNorm2, points1, points2, delaunayTri);
    createTrackbar("alpha", winName, &value, 10, trackerBarHandler, &datas);
    cv::waitKey(0);


    return 0;
}

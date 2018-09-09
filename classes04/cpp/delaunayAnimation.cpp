/**
 * @brief 使用动画展示Delaunay三角剖分和Voronoi的划分的过程
 * 
 * @file delaunayAnimation.cpp
 * @author Shawn Zhang
 * @date 2018-09-09
 */
#include <dlib/opencv.h>
#include <dlib/image_processing.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;
using namespace dlib;

/**
 * @brief 在图像上绘制特征点
 * 
 * @param img 绘制的图像
 * @param fp 特征点的坐标
 * @param color 绘制使用的颜色
 */
static void drawPoint(Mat &img, Point2f fp, Scalar color)
{
    cv::circle(img, fp, 2, color);
}

/**
 * @brief 在图像上绘制Delaunay三角形
 * 
 * @param img 要绘制的图形
 * @param subdiv 划分类
 * @param delaunayColor 绘制使用的颜色 
 */
static void drawDelaunay(Mat &img, Subdiv2D &subdiv, Scalar delaunayColor)
{
    
    // 获取三角形列表
    std::vector<cv::Vec6f> triangleList;
    subdiv.getTriangleList(triangleList);

    std::vector<Point> vertices(3);
    Size size = img.size();
    Rect rect(0, 0, size.width, size.height);

    for (size_t i = 0; i < triangleList.size(); i++)
    {
        Vec6f t = triangleList[i];

        // 转三角形为列表
        vertices[0] = Point(cvRound(t[0]), cvRound(t[1]));
        vertices[1] = Point(cvRound(t[2]), cvRound(t[3]));
        vertices[2] = Point(cvRound(t[4]), cvRound(t[5]));


        // 绘制三角形
        if (rect.contains(vertices[0]) && rect.contains(vertices[1]) && rect.contains(vertices[2]))
        {
            cv::line(img, vertices[0], vertices[1], delaunayColor);
            cv::line(img, vertices[1], vertices[2], delaunayColor);
            cv::line(img, vertices[2], vertices[0], delaunayColor);
        }
    }
}

/**
 * @brief 绘制沃罗诺伊图
 * 
 * @param img 要绘制的图像
 * @param subdiv 划分器
 */
static void drawVoronoi(Mat &img, Subdiv2D &subdiv)
{
    // voronoi面片列表
    std::vector<std::vector<Point2f>> facets;

    // voronoi面片的中心点
    std::vector<Point2f> centers;

    // 获取voronoi图的面片列表和中心点
    subdiv.getVoronoiFacetList(std::vector<int>(), facets, centers);

    // 使用fillConvexPoly方法绘制面片
    std::vector<Point> ifacet;
    // 使用polylines方法绘制面片的边缘
    std::vector<std::vector<Point>> ifaces(1);
    for(size_t i=0; i<facets.size(); i++)
    {
        ifacet.resize(facets[i].size());
        for(size_t k=0; k<facets[i].size(); k++)
        {
            ifacet[k]=facets[i][k];
        }

        // 生成随机颜色
        Scalar color;
        color[0] = std::rand() & 255; // 这是一个小诀窍，因为255的二进制位11111111，所以去与计算，最大值只能是255
        color[1] = std::rand() & 255;
        color[2] = std::rand() & 255;

        // 填充面片
        cv::fillConvexPoly(img, ifacet, color);

        // 绘制面片的边界
        ifaces[0] = ifacet;
        cv::polylines(img, ifaces, true, Scalar());

        // 绘制中心点
        cv::circle(img, centers[i], 3, Scalar());
    }
}

/**
 * @brief Get the Face Landmarks object
 * 
 * @param img 
 * @return std::vector<Point> 
 */
std::vector<Point2f> getFaceLandmarks(Mat& img)
{
    // 面部检测器
    dlib::frontal_face_detector faceDetector = get_frontal_face_detector();
    // 形状检测器
    dlib::shape_predictor landmarkPredictor;

     // 加载面部关键特征点模型
    deserialize("../../common/resources/zxm_shape_predictor_70_face_landmarks.dat") >> landmarkPredictor;

    // 将OpenCV的图像格式转换为dlib的图像格式
    cv_image<bgr_pixel> dlibImg(img);
    std::vector<dlib::rectangle> faceRects=faceDetector(dlibImg);
    
    dlib::full_object_detection landmarks= landmarkPredictor(dlibImg, faceRects[0]);

    std::vector<Point2f> points;
    for(int i=0; i<landmarks.num_parts();i++)
    {
        points.push_back(Point2f(landmarks.part(i).x(), landmarks.part(i).y()));
        cout<<"特征点"<<i<<":"<<points[i]<<endl;
    }

    

    return points;

}

int main(int argc, char const *argv[])
{
    string win="Delaunay Animation";
    Scalar delaunayColor(255, 255, 255), pointsColor(0, 0, 255);
    Mat img = imread("../data/images/smiling-man.jpg");

    // 初始化一个划分器
    Rect rect(1,1,img.size().width, img.size().height);
    Subdiv2D subdiv(rect);

    // 获取关键点点
    std::vector<Point2f> points = getFaceLandmarks(img);

    // 显示三角剖分的图像
    Mat imgDelaunay;
    // 显示Voronoi图的图像
    Mat imgVoronoi = Mat::zeros(img.rows, img.cols, CV_8UC3);

    // 最终显示的图像
    Mat imgDisplay;

    // 在划分器中插入点进行划分
    for(std::vector<Point2f>::iterator it = points.begin();it != points.end(); it++)
    {
        subdiv.insert(*it);

        imgDelaunay = img.clone();
        imgVoronoi = cv::Scalar(0,0,0);

        // 绘制delaunay
        drawDelaunay(imgDelaunay, subdiv, delaunayColor);

        // 绘制点
        for(std::vector<Point2f>::iterator ptr=points.begin(); ptr != points.end(); ptr++ )
        {
            drawPoint(imgDelaunay, *ptr, pointsColor);
        }

        // 绘制Voroni
        drawVoronoi(imgVoronoi, subdiv);
        cv::hconcat(imgDelaunay, imgVoronoi, imgDisplay);
        imshow(win, imgDisplay);
        waitKey(400);
    }


    waitKey(0);

    return 0;
}


#include <dlib/image_processing.h>
#include <dlib/data_io.h>
#include <iostream>

using namespace dlib;
using namespace std;

/**
 * @brief 
 * 
 * @param det 
 * @return double int 
 */
double interocular_distance(const dlib::full_object_detection& det)
{
    dlib::vector<double,2> l, r;
    double cnt = 0;

    for(unsigned long i=36;i<=41;i++)
    {
        l += det.part(i);
        ++cnt;
    }
    l /=cnt;

    cnt = 0;
    for(unsigned long i=42;i<=47;i++)
    {
        r += det.part(i);
        ++cnt;
    }

    r /=cnt;

    return length(l-r);
}

/**
 * @brief Get the interocular distances object
 * 
 * @param objetcs 
 * @return std::vector<std::vector<double> > 
 */
std::vector<std::vector<double> > get_interocular_distances(const std::vector<std::vector<dlib::full_object_detection> >& objetcs)
{
    std::vector<std::vector<double> > temp(objetcs.size());
    for(unsigned long i=0;i<objetcs.size();i++)
    {
        for(unsigned long j=0;j<objetcs[i].size();++j)
        {
            temp[i].push_back(interocular_distance(objetcs[i][j]));
        }
    }

    return temp;
}

int main(int argc, char** argv)
{
    // 从穿入参数中读取facial_landmark_data数据
    try
    {
        if(argc != 3)
        {
            std::cout << "请传入facial_landmark_data数据的路径"<<std::endl;
            return 0;
        }
        const std::string fldDatadir = argv[1];
        const std::string numPoints = argv[2];
        const std::string modelName = "zxm_shape_predictor_"+numPoints+"_face_landmarks.dat";
        const std::string modelPath = fldDatadir + "/" + modelName;

        // 创建shape_predictor_trainer对象并设置参数
        dlib::shape_predictor_trainer trainer;

        // 设置线程数量
        trainer.set_num_threads(2);

        // 设置级联数
        trainer.set_cascade_depth(10);
        // 设置每层决策树的数量
        trainer.set_num_trees_per_cascade_level(500);
        // 设置每个决策树的层数
        trainer.set_tree_depth(4);
        trainer.set_nu(0.1);
        trainer.set_oversampling_amount(20);
        trainer.set_feature_pool_size(400);
        trainer.set_feature_pool_region_padding(0);
        trainer.set_lambda(0.1);
        trainer.set_num_test_splits(20);

        // 估算训练量
        std::cout<<"估算训练时间"<<std::endl;
        trainer.be_verbose();

        dlib::array<array2d<unsigned char> > images_train, images_test;
        std::vector<std::vector<dlib::full_object_detection> > faces_train, faces_test;

        dlib::load_image_dataset(images_train, faces_train, fldDatadir + "/zxm_training_with_face_landmarks.xml");
        dlib::load_image_dataset(images_test, faces_test, fldDatadir + "/zxm_testing_with_face_landmarks.xml");

        // 生成训练模型
        dlib::shape_predictor sp = trainer.train(images_train, faces_train);

        
        // 使用训练的模型检测数据
        std::cout<<"在训练集上的平均误差:"<< dlib::test_shape_predictor(sp,images_train,faces_train,get_interocular_distances(faces_train)) <<std::endl;
        // 使用测试集检查平均误差
        std::cout<<"在测试集上的平均误差:"<< dlib::test_shape_predictor(sp,images_test, faces_test,get_interocular_distances(faces_test)) <<std::endl;

        // 保存模型
        serialize(modelPath) << sp;
    }
    catch(exception& e)
    {
        std::cout<<"\n发生错误!"<<std::endl;
        std::cout<<e.what()<<std::endl;
    }
}
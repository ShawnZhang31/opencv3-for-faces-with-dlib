import os
import sys
import dlib

# 传入参数
fldDatadir = sys.argv[1]
numPoints = sys.argv[2]
modelName = "zxm_python_shape_predictor_" + numPoints + "_face_landmarks.dat"

# 设置shape_predictor_trainer的参数
options = dlib.shape_predictor_training_options()
options.cascade_depth = 10
options.num_trees_per_cascade_level = 500
options.tree_depth = 4
options.nu = 0.1
options.oversampling_amount = 20
options.feature_pool_size = 400
options.feature_pool_region_padding = 0
options.lambda_param = 0.1
options.num_test_splits = 20

# 显示训练的状态
options.be_verbose = True
trainingXmlPath = os.path.join(fldDatadir, "zxm_training_with_face_landmarks.xml")
testingXmlPath = os.path.join(fldDatadir, "zxm_testing_with_face_landmarks.xml")
outputModelPath = os.path.join(fldDatadir, modelName)

# 检查XML的路径是否正确
if os.path.exists(trainingXmlPath) and os.path.exists(testingXmlPath):
    dlib.train_shape_predictor(trainingXmlPath, outputModelPath, options)
    print("\n训练集的精度:{}".format(
        dlib.test_shape_predictor(trainingXmlPath, outputModelPath)))
    
    print("\n在测试集上的精度:{}".format(
        dlib.test_shape_predictor(testingXmlPath, outputModelPath)))
else:
    print("训练集和测试集的xml文件未能找到!")
    print("请检查路径:\n")
    print("train:{}".format(trainingXmlPath))
    print("test:{}".format(testingXmlPath))
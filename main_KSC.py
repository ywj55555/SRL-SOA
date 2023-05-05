import os
# os.environ["CUDA_VISIBLE_DEVICES"]="0"

import numpy as np
import tensorflow as tf
import argparse
from sklearn.preprocessing import StandardScaler
# import svm
import utils
from sklearn.model_selection import train_test_split
np.random.seed(10)
tf.random.set_seed(10)

ap = argparse.ArgumentParser()
ap.add_argument('--method', default='SRL-SOA', help =
                "SRL-SOA, PCA, SpaBS, EGCSR_R, ISSC, None (for no band selection).")
ap.add_argument('--dataset', default='Indian_pines_corrected', help = "Indian_pines_corrected, SalinasA_corrected.")
ap.add_argument('--q', default = 3, help = "Order of the OSEN.")
ap.add_argument('--weights', default = False, help="Evaluate the model.")
ap.add_argument('--epochs', default = 50, help="Number of epochs.")
ap.add_argument('--batchSize', default = 176, help="Batch size.")  # 原始为5 得减少点数据量了！！！
ap.add_argument('--bands', default = 1, help="Compression rate.")
args = vars(ap.parse_args())

param = {}

param['modelType'] = args['method']
# param['weights'] = args['weights'] # True or False.
param['weights'] = False
param['q'] = int(args['q']) # The order of the OSEN.
param['dataset'] = args['dataset'] # Dataset.
param['epochs'] = int(args['epochs'])
param['batchSize'] = int(args['batchSize'])
param['s_bands'] = int(args['bands']) # Number of bands.
parameterSearch = True # Parameter search for the classifier.
# 完全可以不要 Data
# classData, Data = utils.loadData(param['dataset'])

# y_predict = []
# classDataAll, DataAll = utils.loadData()
# 我的数据集
# traindatanpy = '/home/cjl/ywj_code/graduationCode/BS-NETs/trainData/128bandsFalse_60_mulprocess.npy'
# # trainlabelnpy = '/home/cjl/ywj_code/graduationCode/BS-NETs/trainData/128bandsFalse_60_mulprocess_label.npy'
# classDataAll = np.load(traindatanpy)
# classDataAll = classDataAll[::2, :, 5, 5]  # B C H W 必须是 ::2，数据量太大会出问题！！！
# print('max', np.max(classDataAll))
# print('min', np.min(classDataAll))
# scaler = StandardScaler().fit(classDataAll)
# classDataAll = scaler.transform(classDataAll)
# print('max', np.max(classDataAll))
# print('min', np.min(classDataAll))
# print(classDataAll.shape)
import scipy.io
print("begin~~")
ksc_path = '/home/cjl/dataset/ksc/'
data_ksc = scipy.io.loadmat(ksc_path + 'KSC.mat')
ksc_spectral_data = data_ksc['KSC']
image = np.array(ksc_spectral_data, dtype=np.float32)  # .transpose(2,0,1)#band, W, H
print(ksc_spectral_data.shape)
data_gt = scipy.io.loadmat(ksc_path + 'KSC_gt.mat')
gtd = data_gt['KSC_gt']
gtd = np.array(gtd, dtype = 'float32')
xx = np.reshape(image, [image.shape[0] * image.shape[1], image.shape[2]])  # H*W C 因为原始算法是基于纯光谱，单个像素的！
label = np.reshape(gtd, [gtd.shape[0] * gtd.shape[1]])  # H*W
x_class = xx[label != 0]
y_class = label[label != 0]

# train_x = train_x.transpose(0, 2, 3, 1)  # BHW C
# train_label = np.load(trainlabelnpy)
# train_y = np.nanargmax(train_label, axis=1)
# Band selection ...
for i in range(0, 20): # 10 runs ...
    # if param['modelType'] != 'None':
    # classData[i], Data[i] = utils.reduce_bands(param, classData[i], Data[i], i)
    classData = {}
    print('range :', i)
    x_train, x_test, y_train, y_test = train_test_split(x_class, y_class,
                                                        test_size=0.95, random_state=i + 1)
    scaler = StandardScaler().fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)
    classData['x_train'] = x_train
    # classData['x_train'] = classDataAll[np.random.choice(classDataAll.shape[0], int(classDataAll.shape[0] * 0.8), replace=False)]

    utils.reduce_bands(param, classData, None, i)
    del classData
    # print('Classification...')
    # if parameterSearch:
    #     # If hyper-parameter search is selected.
    #     best_parameters, class_model = svm.svm_train_search(classData[i]['x_train'], classData[i]['y_train'])
    #     print('\nBest paramters:' + str(best_parameters))
    # else:
    #     class_model = svm.svm_train(classData[i]['x_train'], classData[i]['y_train'])
    #
    # y_predict.append(class_model.predict(classData[i]['x_test']))

# utils.evalPerformance(classData, y_predict)
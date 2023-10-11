import sys
sys.path.append('/home/CVAE-LIME20230802/')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
'''
'0': すべてのログを表示します（デフォルト）。
'1': INFOログを抑制し、WARNINGとERRORログのみ表示します。
'2': WARNINGログも抑制し、ERRORログのみ表示します。
'3': すべてのログを抑制します。
'''
import warnings
# TensorFlowの警告を無視する
warnings.filterwarnings("ignore", category=UserWarning, module="tensorflow")
# Scikit-learnの警告を無視する
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)  # v1 APIのWAR
from functions import *
from main import main
import itertools

encoder = 0 #int(sys.argv[1])
dataset_no = 0 #int(sys.argv[2])
# target = int(sys.argv[2])
target = 0
## 実験条件
dataset = [['breastcancer', 'hepa', 'liver']][dataset_no]#'breastcancer', 'hepa', 'liver', 'adult', 'wine', 'mine', 'boston'
dataset_class_num = {'adult': 2, 'wine': 6, 'boston':'numerous', 'mine': 5, 'hepa': 2, 'breastcancer':2, 'liver':2}
target_model = [['NN']][target] # 'NN', 'RF', 'DNN', 'GBM', 'XGB', 'SVM'
target_model_training = False
test_range = range(40) # hepaのデータ数が少ないため

# 検証手法の条件
auto_encoder_weighting = [True]
auto_encoder_sampling = [True]
auto_encoder = [['CVAE', 'VAE']][encoder]  #'ICVAE2', 'ICVAE', 'CVAE', 'VAE', 'AE', 'LIME'
auto_encoder_training = False
auto_encoder_epochs = 1000
auto_encoder_latent_dim = 2
one_hot_encoding = False
feature_selection = 'auto'
noise_std = 0.1
model_regressor = None # 'ridge'
stability_check = False
# 実装していないが，過去実験との互換用
filtering = False
select_percent = 100

#　実験組み合わせの生成
p = itertools.product(dataset, target_model, auto_encoder_weighting, auto_encoder_sampling, auto_encoder)

for dataset, target_model, auto_encoder_weighting, auto_encoder_sampling, auto_encoder in p:
    print(f'==============dataset:{dataset}_target_model:{target_model}_weighting:{auto_encoder_weighting}_sampling:{auto_encoder_sampling}_AE:{auto_encoder}=============')
    if auto_encoder == 'AE':
        auto_encoder_sampling = False
    if auto_encoder != 'AE':
        auto_encoder_sampling = True
    if dataset == 'hepa':
        test_range = range(10)
    if dataset != 'hepa':
        test_range = range(40)
    main(dataset,
         dataset_class_num,
         target_model,
         target_model_training,
         test_range,
         auto_encoder_weighting,
         auto_encoder_sampling,
         auto_encoder,
         auto_encoder_training,
         auto_encoder_epochs,
         auto_encoder_latent_dim,
         filtering,
         select_percent,
         one_hot_encoding,
         feature_selection,
         noise_std,
         model_regressor,
         stability_check)
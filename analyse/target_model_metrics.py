import pandas as pd
from sklearn.model_selection import train_test_split
from functions import target_model_loder
import sys
sys.path.append('/home/CVAE-LIME20230802/')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)  # v1 APIのWARNING以上のメッセージも非表示にする

for DATA in [1]:
    for target in [0,1,2]:
        dataset = ['breastcancer', 'wine', 'liver', 'adult', 'credit'][DATA] #, 'boston', 'hepa'
        dataset_class_num = {'breastcancer':2,
                            'hepa': 2,
                            'liver':2,
                            'adult': 2,
                            'wine': 6,
                            'credit':2,
                            'boston':'numerous'}
        target_model = ['NN', 'RF', 'SVM'][target] #, 'NN', 'SVM', 'DNN', 'GBM', 'XGB'
        target_model_training =  True
        print(f'=========Data:{dataset}_Target:{target_model}=========')

        data = pd.read_csv(f'dataset/{dataset}.csv')
        # データセットを特徴量とターゲットに分割
        X = data.drop('target', axis=1)
        y = data['target']

        # データセットをトレーニングセットとテストセットに分割
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        ## ターゲットモデルの学習又はロード(predict_probaで返す)
        model = target_model_loder(dataset = dataset,
                                    target_model = target_model,
                                    target_model_training = target_model_training,
                                    X_train = X_train,
                                    y_train = y_train,
                                    dataset_class_num = dataset_class_num,
                                    )
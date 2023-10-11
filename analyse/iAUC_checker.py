import numpy as np
from functions import read_object_from_file
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

import sys
sys.path.append('/home/CVAE-LIME20230802/')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys
AE = int(sys.argv[1])
dataset = int(sys.argv[2])

def process_data(data):
    # 1. X_testと同じ特徴量数の要素がすべて0のベクトルを生成する
    zero_vector = np.zeros(data['X_test'].shape[0])
    model_output = []
    lime_output = []

    # 現存するキーを取得
    label = list(data['exp'].local_exp.keys())[0]
    
    # 最初のキーを使用してimportant_featuresを取得
    important_features = data['exp'].local_exp[label]

    # 線形モデルの係数を取得
    linear_model_coeffs = dict(data['exp'].local_exp[label])
    
    # 線形モデルの切片を取得
    linear_coef = data['exp'].intercept

    lime_val = 0
    
    for feature, _ in important_features:
        # 2. X_testの内，expで重要と判断した特徴量でベクトルの対応する値を置き換える
        zero_vector[feature] = data['X_test'][feature]

        model = data['model']
        
        # 4. 2で生成したベクトルに対するmodelの特徴量を獲得する
        model_val = model(np.array([zero_vector]))[0]
        model_val_max = max(model_val)  # 2クラス分類の1クラス目を仮定
        model_output.append(model_val_max)

        # 5. 2で生成したベクトルに対する線形モデルの特徴量を獲得する
        # lime_val  = zero_vector * linear_model_coeffs[feature] if feature in linear_model_coeffs else 0
        x = data['X_test'][feature]
        lime_val += linear_model_coeffs[feature] * x            
        lime_val2 = lime_val + sum(linear_coef.values())
        lime_output.append(lime_val2)

    # ６．X_testの内，expで２番目に重要と判断した特徴量で２で生成したベクトルの対応する値を置き換える
    # はループの中で処理されているので省略。

    # 4と5で得られた値同士の相関係数を出す
    correlation_coefficient, _ = pearsonr(model_output, lime_output)
    
    # 4と5で得られた値同士の差を出す
    output_difference = np.mean(np.array(model_output) - np.array(lime_output))
    print(f'label:{label}')
    return correlation_coefficient, abs(output_difference)


dataset = ['breastcancer', 'hepa', 'liver', 'adult', 'wine', 'mine', 'boston'][dataset]
auto_encoder = ['CVAE', 'VAE', 'DAE', 'CVAE', 'AE', 'LIME', 'ICVAE', 'ICVAE2'][AE]

filename = str(dataset)+'_'+str(auto_encoder)

# dataのロード
datas = read_object_from_file('save_data/test_iAUC/' + filename + '.dill')

corr_mean = []
output_differences = []

for i in range(15):
    # 使用例
    correlation, output_difference = process_data(datas[i])
    print("Correlation Coefficient:", correlation)
    print("Output difference:", output_difference)
    corr_mean.append(correlation)
    output_differences.append(output_difference) 
    
print("Coefficient Mean:", np.mean(corr_mean))
print("Different Mean:", np.mean(np.array(output_differences)))


























# Don't be fooled
# iAUC_values = []

# for data_instance in data:
#     e_vector = np.zeros(len(data_instance['X_test'].columns))
    
#     for feature, value in data_instance['exp'].as_list():
#         feature_idx = data_instance['X_test'].columns.get_loc(feature)
#         e_vector[feature_idx] = value

#     likelihoods = []
    
#     for n in range(101): # 0から100まで
#         top_features = topn(e_vector, n)
#         modified_input = data_instance['X_test'].copy()
        
#         # トップn%以外の特徴量を0に設定（この部分は問題によって異なるかもしれません）
#         for idx, col in enumerate(modified_input.columns):
#             if idx not in top_features:
#                 modified_input[col] = 0
        
#         model = data_instance['model']
#         probabilities = model.predict(modified_input)
#         likelihood = log_likelihood(probabilities, true_label) # true_labelを適切に設定する必要があります
#         likelihoods.append(likelihood)
    
#     iAUC = np.mean(likelihoods)
#     iAUC_values.append(iAUC)

# print(np.mean(iAUC_values))

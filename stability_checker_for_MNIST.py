import sys
sys.path.append('/home/CVAE-LIME20230802/')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
'''
'0': すべてのログを表示します（デフォルト）。
'1': INFOログを抑制し、WARNINGとERRORログのみ表示します。
'2': WARNINGログも抑制し、ERRORログのみ表示します。
'3': すべてのログを抑制します。
'''
from functions import *
from main_for_MNIST import main
import itertools
from functions import calculate_jaccard_for_all_combinations, append_to_csv
import statistics

i = int(sys.argv[1])
j = 5 #int(sys.argv[1]) ####
k = int(sys.argv[2])

## 実験条件
conditions = [[['CVAE'],[0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], [0],[True], [True] ],
              [['VAE'] ,[0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], [0],[True], [True] ],
              [['AE']  ,[0], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],[True], [False]],
              [['LIME'],[0], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],[False],[False]],
              ][i]
# [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
# dataset = condition[0]
# target_model = condition[1]
auto_encoder = conditions[0]
noise_std = conditions[1]
kernel_width = conditions[2]
auto_encoder_weighting = conditions[3]
auto_encoder_sampling = conditions[4]

## 実験条件パターン
dataset = [['breastcancer', 'liver', 'wine', 'credit', 'adult', 'MNIST'][j]] #'breastcancer', 'hepa', 'liver', 'wine', 'boston' ,'adult' , 'boston'
dataset_class_num = {'adult': 2, 'wine': 6, 'boston':'numerous', 'mine': 5, 'hepa': 2, 'breastcancer':2, 'liver':2, 'credit':2, 'MNIST':10}
target_model = [['NN', 'RF', 'SVM', 'DNN', 'GBM', 'XGB','CNN'][k]]
target_model_training = False
num_samples = 5000
# test_range = range(40) 

# 検証手法の条件
# auto_encoder_weighting = [True]
# auto_encoder_sampling = [True]
# auto_encoder = ['ICVAE2', 'ICVAE', 'CVAE', 'VAE', 'AE', 'LIME']
auto_encoder_training =  False
auto_encoder_epochs = 100 #1000 MNSITは100
auto_encoder_latent_dim = 2
one_hot_encoding = False
feature_selection = 'auto'
noise_std = noise_std
model_regressor = None # 'ridge'ということ

# 安定性評価
stability_check = True
repeat_num = 1
instance_no = range(100)
kernel_width = kernel_width

# 実装していないが，過去実験との互換用
label_filtering = False
select_percent = 100

preprocess = 'Minimax'

#　実験組み合わせの生成
p = itertools.product(dataset, target_model, auto_encoder_weighting, auto_encoder_sampling, auto_encoder, instance_no, noise_std, kernel_width)

# ... [以前のコードはそのまま]

for dataset, target_model, auto_encoder_weighting, auto_encoder_sampling, auto_encoder, instance_no, noise_std, kernel_width in p:
    print(f'==============dataset:{dataset}_targetmodel:{target_model}_weighting:{auto_encoder_weighting}_sampling:{auto_encoder_sampling}_AE:{auto_encoder}_std:{noise_std}_kernel:{kernel_width}=============')
    if auto_encoder == 'AE':
        auto_encoder_sampling = False
    if auto_encoder != 'AE':
        auto_encoder_sampling = True

    features_from_lime_runs = []
    features_from_lime_runs
    
    # hepaはインスタンス数が15
    if dataset == 'hepa':
        test_range = range(15)
    
    for test_range in [[instance_no] for _ in range(repeat_num)]:
        try:
            feature_list, score, mse, predict_label, label = main(
                dataset,
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
                label_filtering,
                select_percent,
                one_hot_encoding,
                feature_selection,
                noise_std,
                model_regressor,
                stability_check,
                kernel_width,
                num_samples,
                preprocess)
    
            features_from_lime_runs.append([item[0] for item in feature_list])
            append_to_csv(f'save_data/test_stability/{dataset}{target_model}{auto_encoder_weighting}{auto_encoder_sampling}{auto_encoder}{instance_no}.csv',
                      [dataset, target_model, auto_encoder_weighting, auto_encoder_sampling, auto_encoder, instance_no, noise_std, kernel_width, None, score, mse, predict_label, label],
                      ['dataset', 'target_model', 'auto_encoder_weighting', 'auto_encoder_sampling', 'auto_encoder', 'instance_no', 'noise_std', 'kernel_width', 'jaccard_values', 'score', 'mse','predict_label','label'],
                      )
        except Exception as e:
            print(f"Error occurred with dataset:{dataset} and target_model:{target_model}. Skipping. Error: {e}")
            predict_label = ""
            label = ""
            
    if repeat_num != 1:
        jaccard_values = calculate_jaccard_for_all_combinations(features_from_lime_runs)
        jaccard_values_mean = statistics.mean(jaccard_values)
        jaccard_values_mean = f'{jaccard_values_mean:.3f}'
        print('jaccard_value',jaccard_values_mean)
    else:
        jaccard_values_mean = 0
    
    append_to_csv(f'save_data/test_stability/{dataset}{target_model}{auto_encoder_weighting}{auto_encoder_sampling}{auto_encoder}{instance_no}.csv',
                [dataset, target_model, auto_encoder_weighting, auto_encoder_sampling, auto_encoder, instance_no, noise_std, kernel_width, jaccard_values_mean, None, None, predict_label, label],
                ['dataset', 'target_model', 'auto_encoder_weighting', 'auto_encoder_sampling', 'auto_encoder', 'instance_no', 'noise_std', 'kernel_width', 'jaccard_values', 'R2', 'mse','predict_label','label'],
                )


    

    
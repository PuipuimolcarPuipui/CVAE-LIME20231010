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
from main import main
import itertools
from functions import calculate_jaccard_for_all_combinations, append_to_csv
import statistics

LIMEs = 0 #int(sys.argv[1])
DATA = 0 #int(sys.argv[2])
Target = 0 #int(sys.argv[3])

## 実験条件
conditions = [[['CVAE'],[1.0], [0],[True], [True] ], # 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
              [['VAE'] ,[1.0], [0],[True], [True] ],
              [['AE']  ,[0], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],[True], [False]],
              [['LIME'],[0], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],[False],[False]],
              ][LIMEs]

# dataset = condition[0]
# target_model = condition[1]
auto_encoder = conditions[0]
noise_std = conditions[1]
kernel_width = conditions[2]
auto_encoder_weighting = conditions[3]
auto_encoder_sampling = conditions[4]

## 実験条件パターン
dataset = [['breastcancer',
            'credit_one_hot',
            'adult_one_hot',
            'liver',
            'wine',
            'credit',
            'adult',
            'boston'][DATA]] #'breastcancer', 'hepa', 'liver', 'wine', 'boston' ,'adult' , 'boston'
dataset_class_num = {'adult': 2, 'wine': 6, 'boston':'numerous', 'mine': 5, 'hepa': 2, 'breastcancer':2, 'liver':2, 'credit':2, 'wine3':3,'credit_one_hot':2,'adult_one_hot':2}
target_model = [['NN', 'RF', 'SVM', 'DNN', 'GBM', 'XGB'][Target]]
target_model_training = False
num_samples = 5000
# test_range = range(40) 

# 検証手法の条件
# auto_encoder_weighting = [True]
# auto_encoder_sampling = [True]
# auto_encoder = ['ICVAE2', 'ICVAE', 'CVAE', 'VAE', 'AE', 'LIME']
auto_encoder_training =  False
auto_encoder_epochs = 1000
feature_selection = 'auto'
noise_std = noise_std
model_regressor = None # 'ridge'ということ
auto_encoder_latent_dim = {'breastcancer':[2,4,6,8,10,12,14],
                           'credit_one_hot':[2,4,6,8,10,12,14],
                           'adult_one_hot':[2,4,6,8,10,12,14],
                           'liver':[2,4,6],
                           'wine':[2,4,6],
                           }[dataset[0]]

# 条件ベクトルのone-hotエンコード
if dataset == 'wine':
    one_hot_encoding = True
else:
    one_hot_encoding = False

# 安定性評価
stability_check = True
repeat_num = 1
instance_no = range(1)
kernel_width = kernel_width

# 実装していないが，過去実験との互換用
label_filtering = False
select_percent = 100

#活性潜在変数か否かの分散ベクトルの閾値
var_threshold = 0.5

#　実験組み合わせの生成
p = itertools.product(dataset, target_model, auto_encoder_weighting, auto_encoder_sampling, auto_encoder, instance_no, noise_std, kernel_width, auto_encoder_latent_dim)

# ... [以前のコードはそのまま]

for dataset, target_model, auto_encoder_weighting, auto_encoder_sampling, auto_encoder, instance_no, noise_std, kernel_width, auto_encoder_latent_dim in p:
    print(f'==============dataset:{dataset}_targetmodel:{target_model}_weighting:{auto_encoder_weighting}_sampling:{auto_encoder_sampling}_AE:{auto_encoder}_std:{noise_std}_kernel:{kernel_width}_Dim:{auto_encoder_latent_dim}=============')
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
            feature_list, score, mse, predict_label, label, L1, L2, Active_latent_dim = main(
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
                var_threshold)
    
            features_from_lime_runs.append([item[0] for item in feature_list])
            append_to_csv(f'save_data/test_stability/{dataset}{target_model}{auto_encoder_weighting}{auto_encoder_sampling}{auto_encoder}{instance_no}.csv',
                      [dataset, target_model, auto_encoder_weighting, auto_encoder_sampling, auto_encoder, instance_no, noise_std, kernel_width, None, score, mse, predict_label, label, L1, L2, Active_latent_dim],
                      ['dataset', 'target_model', 'auto_encoder_weighting', 'auto_encoder_sampling', 'auto_encoder', 'instance_no', 'noise_std', 'kernel_width', 'jaccard_values', 'score', 'mse','predict_label','label', 'L1', 'L2','Active_latent_dim'],
                      )
        except Exception as e:
            print(f"Error occurred with dataset:{dataset} and target_model:{target_model}. Skipping. Error: {e}")
            predict_label = ""
            label = ""
            L1 = ""
            L2 = ""
            Active_latent_dim = ""
            
    if repeat_num != 1:
        jaccard_values = calculate_jaccard_for_all_combinations(features_from_lime_runs)
        jaccard_values_mean = statistics.mean(jaccard_values)
        jaccard_values_mean = f'{jaccard_values_mean:.3f}'
        print('jaccard_value',jaccard_values_mean)
    else:
        jaccard_values_mean = 0
    
    append_to_csv(f'save_data/test_stability/{dataset}{target_model}{auto_encoder_weighting}{auto_encoder_sampling}{auto_encoder}{instance_no}.csv',
                [dataset, target_model, auto_encoder_weighting, auto_encoder_sampling, auto_encoder, instance_no, noise_std, kernel_width, jaccard_values_mean, None, None, predict_label, label, L1, L2],
                ['dataset', 'target_model', 'auto_encoder_weighting', 'auto_encoder_sampling', 'auto_encoder', 'instance_no', 'noise_std', 'kernel_width', 'jaccard_values', 'R2', 'mse','predict_label','label', 'L1', 'L2','Active_latent_dim'],
                )


    

    
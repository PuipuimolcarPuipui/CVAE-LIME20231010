# import pandas as pd
# from scipy import stats
# import itertools

# # CSVファイルの読み込み
# file_path = '../save_data/result_WRSR.csv'
# data = pd.read_csv(file_path)


# value = ['mse', 'Active_latent_dim', 'L1', 'WRSR']
# latent_dim = [6]
# dataset = ['breastcancer', 'liver', 'wine', 'credit_one_hot', 'adult_one_hot']
# target = ['NN', 'RF', 'SVM']

# p = itertools.product(value, dataset, target, latent_dim)

# for value, dataset, target, latent_dim in p:
#     # 条件に一致する行の抽出
#     filtered_data = data[
#         (data['dataset'] == dataset) & 
#         # (data['noise_std'] == 1) &
#         (data['auto_encoder_latent_dim'] == latent_dim)
#     ]

#     # 'auto_encoder'列が'VAE'または'CVAE'である行を抽出
#     vae_data = filtered_data[(filtered_data['weighting_fn'] == 'VAE') & (filtered_data['target_model'] == target)][value]
#     cvae_data = filtered_data[(filtered_data['weighting_fn'] == 'CVAE') & (filtered_data['target_model'] == target)][value]
#     lime_data = filtered_data[(filtered_data['weighting_fn'] == 'LIME') & (filtered_data['target_model'] == target)][value]

#     # 独立 2 標本 t検定
#     t_stat, p_value = stats.ttest_ind(vae_data, cvae_data)
#     if p_value < 0.05:
#         p = 'o'
#     else:
#         p = 'x'

#     # 結果の表示
#     print(f'対VAE,{p}, Dim{latent_dim}, dataset{dataset}, target{target}, t統計量: {t_stat}, p値: {p_value}, Value:{value}')
    
#     # 独立 2 標本 t検定
#     t_stat, p_value = stats.ttest_ind(lime_data, cvae_data)
#     if p_value < 0.05:
#         p = 'o'
#     else:
#         p = 'x'

#     # 結果の表示
#     print(f'対LIME,{p}, Dim{latent_dim}, dataset{dataset}, target{target}, t統計量: {t_stat}, p値: {p_value}, Value:{value}')


import pandas as pd
from scipy import stats
import itertools

# CSVファイルの読み込み
file_path = '../save_data/result_spearman.csv'
data = pd.read_csv(file_path)

value = ['mse', 'Active_latent_dim', 'L1', 'spearman','r2','RSS','TSS']
latent_dim = [6]
dataset = ['breastcancer', 'liver', 'wine', 'credit_one_hot', 'adult_one_hot']
target = ['NN', 'RF', 'SVM']

p = itertools.product(value, dataset, target, latent_dim)

# 検定方法
Metrics = 'U' #'t' 'U'

# 結果を保存するための空のDataFrameを作成
results = []

for value, dataset, target, latent_dim in p:
    filtered_data = data[
        (data['dataset'] == dataset) & 
        (data['auto_encoder_latent_dim'] == latent_dim)
    ]

    vae_data = filtered_data[(filtered_data['weighting_fn'] == 'VAE') & (filtered_data['target_model'] == target)][value]
    cvae_data = filtered_data[(filtered_data['weighting_fn'] == 'CVAE') & (filtered_data['target_model'] == target)][value]
    lime_data = filtered_data[(filtered_data['weighting_fn'] == 'LIME') & (filtered_data['target_model'] == target)][value]

    # VAE vs CVAE
    if Metrics == 'U':
        u_stat, p_value = stats.mannwhitneyu(vae_data, cvae_data, alternative='two-sided')
    elif Metrics == 't':
        u_stat, p_value = stats.ttest_ind(vae_data, cvae_data, equal_var=False)
        
    signif = 1 if p_value < 0.05 else 0
    results.append({'Comparison': 'VAE_vs_CVAE', 'Significance': signif, 'LatentDim': latent_dim, 
                    'Dataset': dataset, 'TargetModel': target, 'TStatistic': u_stat, 'PValue': p_value, 'Value': value})

    # LIME vs CVAE
    u_stat, p_value = stats.mannwhitneyu(lime_data, cvae_data, alternative='two-sided')
    signif = 1 if p_value < 0.05 else 0
    results.append({'Comparison': 'LIME_vs_CVAE', 'Significance': signif, 'LatentDim': latent_dim, 
                    'Dataset': dataset, 'TargetModel': target, 'TStatistic': u_stat, 'PValue': p_value, 'Value': value})

# # 結果をDataFrameに変換
# results_df = pd.DataFrame(results)

# # 結果をCSVファイルに保存
# output_file = '../save_data/analysis_results.csv'
# results_df.to_csv(output_file, index=False)

# print(f"Results saved to {output_file}")




import pandas as pd
from scipy import stats
import itertools

# CSVファイルの読み込み
file_path = '../save_data/result_spearman.csv'
data = pd.read_csv(file_path)

# value = ['mse', 'Active_latent_dim', 'L1', 'WRSR','r2']
value = ['mse', 'Active_latent_dim', 'L1', 'spearman','r2','RSS','TSS']
latent_dim = [10]
dataset = ['MNIST']
target = ['NN', 'RF', 'SVM']

p = itertools.product(value, dataset, target, latent_dim)

# 結果を保存するための空のDataFrameを作成
# results = []

for value, dataset, target, latent_dim in p:
    filtered_data = data[
        (data['dataset'] == dataset) & 
        (data['auto_encoder_latent_dim'] == latent_dim)
    ]

    vae_data = filtered_data[(filtered_data['weighting_fn'] == 'VAE') & (filtered_data['target_model'] == target)][value]
    cvae_data = filtered_data[(filtered_data['weighting_fn'] == 'CVAE') & (filtered_data['target_model'] == target)][value]

    # VAE vs CVAE
    if Metrics == 'U':
        u_stat, p_value = stats.mannwhitneyu(vae_data, cvae_data, alternative='two-sided')
    elif Metrics == 't':
        u_stat, p_value = stats.ttest_ind(vae_data, cvae_data, equal_var=False)
        
    signif = 1 if p_value < 0.05 else 0
    results.append({'Comparison': 'VAE_vs_CVAE', 'Significance': signif, 'LatentDim': latent_dim, 
                    'Dataset': dataset, 'TargetModel': target, 'TStatistic': u_stat, 'PValue': p_value, 'Value': value})


# 結果をDataFrameに変換
results_df = pd.DataFrame(results)

# 結果をCSVファイルに保存
output_file = f'../save_data/analysis_results_spearman_{Metrics}.csv'
results_df.to_csv(output_file, index=False)

print(f"Results saved to {output_file}")

import pandas as pd
from scipy import stats
import itertools

# CSVファイルの読み込み
file_path = '../save_data/result_master_AD.csv'
data = pd.read_csv(file_path)


value = ['mse', 'score', 'Active_latent_dim', 'L1', 'L2']
latent_dim = [2,4,6]
dataset = ['breastcancer', 'liver', 'wine', 'credit_one_hot', 'adult_one_hot']
target = ['NN', 'RF', 'SVM']

p = itertools.product(value, dataset, target, latent_dim)

for value, dataset, target, latent_dim in p:
    # 条件に一致する行の抽出
    filtered_data = data[
        (data['dataset'] == dataset) & 
        (data['noise_std'] == 1) &
        (data['latent_dim'] == latent_dim)
    ]

    # 'auto_encoder'列が'VAE'または'CVAE'である行を抽出
    vae_data = filtered_data[(filtered_data['auto_encoder'] == 'VAE') & (filtered_data['target_model'] == target)][value]
    cvae_data = filtered_data[(filtered_data['auto_encoder'] == 'CVAE') & (filtered_data['target_model'] == target)][value]

    # 独立 2 標本 t検定
    t_stat, p_value = stats.ttest_ind(vae_data, cvae_data)
    if p_value < 0.05:
        p = 'o'
    else:
        p = 'x'

    # 結果の表示
    print(f'{p}, Dim{latent_dim}, dataset{dataset}, target{target}, t統計量: {t_stat}, p値: {p_value}, Value:{value}')

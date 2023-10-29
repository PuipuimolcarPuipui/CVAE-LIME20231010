import pandas as pd
import matplotlib.pyplot as plt
import itertools

# CSVファイルからデータを読み込む
df = pd.read_csv('../save_data/result_master.csv')

# 実験組み合わせの生成
datasets = ['breastcancer', 'liver', 'wine', 'credit', 'adult']
AEs = ['VAE', 'LIME']

for dataset in datasets:
    plt.figure(figsize=(10, 6))
    
    for AE in AEs:
        if AE == 'LIME':
            # 条件に一致する行を抽出する
            selected_rows = df[
                (df['auto_encoder'] == AE) &
                (df['dataset'] == dataset)
            ]
        else:
            std = 1
            # 条件に一致する行を抽出する
            selected_rows = df[
                (df['auto_encoder'] == AE) &
                (df['dataset'] == dataset) &
                (df['noise_std'] == std)
            ]

        # 'instance_no'をx軸に，'MSE'をy軸にとってグラフを作成する
        plt.scatter(selected_rows['instance_no'], selected_rows['mse'], label=AE)
    
    plt.xlabel('Instance No')
    plt.ylabel('MSE')
    plt.title(f'Scatter plot of MSE ({dataset})')
    plt.grid(True)
    plt.ylim(0,1)
    plt.legend()  # 凡例を追加
    
    # Save the figure
    plt.savefig(f'MSE_plotter/MSE_{dataset}.png')
    
    # Show the figure
    plt.show()

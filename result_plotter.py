import pandas as pd
import matplotlib.pyplot as plt
import sys

sys.path.append('/home/CVAE-LIME20230902')

# CSVファイルを読み込む
data = pd.read_csv('result.csv')

datasets = ['adult', 'credit','wine','MNIST'] #,'breastcancer', 'liver'
x = 'noise_std' 
ys = ['R2','mse']
auto_encoders = ['CVAE','VAE']
target_models = ['NN', 'RF', 'SVM','CNN']

def plotter(data, dataset, auto_encoders, x, y, target_model):
    data = data[data['target_model'] == target_model]
    plt.figure(figsize=(10, 6))  # グラフのサイズを設定

    unique_noises = sorted(data[data['dataset'] == dataset][x].unique())
    
    for auto_encoder in auto_encoders:
        # カラム名 '' と '' の条件をフィルタリング
        filtered_data = data[(data['dataset'] == dataset) & (data['auto_encoder'] == auto_encoder)]

        # noise_stdのユニークな値に基づいてデータを整理
        grouped = filtered_data.groupby(x).mean()

        # グラフを描画
        plt.plot(grouped.index, grouped[y], label=auto_encoder+'-LIME')

    plt.xlabel(x, fontsize=20)
    plt.ylabel(y, fontsize=20)
    plt.title(f'{y} at each {x} for {dataset} of {target_model}', fontsize=14)
    plt.legend(fontsize=20)  # 凡例を表示

    # x軸の表示を0.1刻みに設定
    plt.xticks(unique_noises, rotation=45)
    
    # y軸の表示を0から1の範囲で0.1刻みに設定
    if y == 'R2':
        plt.yticks([i/10 for i in range(1, 11)], fontsize=18)
        plt.ylim(0.1, 1)
    # elif y == 'mse':
        # plt.yticks([i/10 for i in range(3)])

    # 補助線を引く
    plt.grid(which='both', linestyle='--', linewidth=0.5)
    plt.tick_params(axis='both', labelsize=18)
    plt.tight_layout()

    # グラフを保存
    plt.savefig(f'save_data/result_plot/output_graph_{y}_{dataset}_{target_model}.png')
    plt.show()

for dataset in datasets:
    for y in ys:
        for target_model in target_models:
            plotter(data, dataset, auto_encoders, x, y, target_model)

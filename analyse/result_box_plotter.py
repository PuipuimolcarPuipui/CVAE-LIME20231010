import pandas as pd
import matplotlib.pyplot as plt
import sys

sys.path.append('/home/CVAE-LIME20230902')

# CSVファイルを読み込む
data = pd.read_csv('result.csv')

datasets = ['breastcancer','hepa', 'liver', 'credit','wine','adult']
x = 'noise_std'
ys = ['score','mse']
auto_encoders = ['CVAE','VAE']
colors = {'CVAE': 'blue', 'VAE': 'orange'}  # auto_encodersの要素ごとの色のマップ

def plotter(dataset, auto_encoders, x, y):
    plt.figure(figsize=(10, 6))

    box_data = []
    box_positions = []

    unique_noises = sorted(data[data['dataset'] == dataset][x].unique())
    for idx, noise in enumerate(unique_noises):
        for ae_idx, auto_encoder in enumerate(auto_encoders):
            filtered_data = data[(data['dataset'] == dataset) & (data['auto_encoder'] == auto_encoder) & (data[x] == noise)]
            box_data.append(filtered_data[y].values)
            box_positions.append(idx * (len(auto_encoders) + 1) + ae_idx)  # Set the positions closer


    flierprops = dict(marker='o', markersize=5, linestyle='none')
    medianprops = dict(linestyle='-', linewidth=1.5)

    bp = plt.boxplot(box_data, positions=box_positions, patch_artist=True, flierprops=flierprops, medianprops=medianprops)

    for i, patch in enumerate(bp['boxes']):
        auto_encoder = auto_encoders[i % len(auto_encoders)]
        patch.set_facecolor(colors[auto_encoder])
        plt.setp(bp['fliers'][i], 'markerfacecolor', colors[auto_encoder])

    handles = [plt.Line2D([0], [0], color=color, lw=2) for color in colors.values()]
    plt.legend(handles, colors.keys())

    plt.xlabel(x, fontsize=14)
    plt.ylabel(y, fontsize=14)
    plt.title(f'{y} at each {x} for {dataset}', fontsize=16)
    plt.grid(which='both', linestyle='--', linewidth=0.5)

    # Adjusting the xtick positions
    # xtick_positions = [idx * 2 + 0.5 for idx in range(len(unique_noises))]  # Center the tick between CVAE and VAE
    xtick_positions = [idx * (len(auto_encoders) + 1) + 0.5 * len(auto_encoders) for idx in range(len(unique_noises))]  # Adjust the centering of ticks
    plt.xticks(xtick_positions, unique_noises, fontsize=12)
    plt.yticks(fontsize=12)

    plt.savefig(f'save_data/result_plot/output_boxplot_{y}_{dataset}.png')
    plt.show()



for dataset in datasets:
    for y in ys:
        plotter(dataset, auto_encoders, x, y)
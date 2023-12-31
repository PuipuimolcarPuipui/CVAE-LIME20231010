import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import itertools
import warnings
import warnings

# sklearnのFutureWarningを無視する
warnings.simplefilter(action='ignore', category=FutureWarning)

# LIMEの設定
# turb_data_by_AE = None
# filter_by_label = False
# dataset = 'breastcancer' #'breast_cancer','hepatisis','liver' 
# weighting_fn = 'LIME'
# AE_latentsize =  None
# select_percent = None
# instance_no = 0
import sys
i = 1 #int(sys.argv[1])

# オートエンコーダ系の設定
turb_data_by_AE = True
filter_by_label = False
dataset = ['breastcancer']#,'credit','wine','credit_one_hot','adult_one_hot','liver'] #'breastcancer','hepa','liver','adult', 'wine', 'mine', 'boston'
weighting_fn = [['VAE','CVAE','LIME'][i]] #'AE','VAE','CVAE','LIME'
AE_latentsize =  4
select_percent = 100
instance_no = range(2)
target_model = 'NN'

#　実験組み合わせの生成
p = itertools.product(dataset, weighting_fn, instance_no)

for dataset, weighting_fn, instance_no in p:
    print(f'==============dataset:{dataset}__AE:{weighting_fn}__instance:{instance_no}=============')

    if weighting_fn == 'AE':
        turb_data_by_AE = False
    
    if weighting_fn == 'LIME':
        AE_latentsize = None
        select_percent = None
        turb_data_by_AE = None
    
    folder = 'turb_' + str(turb_data_by_AE) + '_filter_' + str(filter_by_label) + '_' + str(dataset)+'_'+str(weighting_fn)+'_'+str(AE_latentsize)+'_'+str(select_percent) + '_' + str(instance_no)

    
    # CSVファイルからデータを読み込む
    training_data = pd.read_csv('save_data/test_samples/' + folder + '/training_data.csv', header=None)
    training_data_label = pd.read_csv('save_data/test_samples/' + folder + '/training_data_label.csv', header=None)
    gen_data = pd.read_csv('save_data/test_samples/' + folder + '/gen_data.csv', header=None)
    gen_data_label = pd.read_csv('save_data/test_samples/' + folder + '/gen_data_label.csv', header=None)
    test_instance = pd.read_csv('save_data/test_samples/' + folder + '/test_instance.csv', header=None)
    test_instance_label = pd.read_csv('save_data/test_samples/' + folder + '/test_instance_label.csv', header=None)
    weights = pd.read_csv('save_data/test_samples/' + folder + '/weights.csv', header=None)

    # 生成サンプル数を訓練データの1割に制限
    # training_dataの行数の1割を計算
    sample_size = int(0.1 * len(training_data))

    # gendata, gen_data_label, weights からデータをランダムに抽出
    random_indices = np.random.choice(gen_data.index, size=sample_size, replace=False)
    gen_data = gen_data.iloc[random_indices].reset_index(drop=True)
    gen_data_label = gen_data_label.iloc[random_indices].reset_index(drop=True)
    weights = weights.iloc[random_indices].reset_index(drop=True)
    
    # gen_data_labelの値をtest_instance_labelに合わせる
    if test_instance_label.iloc[0, 0] == 0:
        gen_data_label =  1 - gen_data_label
    
    # データの結合
    combined_data = pd.concat([training_data, gen_data, test_instance], ignore_index=True)
    combined_label = pd.concat([training_data_label, gen_data_label, test_instance_label], ignore_index=True)

    # t-SNEによる次元削減
    tsne = TSNE(n_components=2, random_state=42)
    transformed_data = tsne.fit_transform(combined_data)
    
    # プロット
    plt.figure(figsize=(10, 8))

    # training_dataのデータセットをプロット
    # combined_labelが0.5未満なら青に、0.5以上なら赤にする条件で色分け
    colors = ['blue' if label < 0.5 else 'red' for label in combined_label[0]]
    markers = ['o' if label < 0.5 else '+' for label in combined_label[0]]

    # マーカーの大きさはweightに比例（gen_dataの部分のみ）
    scale_factor = 300

    # print('weights_length: ', len(weights))
    # print('size_length: ', len(sizes))

    # sizesを1次元配列として初期化
    sizes = np.ones(combined_label.shape[0]) * 50
    sizes[len(training_data):len(training_data) + len(gen_data)] = weights.values.squeeze() * scale_factor

    # training_dataを色とサイズの情報を用いてプロット
    plt.scatter(transformed_data[:len(training_data), 0], 
                transformed_data[:len(training_data), 1], 
                c=colors[:len(training_data)], 
                alpha=0.2, 
                s=sizes[:len(training_data)], 
                marker='^',
                label='Training_data')

    # gen_dataを色とサイズの情報を用いてプロット
    # plt.scatter(transformed_data[len(training_data):, 0], 
    #             transformed_data[len(training_data):, 1], 
    #             c=colors[len(training_data):], #'g', #
    #             alpha=0.7, 
    #             s=sizes[len(training_data):], 
    #             marker=markers[len(training_data):], 
    #             label="Genetating_data")
    
    # gen_dataのそれぞれの点を個別にプロット
    for i, data_point in enumerate(transformed_data[len(training_data):]):
        label_index = len(training_data) + i  # インデックスの計算
        label_value = combined_label.iloc[label_index]  # 正しいインデックス方法でラベル値を取得
        marker = 'o' if label_value.iloc[0] < 0.5 else '+'  # マーカーの決定
        plt.scatter(data_point[0], data_point[1], 
                    c=colors[label_index], 
                    alpha=0.7, 
                    s=sizes[label_index], 
                    marker=marker, 
                    label="Generating Data" if i == 0 else "")  # 最初の点だけにラベルをつける

    # test_instanceをプロット
    plt.scatter(transformed_data[-1:, 0], transformed_data[-1:, 1], alpha=0.5, label='Test_instance', color='black',#colors[-1],
                marker=markers[-1], s=300,
                edgecolors='black', linewidths=2 )

    plt.xlabel('t-SNE feature 1', fontsize=14)
    plt.ylabel('t-SNE feature 2', fontsize=14)
    plt.title('t-SNE visualization (Blue is class 0, Red is class 1). Generated by ' + weighting_fn)
    plt.legend()
    plt.savefig(f'save_data/test_samples_tsne/tSNE{folder}.png')
    plt.clf()


    # 実験結果を読み込み
    import csv
    filename = 'save_data/test_result/' + 'turb_' + str(turb_data_by_AE) + '_filter_' + str(filter_by_label) + '_' + str(dataset)+'_'+str(weighting_fn)+'_'+str(AE_latentsize)+'_'+str(select_percent) +'_'+str(target_model)+'.csv'
    
    # 取得したい行番号 (0から始まるインデックスを想定)
    target_row_index = instance_no + 2  # 例: 3行目を取得したい場合

    # CSVファイルを開く
    with open(filename, mode="r", encoding="utf-8") as file:
        reader = csv.reader(file)
        
        # 指定した行番号のデータを取得
        for index, row in enumerate(reader):
            if index == target_row_index:
                target_row = row
                break
    
    R2 =  float(target_row[10])
    MSE = float(target_row[11])
    print('Dataset:', dataset,
          ', Instance:', instance_no,
          ', R2:', "{:.3f}".format(R2),
          ', MSE:', "{:.3f}".format(MSE))


    # ヒストグラムの作成
    # plt.hist(weights, bins=20, edgecolor='black', range=(0, 1))
    hist, bins, _ = plt.hist(weights, bins=30, edgecolor='black', range=(0, 1))

    # ビンの中央にテキストを配置するための座標を計算
    bin_centers = (bins[:-1] + bins[1:])/2

    # 各ビンの要素数をテキストとして表示
    for count, x in zip(hist, bin_centers):
        # x座標、y座標、テキスト
        plt.text(x, count + 0.5, int(count), ha='center', va='bottom', rotation=45)

    # グラフのタイトルとラベルの設定
    # plt.title(f'Weight Distribution, Dataset:{dataset} , Instance:{instance_no}, R2:{R2}, MSE{MSE}')
    plt.title(f'Weight Distribution, Dataset:{dataset} , Instance:{instance_no}, R2:{format(R2, ".2f")}, MSE:{format(MSE, ".2f")}')
    # plt.yticks(np.arange(0, 11000, 1000))
    plt.xlabel('Weight')
    plt.ylabel('Frequency')
    
    # グラフの表示と保存
    # plt.savefig(f'weight_histgram/Weight Distribution, Dataset:{dataset} , Instance:{instance_no}, R2:{R2}, MSE{MSE}.png')
    plt.savefig(f'save_data/test_samples_weight/tSNE{folder}.png')
    plt.show()

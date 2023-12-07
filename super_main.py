from main import main


def super_main(DATA, target, AE, k):
    ## 実験条件 
    dataset = ['breastcancer','credit_one_hot','adult_one_hot','liver','wine','MNIST'][DATA] #, 'boston', 'hepa', ,'boston','adult','credit'
    dataset_class_num = {'breastcancer':2,
                        'hepa': 2,
                        'liver':2,
                        'adult': 2,
                        'wine': 6,
                        'credit':2,
                        'boston':'numerous',
                        'credit_one_hot':2,
                        'adult_one_hot':2,
                        'MNIST':10}
    target_model = ['NN', 'RF', 'SVM'][target] #, 'NN', 'SVM', 'DNN', 'GBM', 'XGB'
    target_model_training =  False
    test_range = range(100)
    num_samples = 5000

    # 検証条件 
    auto_encoder = ['CVAE', 'VAE', 'LIME'][AE] # , 'ICVAE', 'ICVAE2', 'AE'
    auto_encoder_weighting = True
    auto_encoder_sampling = True
    auto_encoder_training = False
    auto_encoder_epochs = 100 if dataset == 'MNIST' else 1000
    auto_encoder_latent_dim = k
    one_hot_encoding = False
    feature_selection = 'auto'
    model_regressor = None
    noise_std = 1 #1
    kernel_width = None #DAE,LIME用
    label_filter = False #False
    select_percent = 100 #100

    #検証内容
    stability_check = False #False
    iAUC_check = True #False

    #少数サンプルをより削除する割合（クラス0は少数サンプルとする）
    reduce_percent = 0 #0

    # 標準化'Standard'　or 正規化'Minimax'
    preprocess = 'Minimax'

    #活性潜在変数か否かの分散ベクトルの閾値
    var_threshold = 0.5

    #条件ベクトルに入力ベクトルを追加
    add_condition = ""#[0,1,2,3,4,5] #[0,1,2,3,4,5,6,7,8,9,10] #[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20] #Noneの時は条件ベクトルなし, 条件ベクトルのコラム番号を指定

    main(dataset,
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
        label_filter,
        select_percent,
        one_hot_encoding,
        feature_selection,
        noise_std,
        model_regressor,
        stability_check,
        kernel_width,
        num_samples,
        var_threshold,
        preprocess)
    
    
if __name__ == '__main__':
    
    import itertools

   
    # 表形式45
    DATA = [0,1,2,3,4] #5
    target = [0,1,2]
    AE = [0,1,2] #0,1
    k = [6] #6,8,10
    
    # MNIST6
    # DATA = [5]
    # target = [0,1,2]
    # AE = [0,1] 
    # k = [10]
    

    # 実行したい実験の番号を指定
    import sys
    n = int(sys.argv[1])  # 例として、5番目の組み合わせを実行します

    # 組み合わせの総数を計算するためには、各リスト自体を使用します。
    total_combinations = list(itertools.product(DATA, target, AE, k))

    # 指定された番号に対応する組み合わせを取得
    # nは1から始まる番号と仮定して、0から始まるインデックスに変換します。
    combination = total_combinations[n-1]

    # 関数を実行
    super_main(*combination)


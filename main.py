import sys
sys.path.append('/home/CVAE-LIME20230802/')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)  # v1 APIのWARNING以上のメッセージも非表示にする
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from functions import *
import time
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="tensorflow")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")

DATA = 6 #int(sys.argv[1])
target = 2 #int(sys.argv[1])
AE = 0 #int(sys.argv[2])

## 実験条件 
dataset = ['breastcancer','credit','adult','liver','wine','credit_one_hot','adult_one_hot'][DATA] #, 'boston', 'hepa'
dataset_class_num = {'breastcancer':2,
                    'hepa': 2,
                    'liver':2,
                    'adult': 2,
                    'wine': 6,
                    'credit':2,
                    'boston':'numerous',
                    'credit_one_hot':2,
                    'adult_one_hot':2}
target_model = ['NN', 'RF', 'SVM'][target] #, 'NN', 'SVM', 'DNN', 'GBM', 'XGB'
target_model_training =  True
test_range = range(1)
num_samples = 5000

# 検証条件 
auto_encoder = ['CVAE', 'VAE', 'AE', 'LIME', 'ICVAE', 'ICVAE2'][AE]
auto_encoder_weighting = True
auto_encoder_sampling = True
auto_encoder_training = False
auto_encoder_epochs = 1000
auto_encoder_latent_dim = 2
one_hot_encoding = False
feature_selection = 'auto'
model_regressor = None
noise_std = 1 #1
kernel_width = None #DAE,LIME用
label_filter = True #False
select_percent = 100 #100

#検証内容
stability_check = False #False
iAUC_check = True #False

# 少数サンプルをより削除する割合（クラス0は少数サンプルとする）
reduce_percent = 0 #0

def main(dataset,
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
        num_samples):
    print(f'dataset:{dataset}_AE:{auto_encoder}_target_model:{target_model}')
    
    ## 前処理済みデータセットのロード
    # データセットをCSVファイルから読み込む
    data = pd.read_csv(f'dataset/{dataset}.csv')
    # データセットを特徴量とターゲットに分割
    X = data.drop('target', axis=1)
    y = data['target']
    
    # 少数クラス(Class0)を削除
    if reduce_percent != 0:
        from functions import small_sample_reduce
        X, y = small_sample_reduce(X, y, reduce_percent)

    # データセットをトレーニングセットとテストセットに分割
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    ## ターゲットモデルの学習又はロード(predict_probaで返す)
    model = target_model_loder(dataset = dataset,
                                target_model = target_model,
                                target_model_training = target_model_training,
                                X_train = X_train,
                                y_train = y_train,
                                dataset_class_num = dataset_class_num)

    ## 実験結果格納用のCSVを定義
    df = pd.DataFrame([['','','','','','','','','','','','','','','','']],
                        columns=['dataset',
                                'weighting_fn',
                                'epoch_num',
                                'latent_size',
                                'num_samples',
                                'select_percent',
                                'instance_no',
                                'predict_label',
                                'label', 
                                'r2', 
                                'mse',
                                'element1',
                                'element3',
                                'process_time',
                                'target_model',
                                'local_output'])
    output_path = 'save_data/test_result/turb_'+str(auto_encoder_sampling)+'_filter_'+str(label_filter)+'_'+str(dataset)+'_'+str(auto_encoder)+'_'+str(auto_encoder_latent_dim)+'_'+str(select_percent)+'_'+str(target_model)+'.csv'
    df.to_csv(output_path)
    
    # iAUC計算用
    data_for_iAUC = []
    
    ## 実験
    # LIME explainerを作成
    if auto_encoder ==  'LIME':
        from limes.test_lime.lime_tabular import LimeTabularExplainer
        # セッティングを辞書に格納
        lime_setting = {
                        'auto_encoder':auto_encoder,
                        'auto_encoder_weighting':None,
                        'auto_encoder_sampling':None,
                        'auto_encoder_training':None,
                        'epochs':None,
                        'dataset':dataset,
                        'latent_dim':None,
                        'num_samples':num_samples,
                        'instance_no':None,
                        'X_test':X_test,
                        'predict_fn':model,
                        'mode':['regression' if dataset_class_num[dataset]=='numerous' else 'classification'][0],
                        'filtering':label_filter,
                        'select_percent':None,
                        'y_test':y_test,
                        'dataset_class_num':dataset_class_num[dataset],
                        }
        explainer = LimeTabularExplainer(X_train.values, 
                                        mode=['regression' if dataset_class_num[dataset]=='numerous' else 'classification'][0], 
                                        training_labels=y_train, 
                                        feature_names=X_train.columns.tolist(),
                                        # random_state=42,
                                        lime_setting=lime_setting,
                                        feature_selection=feature_selection,
                                        kernel_width=kernel_width)
        
        setting_dic = lime_setting

    else:
        from limes.cvae_lime.lime_tabular import LimeTabularExplainer
        # セッティングを辞書に格納
        auto_encoder_setting = {'auto_encoder':auto_encoder,
                                'auto_encoder_weighting':auto_encoder_weighting,
                                'auto_encoder_sampling':auto_encoder_sampling,
                                'auto_encoder_training':auto_encoder_training,
                                'epochs':auto_encoder_epochs,
                                'dataset':dataset,
                                'latent_dim':auto_encoder_latent_dim,
                                'num_samples':num_samples,
                                'instance_no':None,
                                'X_test':X_test,
                                'predict_fn':model,
                                'mode':['regression' if dataset_class_num[dataset]=='numerous' else 'classification'][0],
                                'filtering':label_filter,
                                'select_percent':select_percent,
                                'y_test':y_test,
                                'dataset_class_num':dataset_class_num[dataset],
                                'one_hot_encoding':one_hot_encoding,
                                'noise_std':noise_std,
                                'kernel_width':kernel_width,                                                            
                                }
        explainer = LimeTabularExplainer(X_train.values, 
                                        mode=['regression' if dataset_class_num[dataset]=='numerous' else 'classification'][0], 
                                        training_labels=y_train, 
                                        feature_names=X_train.columns.tolist(),
                                        # random_state=42,
                                        auto_encoder_setting=auto_encoder_setting,
                                        feature_selection=feature_selection
                                        )
        
        setting_dic = auto_encoder_setting
    
    # テストセットの一部のインスタンスに対して説明を取得
    for i in test_range:
        # 生成サンプルの保存フォルダの作成
        from functions import create_folder
        exp_setting = 'turb_'+str(setting_dic['auto_encoder_sampling'])+'_filter_'+str(setting_dic['filtering'])+'_'+str(setting_dic['dataset'])+'_'+str(setting_dic['auto_encoder'])+'_'+str(setting_dic['latent_dim'])+'_'+str(setting_dic['select_percent'])+'_'+str(i)
        # exp_setting = 'turb_'+str(auto_encoder_sampling)+'_filter_'+str(filtering)+'_'+str(dataset)+'_'+str(auto_encoder)+'_'+str(auto_encoder_latent_dim)+'_'+str(select_percent)+'_'+str(i)
        create_folder('save_data/test_samples/' + exp_setting)
        
        #　計算時間の計測開始
        start = time.time()
        
        #　説明の生成
        exp = explainer.explain_instance(X_test.values[i],
                                        model,
                                        num_samples=num_samples,
                                        instance_no=i,
                                        model_regressor=model_regressor,
                                        top_labels = 1,
                                        label_filter=label_filter,
                                        #  num_features=30
                                        )
        
        # 評価値の算出
        process_time = time.time() - start
        process_time = f'{process_time:.1f}'
        local_output = model(X_test.values[i].reshape(1, -1)).reshape(-1)
        score = exp.score
        score = f'{score:.3f}'
        mse = 0.5*(exp.local_pred - np.max(local_output))**2
        mse = f'{mse[0]:.3f}'
        # iAUC = None
        print(f'instance:{i}, score:{score}, mse:{mse}, class:{np.argmax(local_output)}')

        # 実験結果の保存
        df = pd.read_csv(output_path,index_col=0)
        df.to_csv(output_path)
        temp = pd.DataFrame([[dataset,
                            auto_encoder,
                            int(auto_encoder_epochs),
                            int(auto_encoder_latent_dim),
                            int(num_samples),
                            select_percent,
                            int(i),
                            [np.argmax(local_output) if dataset_class_num[dataset]!='numerous' else local_output[0]],
                            y_test.values[i],
                            score,
                            mse,
                            auto_encoder_sampling,
                            label_filter,
                            process_time,
                            target_model,
                            min(local_output)]],
                            columns=['dataset', 'weighting_fn', 'epoch_num', 'latent_size', 'num_samples','select_percent','instance_no','predict_label','label', 'r2', 'mse','element1','element3','process_time','target_model','local_output'])
        temp = temp.astype({col: 'int' for col in temp.columns if temp[col].dtype == 'bool'})
        df = pd.concat([df, temp], axis=0)
        df.to_csv(output_path)
        
        #　訓練データ等の保存
        np.savetxt('save_data/test_samples/' + exp_setting + '/training_data.csv', X_train, delimiter=',')
        np.savetxt('save_data/test_samples/' + exp_setting + '/test_instance.csv', X_test.values[i].reshape(1, -1), delimiter=',')
        np.savetxt('save_data/test_samples/' + exp_setting + '/training_data_label.csv', y_train, delimiter=',')
        np.savetxt('save_data/test_samples/' + exp_setting + '/test_instance_label.csv', [np.argmax(local_output) if dataset_class_num[dataset]!='numerous' else local_output[0]], delimiter=',')
        # iAUC計算用データの格納
        data_for_iAUC.append({'X_test':X_test.iloc[i],
                            'exp':exp,
                            'model':model})

    if stability_check == True:
        # 実行の度の計算結果を格納
        return exp.as_list(label=np.argmax(local_output)), score, mse, [np.argmax(local_output) if dataset_class_num[dataset]!='numerous' else local_output[0]], y_test.values[i]
    
    if iAUC_check == True:
        # オブジェクトをファイルに書き出す関数
        from functions import write_object_to_file
        filename = str(dataset)+'_'+str(auto_encoder)
        write_object_to_file(data_for_iAUC, 'save_data/test_iAUC/' + filename + '.dill')

if __name__ == '__main__':
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
        num_samples)

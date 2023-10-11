def target_model_loder(dataset = None,
                        target_model = None,
                        target_model_training = None,
                        X_train = None,
                        y_train = None,
                        dataset_class_num = None):
    import pickle
    
    # 指標を計算する関数
    def calculate_and_display_metrics(model, X_valid, y_valid, num_classes, dataset):
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        from sklearn.preprocessing import label_binarize

        num_class = num_classes[dataset]

        try:
            probabilities = model.predict(X_valid)
        except:
            probabilities = model(X_valid)
        

        if num_class == 2:  # 2-class classification
            y_pred = (probabilities > 0.5).astype(int)
        else:  # Multi-class classification
            import numpy as np
            if probabilities.shape == 2:
                y_pred = np.argmax(probabilities, axis=1)
                y_valid = np.argmax(y_valid, axis=1)
            elif num_class == 3:
                y_pred = np.argmax(probabilities, axis=1)
                y_valid = np.argmax(y_valid, axis=1)
            elif num_class == 6:
                y_pred = np.argmax(probabilities, axis=1)
                y_valid = np.argmax(y_valid, axis=1)
            elif num_class == 10:
                y_pred = np.argmax(probabilities, axis=1)
                y_valid = np.argmax(y_valid, axis=1)
            else:
                y_pred = probabilities
                y_valid = y_valid

        acc = accuracy_score(y_valid, y_pred)
        precision = precision_score(y_valid, y_pred, average='macro')
        recall = recall_score(y_valid, y_pred, average='macro')
        f1 = f1_score(y_valid, y_pred, average='macro')

        # if num_class == 2:
        #     auc = roc_auc_score(y_valid, probabilities)
        #     print(f"ROC AUC: {auc:.4f}")
        # else:  # For multi-class, compute the AUC for each class
        #     y_valid_bin = label_binarize(y_valid, classes=[i for i in range(num_class)])
        #     y_pred = label_binarize(probabilities, classes=[i for i in range(num_class)])
        #     # auc = roc_auc_score(y_valid_bin, y_pred, multi_class='ovr', average='macro')
        #     # print(f"Macro ROC AUC: {auc:.4f}")

        print(f"Accuracy: {acc:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
    
    # モデルのファイルパス
    model_filepath = f'save_data/target_model/{target_model}_{dataset}.pkl'
    
    # X_train,y_trainから検証用を抽出
    from sklearn.model_selection import train_test_split
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    # ターゲットモデル毎の設定(predict_probaで返す様に設定せよ)
    if target_model == 'RF':
        
        if target_model_training == True:
            if dataset_class_num[dataset] == 'numerous':
                from sklearn.ensemble import RandomForestRegressor
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model.fit(X_train, y_train)
                model = model.predict
            else:
                from sklearn.ensemble import RandomForestClassifier
                model = RandomForestClassifier(random_state=42)
                model.fit(X_train, y_train)
                calculate_and_display_metrics(model, X_valid, y_valid, dataset_class_num, dataset)
                model = model.predict_proba
                
            with open(model_filepath, 'wb') as file:
                pickle.dump(model, file)
            
        elif target_model_training == False:
            with open(model_filepath, 'rb') as file:
                model = pickle.load(file)
                
        return model
    
    if target_model == 'NN':
        from keras.models import Sequential, load_model
        from keras.layers import Dense

        if target_model_training == True:
            model = Sequential()

            if dataset_class_num[dataset] == 2:
                model.add(Dense(12, input_dim=X_train.shape[1], activation='relu'))
                model.add(Dense(8, activation='relu'))
                model.add(Dense(1, activation='sigmoid'))
                model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
                model.fit(X_train, y_train, epochs=30, batch_size=10, verbose=0)
                calculate_and_display_metrics(model, X_valid, y_valid, dataset_class_num, dataset)
                # 過学習対策
                # from keras.callbacks import EarlyStopping
                # early_stop = EarlyStopping(monitor='val_loss', patience=5)
                # model.fit(X_train, y_train, epochs=50, batch_size=10, verbose=1, callbacks=[early_stop], validation_data=(X_valid, y_valid))
                with open(model_filepath, 'wb') as file:
                    pickle.dump(model, file)     
            elif dataset_class_num[dataset] == 'numerous':
                model = Sequential([
                    Dense(12, activation='relu', input_shape=(X_train.shape[1],)),
                    Dense(8, activation='relu'),
                    Dense(1)
                ])
                model.compile(optimizer='adam', loss='mse')
                model.fit(X_train, y_train, epochs=300, batch_size=32, validation_split=0.1, verbose=0)
                # 過学習対策
                # from keras.callbacks import EarlyStopping
                # early_stop = EarlyStopping(monitor='val_loss', patience=5)
                # model.fit(X_train, y_train, epochs=50, batch_size=10, verbose=1, callbacks=[early_stop], validation_data=(X_valid, y_valid))
                with open(model_filepath, 'wb') as file:
                    pickle.dump(model, file)     
            else:
                from keras.utils import to_categorical
                y_train = to_categorical(y_train, num_classes=dataset_class_num[dataset])
                y_valid = to_categorical(y_valid, num_classes=dataset_class_num[dataset])
                model.add(Dense(12, input_dim=X_train.shape[1], activation='relu'))
                model.add(Dense(8, activation='relu'))
                model.add(Dense(dataset_class_num[dataset], activation='softmax'))
                model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
                model.fit(X_train, y_train, epochs=50, batch_size=10, verbose=1)
                calculate_and_display_metrics(model, X_valid, y_valid, dataset_class_num, dataset)
                # 過学習対策
                # from keras.callbacks import EarlyStopping
                # early_stop = EarlyStopping(monitor='val_loss', patience=5)
                # model.fit(X_train, y_train, epochs=50, batch_size=10, verbose=1, callbacks=[early_stop], validation_data=(X_valid, y_valid))
                with open(model_filepath, 'wb') as file:
                    pickle.dump(model, file)
        
        elif target_model_training == False:
            with open(model_filepath, 'rb') as file:
                model = pickle.load(file)

                
        # 出力形式の整形
        if dataset_class_num[dataset] == 'numerous':
            def custom_predict_fn(data):
                import numpy as np
                # Kerasモデルのpredictメソッドを使って、2D配列の確率を取得
                probabilities = model.predict(data, verbose=0)
                # 確率を1D配列に変換
                return probabilities
            
        elif dataset_class_num[dataset] > 2:
            def custom_predict_fn(data):
                import numpy as np
                # Kerasモデルのpredictメソッドを使って、2D配列の確率を取得
                probabilities = model.predict(data, verbose=0)
                return probabilities
        else:
            def custom_predict_fn(data):
                import numpy as np
                # Kerasモデルのpredictメソッドを使って、2D配列の確率を取得
                probabilities = model.predict(data, verbose=0)
                # 確率を1D配列に変換
                return np.hstack((1-probabilities, probabilities))
                    
        return custom_predict_fn
    
    if target_model == 'CNN':
        from keras.models import Sequential, load_model
        from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
        from keras.utils import to_categorical
        
        # モデルを訓練するためのデータをCNNに合わせて変形
        X_train = X_train.values.reshape(X_train.shape[0], 28, 28, 1)
        X_valid = X_valid.values.reshape(X_valid.shape[0], 28, 28, 1)

        if target_model_training == True:
            model = Sequential()

            # CNNモデルの定義
            model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Dropout(0.25))
            model.add(Flatten())
            model.add(Dense(128, activation='relu'))
            model.add(Dropout(0.5))
            model.add(Dense(10, activation='softmax'))

            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

            y_train = to_categorical(y_train, 10)
            y_valid = to_categorical(y_valid, 10)

            model.fit(X_train, y_train, validation_data=(X_valid, y_valid), epochs=10, batch_size=200, verbose=1)
            calculate_and_display_metrics(model, X_valid, y_valid, dataset_class_num, dataset)
            # モデルの保存
            with open(model_filepath, 'wb') as file:
                pickle.dump(model, file)
        
        elif target_model_training == False:
            with open(model_filepath, 'rb') as file:
                model = pickle.load(file)
        
                # 出力形式の整形
        if dataset_class_num[dataset] == 'numerous':
            def custom_predict_fn(data):
                import numpy as np
                # Kerasモデルのpredictメソッドを使って、2D配列の確率を取得
                probabilities = model.predict(data, verbose=0)
                # 確率を1D配列に変換
                return probabilities
            
        elif dataset_class_num[dataset] > 2:
            def custom_predict_fn(data):
                import numpy as np
                # データの形状を変更
                data = data.reshape(data.shape[0], 28, 28, 1)
                # Kerasモデルのpredictメソッドを使って、2D配列の確率を取得
                probabilities = model.predict(data, verbose=0)
                return probabilities
        else:
            def custom_predict_fn(data):
                import numpy as np
                # Kerasモデルのpredictメソッドを使って、2D配列の確率を取得
                probabilities = model.predict(data, verbose=0)
                # 確率を1D配列に変換
                return np.hstack((1-probabilities, probabilities))
                    
        return custom_predict_fn, model       
        
    
    if target_model == 'DNN':
        from keras.models import Sequential, load_model
        from keras.layers import Dense

        if target_model_training == True:
            model = Sequential()
            
            if dataset_class_num[dataset] == 2:
                model.add(Dense(12, input_dim=X_train.shape[1], activation='relu'))
                model.add(Dense(10, activation='relu'))
                model.add(Dense(8, activation='relu'))
                model.add(Dense(4, activation='relu'))
                model.add(Dense(1, activation='sigmoid'))
                model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
                # model.fit(X_train, y_train, epochs=50, batch_size=10, verbose=1)
                # 過学習対策
                from keras.callbacks import EarlyStopping
                early_stop = EarlyStopping(monitor='val_loss', patience=5)
                model.fit(X_train, y_train, epochs=50, batch_size=10, verbose=1, callbacks=[early_stop], validation_data=(X_valid, y_valid))
                calculate_and_display_metrics(model, X_valid, y_valid, dataset_class_num, dataset)
                
            elif dataset_class_num[dataset] == 'numerous':
                model = Sequential([
                    Dense(12, activation='relu', input_shape=(X_train.shape[1],)),
                    Dense(10, activation='relu'),
                    Dense(8, activation='relu'),
                    Dense(4, activation='relu'),
                    Dense(1)
                ])
                model.compile(optimizer='adam', loss='mse')
                # model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.1)
                # 過学習対策
                from keras.callbacks import EarlyStopping
                early_stop = EarlyStopping(monitor='val_loss', patience=5)
                model.fit(X_train, y_train, epochs=50, batch_size=10, verbose=1, callbacks=[early_stop], validation_data=(X_valid, y_valid))

            else:
                from keras.utils import to_categorical
                y_train = to_categorical(y_train, num_classes=dataset_class_num[dataset])
                y_valid = to_categorical(y_valid, num_classes=dataset_class_num[dataset])
                model.add(Dense(12, input_dim=X_train.shape[1], activation='relu'))
                model.add(Dense(10, activation='relu'))
                model.add(Dense(8, activation='relu'))
                model.add(Dense(4, activation='relu'))
                model.add(Dense(dataset_class_num[dataset], activation='softmax'))
                model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
                # model.fit(X_train, y_train, epochs=50, batch_size=10, verbose=1)
                # 過学習対策
                from keras.callbacks import EarlyStopping
                early_stop = EarlyStopping(monitor='val_loss', patience=5)
                model.fit(X_train, y_train, epochs=50, batch_size=10, verbose=1, callbacks=[early_stop], validation_data=(X_valid, y_valid))
                calculate_and_display_metrics(model, X_valid, y_valid, dataset_class_num, dataset)
                
            with open(model_filepath, 'wb') as file:
                pickle.dump(model, file)
        
        elif target_model_training == False:
            with open(model_filepath, 'rb') as file:
                model = pickle.load(file)
        
        # 出力形式の整形
        if dataset_class_num[dataset] == 'numerous':
            def custom_predict_fn(data):
                import numpy as np
                # Kerasモデルのpredictメソッドを使って、2D配列の確率を取得
                probabilities = model.predict(data, verbose=0)
                # 確率を1D配列に変換
                return probabilities
            
        elif dataset_class_num[dataset] > 2:
            def custom_predict_fn(data):
                import numpy as np
                # Kerasモデルのpredictメソッドを使って、2D配列の確率を取得
                probabilities = model.predict(data, verbose=0)
                return probabilities
        else:
            def custom_predict_fn(data):
                import numpy as np
                # Kerasモデルのpredictメソッドを使って、2D配列の確率を取得
                probabilities = model.predict(data, verbose=0)
                # 確率を1D配列に変換
                return np.hstack((1-probabilities, probabilities))
                
        return custom_predict_fn
    
    if target_model == 'SVM':

        if target_model_training == True:
            if dataset_class_num[dataset] == 'numerous':
                from sklearn.svm import SVR
                model = SVR()
                model.fit(X_train, y_train)
                model = model.predict
            else:
                from sklearn.svm import SVC
                model = SVC(probability=True, random_state=42)
                model.fit(X_train, y_train)
                calculate_and_display_metrics(model, X_valid, y_valid, dataset_class_num, dataset)
                model = model.predict_proba
            with open(model_filepath, 'wb') as file:
                pickle.dump(model, file)
        
        elif target_model_training == False:
            with open(model_filepath, 'rb') as file:
                model = pickle.load(file)
                
        return model
    
    if target_model == 'GBM':
        
        if target_model_training == True:
            if dataset_class_num[dataset] == 'numerous':
                from lightgbm import LGBMRegressor, early_stopping
                model = LGBMRegressor(max_depth=4, colsample_bytree=0.5, 
                        reg_lambda=0.5, reg_alpha=0.5, 
                        importance_type="gain", random_state=100)
                # model.fit(X_train, y_train)
                # 過学習対策
                model.fit(X_train, y_train, early_stopping_rounds=50, eval_set=[(X_valid, y_valid)])
            else:
                from lightgbm import LGBMClassifier, early_stopping
                model = LGBMClassifier(max_depth=4, colsample_bytree=0.5, 
                        reg_lambda=0.5, reg_alpha=0.5, 
                        importance_type="gain", random_state=100)
                # model.fit(X_train, y_train)
                # 過学習対策
                model.fit(X_train, y_train, early_stopping_rounds=50, eval_set=[(X_valid, y_valid)])
            with open(model_filepath, 'wb') as file:
                pickle.dump(model, file)
        
        elif target_model_training == False:
            with open(model_filepath, 'rb') as file:
                model = pickle.load(file)
        
        if dataset_class_num[dataset] == 'numerous':
            def predict_fn(X):
                if len(X.shape)==1:
                    return model.predict(X.reshape(1,-1))[0]
                else:
                    return model.predict(X)
        else:
            # 出力形式の整形
            def predict_fn(X):
                if len(X.shape)==1:
                    return model.predict_proba(X.reshape(1,-1))[0]
                else:
                    return model.predict_proba(X)

        return predict_fn
    
    if target_model == 'XGB':
        import xgboost as xgb

        if target_model_training == True:
            if dataset_class_num[dataset] == 'numerous':
                model = xgb.XGBRegressor(random_state=42, eval_metric="rmse")
                # model.fit(X_train, y_train)
                # 過学習対策
                model.fit(X_train, y_train, early_stopping_rounds=10, eval_set=[(X_valid, y_valid)])
                model = model.predict
            else:
                model = xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric="logloss")
                # model.fit(X_train, y_train)
                # 過学習対策
                model.fit(X_train, y_train, early_stopping_rounds=10, eval_set=[(X_valid, y_valid)])
                model = model.predict_proba
            with open(model_filepath, 'wb') as file:
                pickle.dump(model, file)
        
        elif target_model_training == False:
            with open(model_filepath, 'rb') as file:
                model = pickle.load(file)
                
        return model

def create_folder(path):
    """指定したパスにフォルダを作成します。"""
    import os
    if not os.path.exists(path):
        os.makedirs(path)
        
def quantize_matrix(mat):
    '''
    ・入力は0~1で正規化している必要がある
    ・最大値と最小値を4分割し,0~3の整数値に置き換える
    '''
    # 入力の形状を保持しておきます
    original_shape = mat.shape

    # 行列を1次元に変換
    flattened = mat.ravel()

    # 各要素に対して量子化を適用
    import numpy as np
    quantized = np.piecewise(flattened, 
                             [flattened < 0.25, 
                              (0.25 <= flattened) & (flattened < 0.5),
                              (0.5 <= flattened) & (flattened < 0.75),
                              0.75 <= flattened],
                             [0, 1, 2, 3])

    # 元の形状に戻して返却
    return quantized.reshape(original_shape)

def get_highest_probability_index(data):
    '''
    クラス分類の出力値のargmaxを返す
    '''
    import numpy as np
    # Numpy配列に変換
    arr = np.array(data)

    # 最大値のインデックスを返す
    return np.argmax(arr)

def jaccard_index(set_a, set_b):
    """ジャカード指数を計算する関数"""
    intersection = len(set_a.intersection(set_b))
    union = len(set_a.union(set_b))
    return intersection / union

def calculate_jaccard_for_all_combinations(features_list):
    """複数の特徴のリストから、すべての組み合わせのジャカード指数を計算する関数"""
    n = len(features_list)
    jaccard_values = []

    for i in range(n):
        for j in range(i+1, n):
            set_a = set(features_list[i])
            set_b = set(features_list[j])
            jaccard_values.append(jaccard_index(set_a, set_b))

    return jaccard_values

def extract_top_n_percent(samples, weights, labels, n):
    import numpy as np
    # 上位n%のインデックス数を計算
    top_n_idx_count = int(len(weights) * n / 100)

    # weightsの上位n%のインデックスを取得
    top_indices = np.argsort(weights)[-top_n_idx_count:]

    # 上位n%に対応するsamplesとlabelsを取得
    top_samples = samples[top_indices]
    top_labels = labels[top_indices]
    top_weights = weights[top_indices]

    return top_samples, top_weights, top_labels

# データをCSVファイルに書き込む関数
def append_to_csv(filename, data, column_names):
    '''
    filename:出力先のパス
    data:リスト形式で渡す
    column_names:リスト形式でカラム名を渡す
    '''
    import csv
    # ファイルが存在しない場合、ヘッダを書き込む
    try:
        with open(filename, 'r') as f:
            pass
    except FileNotFoundError:
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            # writer.writerow(["dataset", "target_model", "auto_encoder_weighting", "auto_encoder_sampling", "auto_encoder", "instance_no", "noise_std", "kernel_width"])
            writer.writerow(column_names)

    # データを追加
    with open(filename, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(data)


def filtering(neighborhood_data, neighborhood_labels, weights, label):
    # neighborhood_labels の label の列の値が各行の最大値である行のインデックスを抽出
    import numpy as np
    max_indices = np.argmax(neighborhood_labels, axis=1)
    selected_rows = np.where(max_indices == label)[0]

    # 対応する行の weights と neighborhood_data を再定義
    weights = weights[selected_rows]
    neighborhood_data = neighborhood_data[selected_rows]
    neighborhood_labels = neighborhood_labels[selected_rows]
    
    return neighborhood_data, neighborhood_labels, weights


def small_sample_reduce(X, y, reduce_percent):
    import numpy as np
    # 値が0のインデックスを取得
    zero_indices = np.where(y == 0)[0]

    # 削除する行数を計算
    delete_count = int(len(zero_indices) * reduce_percent)  # 例として50%を削除

    # 削除するインデックスをランダムに選択
    delete_indices = np.random.choice(zero_indices, delete_count, replace=False)

    # Xとyから指定されたインデックスを削除
    X_new = X.drop(delete_indices)
    y_new = y.drop(delete_indices)
    
    return X_new, y_new

def iAUC(model, surrogate_model, test_data):
    import numpy as np
    from sklearn.metrics import log_loss
    
    def masking_function(x, s, mask_value=-999):
        """
        Return x after replacing features where s_i = 0 by mask_value.
        """
        return [xi if si else mask_value for xi, si in zip(x, s)]

    def topn_attributions(lime_explanation, n):
        """
        Return a binary mask where the top n% of features have a value of 1.
        """
        n_features = len(lime_explanation)
        top_n = int(np.ceil(n * n_features / 100))
        sorted_indices = sorted(range(n_features), key=lambda i: abs(lime_explanation[i]), reverse=True)
        s = [0] * n_features
        for idx in sorted_indices[:top_n]:
            s[idx] = 1
        return s

    def compute_iAUC(model, surrogate_model, test_data):
        '''
        lime_explanation:特徴量と係数が入ったリスト
        
        '''
        iAUC_values = []
        
        for n in range(101):  # From 0 to 100
            log_likelihoods = []
            
            for x, y, exp in test_data:
                # Get LIME explanation
                # exp = explainer.explain_instance(x, model.predict_proba).as_list()
                lime_explanation = [value for feature, value in exp.local_exp]
                
                # Mask top n% of features
                s = topn_attributions(lime_explanation, n)
                x_masked = masking_function(x, s)
                
                # Compute the log-likelihood using surrogate model
                probs = surrogate_model.predict_proba([x_masked])
                log_likelihood = -log_loss([y], probs, labels=class_names)
                log_likelihoods.append(log_likelihood)
                
            expected_log_likelihood = np.mean(log_likelihoods)
            iAUC_values.append(expected_log_likelihood)
        
        return np.trapz(iAUC_values)  # Compute the area under the curve
    
    return compute_iAUC(model, surrogate_model, test_data)

# オブジェクトをファイルに書き出す関数
def write_object_to_file(obj, filename):
    import dill
    with open(filename, 'wb') as f:
        dill.dump(obj, f)

# ファイルからオブジェクトを読み出す関数
def read_object_from_file(filename):
    import dill
    with open(filename, 'rb') as f:
        return dill.load(f)
    
# MNISTを含むexpの画像を表示するプログラム
def MNIST_exp(test_instance, exp, auto_encoder, target_model, instance_no, label, X_train, original_model=None):
    import numpy as np
    import matplotlib.pyplot as plt
    from PIL import Image

    # サンプルのデータ (特徴量の番号, 特徴量のデータ)
    features_list = list(exp.values())[0]

    # 1. 特徴量の番号でソート
    sorted_features = sorted(features_list, key=lambda x: x[0])

    # 2. 特徴量のデータだけ取り出し
    data_only = [item[1] for item in sorted_features]

    # 3. このデータを28x28の形状にreshape
    image_data1 = np.array(data_only).reshape(28, 28)

    # Test instanceのデータを28x28の形状にreshape
    image_data2 = np.array(test_instance).reshape(28, 28)

    # 最初の画像を描画して保存
    plt.imshow(image_data1, cmap='gray')
    plt.axis('off')
    plt.savefig(f"temp/exp_{auto_encoder}_{target_model}.png", bbox_inches='tight', pad_inches=0)
    plt.close()  # グラフをクリア

    # 2番目の画像を描画して保存
    plt.imshow(image_data2, cmap='gray')
    plt.axis('off')
    plt.savefig(f"temp/testinstance_{auto_encoder}_{target_model}.png", bbox_inches='tight', pad_inches=0)
    plt.close()  # グラフをクリア
    
    if original_model != None:
        from tf_keras_vis.gradcam import Gradcam
        from tf_keras_vis.saliency import Saliency
        from tf_keras_vis.utils import normalize

        test_instance = np.array(test_instance).reshape(1, 28, 28, 1)

        def model_modifier(m):
            return m[:, label]
        
        # Generate GradCam AttributionMAP
        gradcam = Gradcam(original_model)
        cam = gradcam(model_modifier, seed_input=test_instance, penultimate_layer=-1)  # You may need to specify a different layer
        cam = normalize(cam)
        
        # score関数を作成します。この関数は、モデルの出力から特定のクラスのスコアを取得します。
        def get_score(output):
            return output[:, label]
        
        # Generate IntegralGradient AttributionMAP
        integral_gradient = Saliency(original_model)
        mask = integral_gradient(score=get_score, seed_input=test_instance)
        mask = normalize(mask)
        
        # Compute SHAP values
        import shap
        from lime import lime_image
        from lime.wrappers.scikit_image import SegmentationAlgorithm
        import tensorflow as tf
        tf.config.experimental_run_functions_eagerly(False)
        background_data_indices = np.random.choice(len(X_train), 100, replace=False)
        background_data = X_train[background_data_indices]
        background_data = np.array(background_data).reshape(len(background_data), 28, 28, 1)
        explainer = shap.DeepExplainer(original_model, background_data)
        shap_values = explainer.shap_values(test_instance)
        shap_image = shap_values[0][0].reshape(28, 28)
        
        # LIMEの説明を生成
        from lime import lime_image
        from lime.wrappers.scikit_image import SegmentationAlgorithm
        import tensorflow as tf
        tf.config.experimental_run_functions_eagerly(True)

        explainer = lime_image.LimeImageExplainer()
        segmenter = SegmentationAlgorithm('quickshift', kernel_size=1, max_dist=200, ratio=0.2)
        # LIMEはモデルの出力として確率値を必要とするため、predict_proba関数を作成します。
        def predict_proba(images):
            # グレースケール画像に変換
            images = np.mean(images, axis=-1, keepdims=True)
            preds = original_model.predict(images)
            return preds
        explanation = explainer.explain_instance(test_instance.reshape(28, 28), predict_proba, top_labels=5, hide_color=0, num_samples=1000, segmentation_fn=segmenter)
        temp, _ = explanation.get_image_and_mask(label, positive_only=False, hide_rest=False, num_features=10, min_weight=0.01)
        
        # Plot and save all images
        fig, axs = plt.subplots(1, 6, figsize=(20,5))

        axs[0].imshow(image_data2, cmap='gray')
        axs[0].axis('off')
        
        axs[1].imshow(image_data1, cmap='jet')
        axs[1].axis('off')
        
        axs[2].imshow(temp, cmap='jet') # LIMEの説明
        axs[2].axis('off')
        
        axs[3].imshow(shap_image, cmap='jet') # Displaying SHAP
        axs[3].axis('off')
        
        axs[4].imshow(cam[0], cmap='jet')
        axs[4].axis('off')
        
        axs[5].imshow(mask[0], cmap='jet')
        axs[5].axis('off')
        

        plt.subplots_adjust(wspace=0, hspace=0)
        plt.tight_layout()
        plt.savefig(f"save_data/MNIST_png/combined_GradCAM_and_IG_{auto_encoder}_{target_model}_{label}_{instance_no}.png", bbox_inches='tight', pad_inches=0)
        plt.close()
        
    else:
        # 2つの画像を読み込む
        img1 = Image.open(f"temp/exp_{auto_encoder}_{target_model}.png")
        img2 = Image.open(f"temp/testinstance_{auto_encoder}_{target_model}.png")

        # 画像を横に結合
        combined_img = Image.new('RGB', (img1.width + img2.width, img1.height))
        combined_img.paste(img2, (0, 0))
        combined_img.paste(img1, (img1.width, 0))

        # 結合した画像を保存
        combined_img.save(f"save_data/MNIST_png/combined_{auto_encoder}_{target_model}_{label}_{instance_no}.png")


    
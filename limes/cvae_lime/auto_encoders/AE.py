import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Lambda, concatenate
from tensorflow.keras.optimizers import Adam
from sklearn.datasets import load_breast_cancer
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import mse
import tensorflow.keras.backend as K
from keras.losses import binary_crossentropy, mean_squared_error



def AE_training(X_train=None,
            y_train=None,
            X_test=None,
            y_test=None,
            epochs=None,
            latent_dim=None,
            dataset=None,
            auto_encoder=None,
            dataset_class_num=None,
            one_hot_encoding=None,
            verbose=1
            ):
    import sys
    sys.path.append('/home/CVAE-LIME20230902')
    import warnings
    # TensorFlowの警告を無視する
    warnings.filterwarnings("ignore", category=UserWarning, module="tensorflow")
    import warnings
    warnings.filterwarnings('ignore', category=UserWarning)

    input_dim = X_train.shape[1]
    # 隠れ層の次元数を設定
    import math
    temp = math.sqrt(input_dim * latent_dim)
    Dense_dim = math.ceil(temp / 2) * 2 if temp % 2 != 0 else temp
    # print('CHECK')
    # if input_dim >= 12:
    #     Dense_dim = int(input_dim/6)
    # else:
    #     Dense_dim = 4
  
    if auto_encoder=='AE':
        # オートエンコーダのモデルの構築
        input_layer = Input(shape=(input_dim,))
        encoded = Dense(latent_dim, activation='tanh')(input_layer)
        decoded = Dense(input_dim, activation='tanh')(encoded)
        autoencoder = Model(input_layer, decoded)
        autoencoder.compile(optimizer=Adam(), loss='mean_squared_error')

        # 検証用データセットの作成
        from sklearn.model_selection import train_test_split
        X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.1, random_state=42)
        
        # モデルの学習
        history = autoencoder.fit(X_train, X_train, epochs=epochs, batch_size=16, shuffle=True, validation_data=(X_valid, X_valid), verbose=verbose)

        # モデルの保存
        # エンコーダとデコーダのモデルを保存
        encoder = Model(input_layer, encoded)
        encoder.save(f'save_data/auto_encoder_model/model/{auto_encoder}_{dataset}_encoder_dim{latent_dim}_epoch{epochs}.keras')

        decoder_input = Input(shape=(latent_dim,))
        decoder_output = autoencoder.layers[2](decoder_input)
        decoder = Model(decoder_input, decoder_output)
        decoder.save(f'save_data/auto_encoder_model/model/{auto_encoder}_{dataset}_decoder_dim{latent_dim}_epoch{epochs}.keras')

        # テストデータの再構成
        X_test_reconstructed = autoencoder.predict(X_test, verbose=verbose)

        # 再構成誤差の計算
        reconstruction_mse = np.mean(np.power(X_test - X_test_reconstructed, 2), axis=1)

        # 再構成誤差の出力
        # print('Reconstruction error:', np.mean(reconstruction_mse))

        # 学習曲線のプロット
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'learning_curve({auto_encoder}_{dataset}_dim{latent_dim}_epoch{epochs})')
        plt.legend()
        plt.savefig(f'save_data/auto_encoder_model/learning_curve/learning_curve_{auto_encoder}_{dataset}_dim{latent_dim}_epoch{epochs}.png')
        plt.clf()
        
    if auto_encoder == 'VAE':
        def sampling(args):
            """Reparameterization trickを使用して標準的なガウス分布からサンプリングする。
            """
            z_mean, z_log_var = args
            batch = K.shape(z_mean)[0]
            dim = K.int_shape(z_mean)[1]
            epsilon = K.random_normal(shape=(batch, dim))
            return z_mean + K.exp(0.5 * z_log_var) * epsilon
        
        input_dim = X_train.shape[1]

        # VAEモデルのアーキテクチャ
        inputs = Input(shape=(input_dim,))
        x = Dense(Dense_dim, activation='tanh')(inputs)
        z_mean = Dense(latent_dim, name='z_mean')(x)
        z_log_var = Dense(latent_dim, name='z_log_var')(x)

        # Reparameterization trick
        z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

        encoder = Model(inputs, [z_mean, z_log_var, z])

        decoder_h = Dense(Dense_dim, activation='tanh')
        decoder_mean = Dense(input_dim, activation='tanh')
        h_decoded = decoder_h(z)
        x_decoded_mean = decoder_mean(h_decoded)
        
        vae = Model(inputs, x_decoded_mean)

        kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        
        # バイナリクロスエントロピーを使用した場合
        # xent_loss = input_dim * binary_crossentropy(inputs, x_decoded_mean)
        # vae_loss = K.mean(xent_loss + kl_loss)

        # MSEを使用した場合        
        mse_loss = mean_squared_error(inputs, x_decoded_mean) * input_dim
        vae_loss = K.mean(mse_loss + kl_loss)

        vae.add_loss(vae_loss)
        vae.compile(optimizer=Adam())
        
        from sklearn.model_selection import train_test_split
        X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

        history = vae.fit(X_train, None, shuffle=True, epochs=epochs, batch_size=16, validation_data=(X_valid, None), verbose=verbose)

        encoder.save(f'save_data/auto_encoder_model/model/{auto_encoder}_{dataset}_encoder_dim{latent_dim}_epoch{epochs}.keras')

        decoder_input = Input(shape=(latent_dim,))
        _h_decoded = decoder_h(decoder_input)
        _x_decoded_mean = decoder_mean(_h_decoded)
        decoder = Model(decoder_input, _x_decoded_mean)
        decoder.save(f'save_data/auto_encoder_model/model/{auto_encoder}_{dataset}_decoder_dim{latent_dim}_epoch{epochs}.keras')
        
        X_test_reconstructed = vae.predict(X_test, verbose=verbose)
        reconstruction_mse = np.mean(np.power(X_test - X_test_reconstructed, 2), axis=1)
        # print('Reconstruction error:', np.mean(reconstruction_mse))
        
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'learning_curve({auto_encoder}_{dataset}_dim{latent_dim}_epoch{epochs})')
        plt.legend()
        plt.savefig(f'save_data/auto_encoder_model/learning_curve/learning_curve_{auto_encoder}_{dataset}_dim{latent_dim}_epoch{epochs}.png')
        plt.clf()
        
    if auto_encoder == 'CVAE':
        def sampling(args):
            """Reparameterization trickを使用して標準的なガウス分布からサンプリングする。
            """
            z_mean, z_log_var = args
            batch = K.shape(z_mean)[0]
            dim = K.int_shape(z_mean)[1]
            epsilon = K.random_normal(shape=(batch, dim))
            return z_mean + K.exp(0.5 * z_log_var) * epsilon
        
        #　条件付けに関する設定
        input_dim = X_train.shape[1]
        if one_hot_encoding == True:
            condition_dim = [4 if dataset_class_num == 'numerous' else dataset_class_num][0]
        else:
            condition_dim = 1
        
        # CVAEモデルのアーキテクチャ
        # 入力と条件を結合
        inputs = Input(shape=(input_dim,))
        condition_input = Input(shape=(condition_dim,))
        combined_input = concatenate([inputs, condition_input])
        
        x = Dense(int(Dense_dim), activation='tanh')(combined_input)
        z_mean = Dense(latent_dim, name='z_mean')(x)
        z_log_var = Dense(latent_dim, name='z_log_var')(x)

        # Reparameterization trick
        z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])
        # CVAEの変更点
        z_cond = concatenate([z, condition_input])

        encoder = Model([inputs, condition_input], [z_mean, z_log_var, z])

        decoder_h = Dense(int(Dense_dim), activation='tanh')
        decoder_mean = Dense(input_dim, activation='tanh')
        h_decoded = decoder_h(z_cond)
        x_decoded_mean = decoder_mean(h_decoded)
        
        vae = Model([inputs, condition_input], x_decoded_mean)

        kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        
        # バイナリクロスエントロピーを使用した場合
        # xent_loss = input_dim * binary_crossentropy(inputs, x_decoded_mean)
        # vae_loss = K.mean(xent_loss + kl_loss)

        # MSEを使用した場合        
        mse_loss = mean_squared_error(inputs, x_decoded_mean) * input_dim
        vae_loss = K.mean(mse_loss + kl_loss)
        
        vae.add_loss(vae_loss)
        vae.compile(optimizer=Adam())
        
        from sklearn.model_selection import train_test_split
        X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

        if one_hot_encoding == True:
            # y_trainが量的変数の場合は量子化する
            from functions import quantize_matrix
            if dataset_class_num == 'numerous':
                y_train = quantize_matrix(y_train)
                y_valid = quantize_matrix(y_valid)
                y_test = quantize_matrix(y_test)
                # ラベルをone-hotエンコーディングする
                from tensorflow.keras.utils import to_categorical
                y_train = to_categorical(y_train, num_classes=int(4))
                y_valid = to_categorical(y_valid, num_classes=int(4))
                y_test = to_categorical(y_test, num_classes=int(4))
            else:    
                # ラベルをone-hotエンコーディングする
                from tensorflow.keras.utils import to_categorical
                y_train = to_categorical(y_train, num_classes=int(dataset_class_num))
                y_valid = to_categorical(y_valid, num_classes=int(dataset_class_num))
                y_test = to_categorical(y_test, num_classes=int(dataset_class_num))
            
        
        history = vae.fit([X_train, y_train], None, shuffle=True, epochs=epochs, batch_size=16, validation_data=([X_valid, y_valid], None), verbose=verbose)

        encoder.save(f'save_data/auto_encoder_model/model/{auto_encoder}_{dataset}_encoder_dim{latent_dim}_epoch{epochs}.keras')

        decoder_input_dim = latent_dim + condition_dim
        decoder_input = Input(shape=(decoder_input_dim,))
        _h_decoded = decoder_h(decoder_input)
        _x_decoded_mean = decoder_mean(_h_decoded)
        decoder = Model(decoder_input, _x_decoded_mean)
        decoder.save(f'save_data/auto_encoder_model/model/{auto_encoder}_{dataset}_decoder_dim{latent_dim}_epoch{epochs}.keras')
        
        X_test_reconstructed = vae.predict([X_test, y_test], verbose=verbose)
        reconstruction_mse = np.mean(np.power(X_test - X_test_reconstructed, 2), axis=1)
        # print('Reconstruction error:', np.mean(reconstruction_mse))
        
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'learning_curve({auto_encoder}_{dataset}_dim{latent_dim}_epoch{epochs})')
        plt.legend()
        plt.savefig(f'save_data/auto_encoder_model/learning_curve/learning_curve_{auto_encoder}_{dataset}_dim{latent_dim}_epoch{epochs}.png')
        plt.clf()


    if auto_encoder == 'ICVAE':
        def sampling(args):
            """Reparameterization trickを使用して標準的なガウス分布からサンプリングする。
            """
            z_mean, z_log_var = args
            batch = K.shape(z_mean)[0]
            dim = K.int_shape(z_mean)[1]
            epsilon = K.random_normal(shape=(batch, dim))
            return z_mean + K.exp(0.5 * z_log_var) * epsilon
        
        #　条件付けに関する設定
        input_dim = X_train.shape[1]
        if one_hot_encoding == True:
            condition_dim = [4 if dataset_class_num == 'numerous' else dataset_class_num][0]
        else:
            condition_dim = 1
        
        # CVAEモデルのアーキテクチャ
        # 入力と条件を結合
        inputs = Input(shape=(input_dim,))
        condition_input = Input(shape=(condition_dim,))
        # combined_input = concatenate([inputs, condition_input]) #CVAE
        
        # x = Dense(int(Dense_dim), activation='tanh')(combined_input) #CVAE
        x = Dense(int(Dense_dim), activation='tanh')(inputs)
        z_mean = Dense(latent_dim, name='z_mean')(x)
        z_log_var = Dense(latent_dim, name='z_log_var')(x)

        # Reparameterization trick
        z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])
        # CVAEの変更点
        z_cond = concatenate([z, condition_input])

        # encoder = Model([inputs, condition_input], [z_mean, z_log_var, z]) #CVAE
        encoder = Model([inputs], [z_mean, z_log_var, z])

        decoder_h = Dense(int(Dense_dim), activation='tanh')
        decoder_mean = Dense(input_dim, activation='tanh')
        h_decoded = decoder_h(z_cond)
        x_decoded_mean = decoder_mean(h_decoded)
        
        vae = Model([inputs, condition_input], x_decoded_mean)

        kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        
        # バイナリクロスエントロピーを使用した場合
        # xent_loss = input_dim * binary_crossentropy(inputs, x_decoded_mean)
        # vae_loss = K.mean(xent_loss + kl_loss)

        # MSEを使用した場合        
        mse_loss = mean_squared_error(inputs, x_decoded_mean) * input_dim
        vae_loss = K.mean(mse_loss + kl_loss)
        
        vae.add_loss(vae_loss)
        vae.compile(optimizer=Adam())
        
        from sklearn.model_selection import train_test_split
        X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

        if one_hot_encoding == True:
            # y_trainが量的変数の場合は量子化する
            from functions import quantize_matrix
            if dataset_class_num == 'numerous':
                y_train = quantize_matrix(y_train)
                y_valid = quantize_matrix(y_valid)
                y_test = quantize_matrix(y_test)
                # ラベルをone-hotエンコーディングする
                from tensorflow.keras.utils import to_categorical
                y_train = to_categorical(y_train, num_classes=int(4))
                y_valid = to_categorical(y_valid, num_classes=int(4))
                y_test = to_categorical(y_test, num_classes=int(4))
            else:    
                # ラベルをone-hotエンコーディングする
                from tensorflow.keras.utils import to_categorical
                y_train = to_categorical(y_train, num_classes=int(dataset_class_num))
                y_valid = to_categorical(y_valid, num_classes=int(dataset_class_num))
                y_test = to_categorical(y_test, num_classes=int(dataset_class_num))
            
        
        history = vae.fit([X_train, y_train], None, shuffle=True, epochs=epochs, batch_size=16, validation_data=([X_valid, y_valid], None), verbose=verbose)

        encoder.save(f'save_data/auto_encoder_model/model/{auto_encoder}_{dataset}_encoder_dim{latent_dim}_epoch{epochs}.keras')

        decoder_input_dim = latent_dim + condition_dim
        decoder_input = Input(shape=(decoder_input_dim,))
        _h_decoded = decoder_h(decoder_input)
        _x_decoded_mean = decoder_mean(_h_decoded)
        decoder = Model(decoder_input, _x_decoded_mean)
        decoder.save(f'save_data/auto_encoder_model/model/{auto_encoder}_{dataset}_decoder_dim{latent_dim}_epoch{epochs}.keras')
        
        X_test_reconstructed = vae.predict([X_test, y_test],verbose=verbose)
        reconstruction_mse = np.mean(np.power(X_test - X_test_reconstructed, 2), axis=1)
        # print('Reconstruction error:', np.mean(reconstruction_mse))
        
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'learning_curve({auto_encoder}_{dataset}_dim{latent_dim}_epoch{epochs})')
        plt.legend()
        plt.savefig(f'save_data/auto_encoder_model/learning_curve/learning_curve_{auto_encoder}_{dataset}_dim{latent_dim}_epoch{epochs}.png')
        plt.clf()
        
    if auto_encoder == 'ICVAE2':
        def sampling(args):
            """Reparameterization trickを使用して標準的なガウス分布からサンプリングする。
            """
            z_mean, z_log_var = args
            batch = K.shape(z_mean)[0]
            dim = K.int_shape(z_mean)[1]
            epsilon = K.random_normal(shape=(batch, dim))
            return z_mean + K.exp(0.5 * z_log_var) * epsilon
        
        #　条件付けに関する設定
        input_dim = X_train.shape[1]
        if one_hot_encoding == True:
            condition_dim = [4 if dataset_class_num == 'numerous' else dataset_class_num][0]
        else:
            condition_dim = 1
        
        # CVAEモデルのアーキテクチャ
        # 入力と条件を結合
        inputs = Input(shape=(input_dim,))
        condition_input = Input(shape=(condition_dim,))
        combined_input = concatenate([inputs, condition_input]) #CVAE
        
        x = Dense(int(Dense_dim), activation='tanh')(combined_input) #CVAE
        # x = Dense(int(Dense_dim), activation='tanh')(inputs) #ICVAE
        z_mean = Dense(latent_dim, name='z_mean')(x)
        z_log_var = Dense(latent_dim, name='z_log_var')(x)

        # Reparameterization trick
        z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])
        # CVAEの変更点
        # z_cond = concatenate([z, condition_input])

        encoder = Model([inputs, condition_input], [z_mean, z_log_var, z]) #CVAE
        # encoder = Model([inputs], [z_mean, z_log_var, z])

        decoder_h = Dense(int(Dense_dim), activation='tanh')
        decoder_mean = Dense(input_dim, activation='tanh')
        # h_decoded = decoder_h(z_cond) #CVAE
        h_decoded = decoder_h(z) #CVAE
        x_decoded_mean = decoder_mean(h_decoded)
        
        vae = Model([inputs, condition_input], x_decoded_mean)

        kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        
        # バイナリクロスエントロピーを使用した場合
        # xent_loss = input_dim * binary_crossentropy(inputs, x_decoded_mean)
        # vae_loss = K.mean(xent_loss + kl_loss)

        # MSEを使用した場合        
        mse_loss = mean_squared_error(inputs, x_decoded_mean) * input_dim
        vae_loss = K.mean(mse_loss + kl_loss)
        
        vae.add_loss(vae_loss)
        vae.compile(optimizer=Adam())
        
        from sklearn.model_selection import train_test_split
        X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

        if one_hot_encoding == True:
            # y_trainが量的変数の場合は量子化する
            from functions import quantize_matrix
            if dataset_class_num == 'numerous':
                y_train = quantize_matrix(y_train)
                y_valid = quantize_matrix(y_valid)
                y_test = quantize_matrix(y_test)
                # ラベルをone-hotエンコーディングする
                from tensorflow.keras.utils import to_categorical
                y_train = to_categorical(y_train, num_classes=int(4))
                y_valid = to_categorical(y_valid, num_classes=int(4))
                y_test = to_categorical(y_test, num_classes=int(4))
            else:    
                # ラベルをone-hotエンコーディングする
                from tensorflow.keras.utils import to_categorical
                y_train = to_categorical(y_train, num_classes=int(dataset_class_num))
                y_valid = to_categorical(y_valid, num_classes=int(dataset_class_num))
                y_test = to_categorical(y_test, num_classes=int(dataset_class_num))
            
        
        history = vae.fit([X_train, y_train], None, shuffle=True, epochs=epochs, batch_size=16, validation_data=([X_valid, y_valid], None), verbose=verbose)

        encoder.save(f'save_data/auto_encoder_model/model/{auto_encoder}_{dataset}_encoder_dim{latent_dim}_epoch{epochs}.keras')

        decoder_input_dim = latent_dim
        decoder_input = Input(shape=(decoder_input_dim,))
        _h_decoded = decoder_h(decoder_input)
        _x_decoded_mean = decoder_mean(_h_decoded)
        decoder = Model(decoder_input, _x_decoded_mean)
        decoder.save(f'save_data/auto_encoder_model/model/{auto_encoder}_{dataset}_decoder_dim{latent_dim}_epoch{epochs}.keras')
        
        X_test_reconstructed = vae.predict([X_test, y_test],verbose=verbose)
        reconstruction_mse = np.mean(np.power(X_test - X_test_reconstructed, 2), axis=1)
        # print('Reconstruction error:', np.mean(reconstruction_mse))
        
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'learning_curve({auto_encoder}_{dataset}_dim{latent_dim}_epoch{epochs})')
        plt.legend()
        plt.savefig(f'save_data/auto_encoder_model/learning_curve/learning_curve_{auto_encoder}_{dataset}_dim{latent_dim}_epoch{epochs}.png')
        plt.clf()

def AE_load(X_test=None,
        inverse=None,
        X_test_predict=None,
        predict_fn=None,
        epochs=None,
        latent_dim=None,
        dataset=None,
        auto_encoder=None,
        num_samples=None,
        instance_no=None,
        auto_encoder_weighting=None,
        auto_encoder_sampling=None,
        one_hot_encoding=None,
        noise_std=None,
        kernel_width=None,
        VAR_threshold=None
        ):
    '''
    X_test:説明対象のインスタンス
    inverse:LIMEが生成したサンプル
    X_test_predict:説明対象のモデルの説明対象のインスタンスの出力値
    epochs:エポック数
    latent_dim:潜在空間次元数
    dataset:データセット名
    auto_encoder:オートエンコーダ名
    num_samples:サンプル数
    instance_no:インスタンス番号
    auto_encoder_weighting:boolean
    auto_encoder_sampling:boolean
    one_hot_encoding:boolean
    '''
    import math
    
    # モデルの読み込み
    encoder = load_model(f'save_data/auto_encoder_model/model/{auto_encoder}_{dataset}_encoder_dim{latent_dim}_epoch{epochs}.keras',safe_mode=False)
    decoder = load_model(f'save_data/auto_encoder_model/model/{auto_encoder}_{dataset}_decoder_dim{latent_dim}_epoch{epochs}.keras',safe_mode=False)
    
    # 重み付けのみの場合
    if auto_encoder_weighting == True and auto_encoder_sampling == False:
        # samplesの定義
        samples = inverse
        # 潜在変数の取得
        latent_vector = encoder.predict(X_test.reshape(1, -1),verbose=0)
        latent_vectors = encoder.predict(inverse,verbose=0)
        # 距離の計算
        distances = np.linalg.norm(latent_vectors - latent_vector, axis=1)
        # 重みの定義
        weights = np.power(1/ math.e, distances)
        # weights = np.sqrt(np.exp(-(distances ** 2) / kernel_width ** 2))
        
        
    # 重み付けとサンプル生成を行う場合
    elif auto_encoder_sampling == True:
        # 潜在変数の取得
        # CVAEの場合
        if auto_encoder == 'CVAE' or auto_encoder == 'ICVAE2':
            if one_hot_encoding == True:
                latent_vector, latent_var, _ = encoder.predict([X_test.reshape(1, len(X_test)), X_test_predict],verbose=0) 
                # # 入力とクラスを結合
                # combined_input = concatenate([X_test.reshape(1, len(X_test)), X_test_predict])
                # latent_vector = encoder.predict(combined_input)
            else:
                import tensorflow as tf
                # X_test_predict = np.full((1, 30), X_test_predict)
                # X_test_predict = tf.cast(X_test_predict, dtype=tf.float32)
                X_test_predict = X_test_predict.astype(float)
                latent_vector, latent_var, _ = encoder.predict([X_test.reshape(1, len(X_test)), X_test_predict.reshape(1, -1)],verbose=0)
        # CVAE以外の場合
        elif auto_encoder == 'AE':
            latent_vector = encoder.predict(X_test.reshape(1, len(X_test)),verbose=0)
        else:
            latent_vector, latent_var, _ = encoder.predict(X_test.reshape(1, len(X_test)),verbose=0)
        
        # 活性潜在変数の計算
        from functions import count_below_threshold
        VAR = np.exp(latent_var)
        Active_latent_dim = count_below_threshold(VAR, VAR_threshold)
        
        # (VAE,CVAE対策)潜在変数がnumpy形式で無い場合はnumpyに統一.平均ベクトルを返す(encoderを定義するときに返り値を定義しなおせばいいかもしれない)
        if not isinstance(latent_vector, np.ndarray):
            latent_vector = np.array(latent_vector)[1]
        # 5000個のノイズを加えた潜在ベクトルを作成
        noise = np.random.normal(loc=0, scale=noise_std, size=(num_samples, *latent_vector.shape))
        latent_vectors = np.squeeze(latent_vector + noise)
        # samplesを定義
        # CVAEの場合
        if auto_encoder == 'CVAE' or auto_encoder == 'ICVAE':
            # 各潜在変数にクラスを付与
            expanded_X_test_predict = np.tile(X_test_predict, (latent_vectors.shape[0], 1))
            if auto_encoder == 'ICVAE':
                expanded_X_test_predict = expanded_X_test_predict.astype('float32')
            # 潜在変数とクラスを結合
            z_cond = concatenate([latent_vectors, expanded_X_test_predict])
            # 結合されたデータをデコード
            # samples = decoder.predict([latent_vectors, expanded_X_test_predict]) # サンプルがずれる
            if auto_encoder == 'ICVAE2':
                z_cond = latent_vectors
            samples = decoder.predict(z_cond, verbose=0) #202309051408修正
        # CVAE以外の場合
        else:
            samples = decoder.predict(latent_vectors, verbose=0)
        
        # if auto_encoder == 'VAE':
        #     # 各行間のGower距離を計算
        #     import gower
        #     distances = gower.gower_matrix(data_x=latent_vectors, data_y=latent_vector)
        #     # 1から距離を引く
        #     distances = 1 - distances
        #     # distancesが2次元であれば、1次元にリシェープする
        #     if len(distances.shape) == 2 and distances.shape[1] == 1:
        #         distances = distances.ravel()    
        # else:
        # 距離の計算
        distances = np.linalg.norm(latent_vectors - latent_vector, axis=1)
            
        # 重みの定義
        weights = np.power(1/ math.e, distances)
        # weights = np.sqrt(np.exp(-(distances ** 2) / kernel_width ** 2))
        
        
    if auto_encoder_weighting == False:
        # import sys
        # import warnings
        # warnings.warn('This is not supported now.(DAISUKE YASUI)')
        # sys.exit()
        weights = np.ones_like(weights)
        
        
    # モデルの出力値の定義
    labels = predict_fn(samples)
            
    
    # labelsの1列目をheatmap用の色情報として取得
    if dataset == 'boston':
        color_map = labels
    else:
        color_map = labels[:, 0]

    # プロット
    plt.scatter(latent_vectors[:, 0], latent_vectors[:, 1], c=color_map, label='Noisy Sample', s=2 * weights, cmap='viridis') # color_mapをc引数に追加
    plt.scatter(latent_vector[:, 0], latent_vector[:, 1], c='b', label='Test Sample', s=100)
    plt.colorbar().set_label('Value from labels')  # カラーバーを追加して色情報を示す
    plt.xlabel('Latent X')
    plt.ylabel('Latent Y')
    plt.title(f'latent_space({auto_encoder}_{dataset}_epoch{epochs}_instance{instance_no})')
    plt.legend()
    # X軸とY軸の範囲を-5から5に設定
    plt.xlim([-5, 5])
    plt.ylim([-5, 5])
    plt.savefig(f'save_data/auto_encoder_model/latent_space/latent_space_{auto_encoder}_{dataset}_epoch{epochs}_instance{instance_no}.png',dpi=1000)
    plt.clf()
    
    #　カーネル幅に合わせて上位n%のサンプルを選別    
    # if auto_encoder == 'AE':
    #     from functions import extract_top_n_percent
    #     samples, weights, labels = extract_top_n_percent(samples, weights, labels, kernel_width)
        
    return samples, weights, labels, Active_latent_dim

if __name__ == '__main__':
    # データの読み込み
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    X = df.values

    # データの正規化
    scaler = MinMaxScaler()
    X_normalized = scaler.fit_transform(X)

    # データの分割
    X_train, X_test = train_test_split(X_normalized, test_size=0.2, random_state=42)

    import tensorflow as tf
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    #　テストインスタンスの定義
    x_test_sample = X_test[0].reshape(1, -1)
    inverse = np.tile(x_test_sample, (5000, 1))
    
    AE_training(X_train=X_train,
            y_train=X_train,
            X_test=x_test_sample,
            epochs=100,
            latent_dim=2,
            dataset='breastcancer',
            auto_encoder='AE'
            )
    
    
    AE_load(X_test=x_test_sample,
            inverse=inverse,
            epochs=100,
            latent_dim=2,
            dataset='breastcancer',
            auto_encoder='AE',
            num_samples=5000,
            instance_no=1,
            auto_encoder_weighting=True,
            auto_encoder_sampling=False
            )
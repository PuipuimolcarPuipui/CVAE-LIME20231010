import os
from PIL import Image

def create_gif_from_pngs(folder_path, keyword, output_gif_name, duration=2000):
    """
    特定のフォルダ内の特定の文字列を含むpng画像をスライドショーとしてgifにする関数
    :param folder_path: 画像が保存されているフォルダのパス
    :param keyword: 画像名に含まれるべき文字列
    :param output_gif_name: 出力するgifのファイル名
    :param duration: 各画像が表示される時間（ミリ秒）
    """
    
    # 特定の文字列を含むpngファイルをリストアップ
    images = [f for f in os.listdir(folder_path) if f.endswith('.png') and keyword in f]
    images.sort()  # アルファベット順に並べ替え
    
    if not images:
        print("該当する画像が見つかりませんでした。")
        return
    
    # 画像をオープンしてリストに格納
    img_list = [Image.open(os.path.join(folder_path, img)) for img in images]
    
    # GIFを作成
    img_list[0].save(
        os.path.join(folder_path, output_gif_name),
        append_images=img_list[1:],
        save_all=True,
        duration=duration,
        loop=0
    )
    print(f"'{output_gif_name}'として保存されました。")

if __name__ == "__main__":
    folder = "MNIST_png"
    import itertools
    p = itertools.product(['VAE','CVAE'], ['NN','RF','SVM'], range(10))
    for AE, Clf, No in p:
      keyword = f"combined_{AE}_{Clf}_{No}"
      output_name = f"{AE}_{Clf}_{No}.gif"
      create_gif_from_pngs(folder, keyword, output_name)
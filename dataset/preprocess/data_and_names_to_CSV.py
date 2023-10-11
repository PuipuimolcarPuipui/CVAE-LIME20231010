import pandas as pd

# .namesファイルからカラム名を読み込む関数
def load_column_names(names_file):
    with open(names_file, 'r') as f:
        lines = f.readlines()
    # カラム名のリストを作成
    column_names = [line.split(':')[0] for line in lines if not line.startswith('|')]
    return column_names

# .dataファイルからデータを読み込む関数
def load_data(data_file):
    data = pd.read_csv(data_file, header=None)
    return data

# メイン関数
def main():
    # .namesファイルと.dataファイルのパス
    folder = 'mushroom'
    names_file = folder + '/' + folder + '.names'
    data_file = folder + '/' + folder + '.data'
    test_file = folder + '/' + folder + '.test'
    
    # Train
    # カラム名とデータを読み込む
    column_names = load_column_names(names_file)
    data = load_data(data_file)
    
    # カラム名を設定
    data.columns = column_names
    
    # CSVファイルに保存
    output_file = 'output_data.csv'
    data.to_csv(output_file, index=False)
    print(f'Data saved to {output_file}')

    # Test
    # カラム名とデータを読み込む
    column_names = load_column_names(names_file)
    data = load_data(test_file)
    
    # カラム名を設定
    data.columns = column_names
    
    # CSVファイルに保存
    output_file = 'output_test.csv'
    data.to_csv(output_file, index=False)
    print(f'Data saved to {output_file}')
    
# プログラムを実行
if __name__ == '__main__':
    main()

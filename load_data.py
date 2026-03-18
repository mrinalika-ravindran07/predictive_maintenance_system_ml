import pandas as pd

def load_train_data(path):
    cols = ["engine_id", "cycle"] + [f"op_setting{i}" for i in range(1, 4)] + [f"sensor{i}" for i in range(1, 22)]
    
    df = pd.read_csv(path, sep="\s+", header=None)
    df.columns = cols
    
    return df

if __name__ == "__main__":
    train = load_train_data("E:/Mrinalika/Zaalima Internship/dataset/train_FD001.txt")
    print(train.head())
    print(train.shape)

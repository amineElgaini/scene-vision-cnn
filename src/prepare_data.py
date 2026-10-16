import hashlib
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

DATA_DIR = Path("data/seg_train_test/seg_train")
OUT_DIR = Path("data/splits")
SEED = 42


def build_dataframe(data_dir=DATA_DIR):
    rows = []
    for class_dir in sorted(data_dir.iterdir()):
        if class_dir.is_dir():
            for img in sorted(class_dir.glob("*")):
                rows.append({"path": str(img), "label": class_dir.name})
    return pd.DataFrame(rows)


def file_hash(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def remove_duplicates(df):
    df = df.copy()
    df["hash"] = df["path"].apply(file_hash)
    n_before = len(df)
    df = df.drop_duplicates("hash").drop(columns="hash").reset_index(drop=True)
    print(f"Duplicates removed: {n_before - len(df)}")
    return df


def add_label_ids(df):
    class_names = sorted(df["label"].unique())
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    df["label_id"] = df["label"].map(class_to_idx)
    return df


def split_data(df):
    df_train, df_temp = train_test_split(
        df, test_size=0.30, stratify=df["label"], random_state=SEED
    )
    df_val, df_test = train_test_split(
        df_temp, test_size=0.50, stratify=df_temp["label"], random_state=SEED
    )
    return df_train, df_val, df_test


def prepare():
    df = build_dataframe()
    print("Images found:", len(df))
    df = remove_duplicates(df)
    df = add_label_ids(df)
    df_train, df_val, df_test = split_data(df)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df_train.to_csv(OUT_DIR / "train.csv", index=False)
    df_val.to_csv(OUT_DIR / "val.csv", index=False)
    df_test.to_csv(OUT_DIR / "test.csv", index=False)
    print("Train / Val / Test:", len(df_train), len(df_val), len(df_test))


if __name__ == "__main__":
    prepare()
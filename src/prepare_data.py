#%% Import libraries
import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split

#%% Config
clean_data_dir = os.path.join(os.path.dirname(__file__), "..", "clean-data")
train_dir = os.path.join(clean_data_dir, "test")    # larger set → train
val_test_dir = os.path.join(clean_data_dir, "train") # smaller set → val + test
output_dir = os.path.join(os.path.dirname(__file__), "..", "vnct")

#%% Functions
def load_category_files(directory):
    records = []
    categories = sorted(f.replace(".txt", "") for f in os.listdir(directory) if f.endswith(".txt"))
    label_map = {cat: idx for idx, cat in enumerate(categories)}

    for cat in categories:
        filepath = os.path.join(directory, f"{cat}.txt")
        with open(filepath, "r", encoding="utf-8") as fh:
            lines = [ln.strip() for ln in fh if ln.strip()]
        for text in lines:
            records.append({"text": text, "label_name": cat, "label": label_map[cat]})
    return pd.DataFrame(records), label_map


def split_val_test(df, seed=36):
    val_df, test_df = train_test_split(
        df,
        test_size=0.5,
        stratify=df["label"],
        random_state=seed,
    )
    return val_df.reset_index(drop=True), test_df.reset_index(drop=True)


#%% Run
def main():
    os.makedirs(output_dir, exist_ok=True)

    print("Loading train set (clean-data/test/) …")
    train_df, label_map = load_category_files(train_dir)

    print("Loading val+test source (clean-data/train/) and splitting 50/50 …")
    val_test_df, _ = load_category_files(val_test_dir)
    val_df, test_df = split_val_test(val_test_df)

    print(f"\nFinal Distribution:")
    print(f"  Train: {len(train_df):,}")
    print(f"  Val:   {len(val_df):,}")
    print(f"  Test:  {len(test_df):,}")

    print(f"\n{'Category':<25} {'Train':>6} {'Val':>6} {'Test':>6}")
    print("-" * 50)
    for cat, idx in label_map.items():
        tr = (train_df["label"] == idx).sum()
        va = (val_df["label"] == idx).sum()
        te = (test_df["label"] == idx).sum()
        print(f"{cat:<25} {tr:>6} {va:>6} {te:>6}")

    label_map_path = os.path.join(output_dir, "label_map.json")
    with open(label_map_path, "w", encoding="utf-8") as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)
    print(f"\nLabel map saved → {label_map_path}")

    return train_df, val_df, test_df, label_map

if __name__ == "__main__":
    main()
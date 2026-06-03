#%% Import libraries
import os
import pandas as pd

#%% Config
clean_data_dir = os.path.join(os.path.dirname(__file__), "..", "clean-data")
train_dir    = os.path.join(clean_data_dir, "train")
val_dir      = os.path.join(clean_data_dir, "val")
test_dir     = os.path.join(clean_data_dir, "test")

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


#%% Run
def main():
    train_df, label_map = load_category_files(train_dir)
    val_df,   _         = load_category_files(val_dir)
    test_df,  _         = load_category_files(test_dir)

    print(f"\nFinal Distribution:")
    print(f"  Train: {len(train_df):,}")
    print(f"  Val:   {len(val_df):,}")
    print(f"  Test:  {len(test_df):,}")

    print(f"\n{'Category':<25} {'Train':>6} {'Val':>6} {'Test':>6}")
    print("-" * 50)
    for cat, idx in label_map.items():
        tr = (train_df["label"] == idx).sum()
        va = (val_df["label"]   == idx).sum()
        te = (test_df["label"]  == idx).sum()
        print(f"{cat:<25} {tr:>6} {va:>6} {te:>6}")

    return train_df, val_df, test_df, label_map


if __name__ == "__main__":
    main()
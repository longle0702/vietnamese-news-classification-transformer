import os
import random
import unicodedata
import regex as re
from pyvi import ViTokenizer

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW_DIRS     = [
    os.path.join(PROJECT_ROOT, "Train_Full"),
    os.path.join(PROJECT_ROOT, "Test_Full"),
]
CLEAN_DIR    = os.path.join(PROJECT_ROOT, "clean-data")
SEED         = 36
TRAIN_RATIO  = 0.70
VAL_RATIO    = 0.15
# test gets the remainder (0.15)

def normalize_unicode(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def segment_vietnamese_words(text: str) -> str:
    return ViTokenizer.tokenize(text)


def convert_to_lowercase(text: str) -> str:
    return text.lower()


def clean_text(text: str) -> str:
    text = re.sub(r'\b\d+\b', ' ', text)
    text = re.sub(
        r'[^\s\wáàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệóòỏõọôốồổỗộơớờởỡợíìỉĩịúùủũụưứừửữựýỳỷỹỵđ_]',
        ' ',
        text,
    )
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def process_vietnamese_text(text: str) -> str:
    text = normalize_unicode(text)
    text = segment_vietnamese_words(text)
    text = convert_to_lowercase(text)
    text = clean_text(text)
    return text


def read_raw_category(category_path: str) -> list[str]:
    texts = []
    for fname in os.listdir(category_path):
        fpath = os.path.join(category_path, fname)
        with open(fpath, "r", encoding="utf-16") as f:
            raw = f.read()
        texts.append(process_vietnamese_text(raw))
    return texts


def write_clean_split(texts: list[str], split: str, category: str) -> None:
    out_path = os.path.join(CLEAN_DIR, split, f"{category}.txt")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for text in texts:
            f.write(text + "\n")


def main():
    random.seed(SEED)
    all_categories: set[str] = set()
    for raw_dir in RAW_DIRS:
        for name in os.listdir(raw_dir):
            if os.path.isdir(os.path.join(raw_dir, name)):
                all_categories.add(name)

    total_train = total_val = total_test = 0

    print(f"Splitting with ratio  train={TRAIN_RATIO:.0%}  val={VAL_RATIO:.0%}  test={1-TRAIN_RATIO-VAL_RATIO:.0%}")
    print(f"{'Category':<25} {'Total':>6} {'Train':>6} {'Val':>6} {'Test':>6}")
    print("-" * 55)

    for category in sorted(all_categories):
        texts: list[str] = []
        for raw_dir in RAW_DIRS:
            cat_path = os.path.join(raw_dir, category)
            if os.path.isdir(cat_path):
                texts.extend(read_raw_category(cat_path))

        random.shuffle(texts)

        n = len(texts)
        n_train = round(n * TRAIN_RATIO)
        n_val   = round(n * VAL_RATIO)

        train_texts = texts[:n_train]
        val_texts   = texts[n_train : n_train + n_val]
        test_texts  = texts[n_train + n_val:]

        write_clean_split(train_texts, "train", category)
        write_clean_split(val_texts,   "val",   category)
        write_clean_split(test_texts,  "test",  category)

        total_train += len(train_texts)
        total_val   += len(val_texts)
        total_test  += len(test_texts)

        print(f"  {category:<23} {n:>6} {len(train_texts):>6} {len(val_texts):>6} {len(test_texts):>6}")

    print("-" * 55)
    print(f"  {'TOTAL':<23} {total_train+total_val+total_test:>6} {total_train:>6} {total_val:>6} {total_test:>6}")
    print(f"\nDone. Clean data written to: {CLEAN_DIR}")


if __name__ == "__main__":
    main()

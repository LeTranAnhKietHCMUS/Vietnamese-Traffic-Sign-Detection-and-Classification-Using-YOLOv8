"""
=============================================================
  STRATIFIED SPLIT DATASET (YOLO format)
  Phân chia dataset theo tỉ lệ Train / Validation / Test
  đảm bảo TẤT CẢ classes có mặt đầy đủ trong cả 3 tập.
=============================================================

Phương pháp Stratified Split được áp dụng thay vì phân chia ngẫu nhiên
đơn thuần, nhằm đảm bảo mỗi tập đều có đủ mẫu đại diện cho tất cả
67 classes – đặc biệt quan trọng với các class hiếm như:
  - W.233  : chỉ có 3 ảnh
  - W.246c : chỉ có 5 ảnh
  - W.203b : chỉ có 11 ảnh

Thuật toán:
  1. Xây dựng bảng ánh xạ: mỗi file ảnh → tập hợp class xuất hiện
  2. Shuffle toàn bộ danh sách file với seed=42 (tái lập kết quả)
  3. Chia theo tỉ lệ 80-10-10
  4. Fix: nếu class nào vắng mặt trong Test hoặc Valid, chuyển 1 file
     từ Train sang tập thiếu (ưu tiên file ít class nhất)
  5. Lặp đến khi tất cả 67 classes có mặt đủ trong cả 3 tập

Cấu trúc thư mục yêu cầu:
    D:/Dataset/Final/
        split_dataset/
            labels/   ← toàn bộ nhãn YOLO (.txt) sau khi gộp 2 bộ

Kết quả:
    D:/Dataset/Final/
        split_dataset/
            train_files.txt   ← danh sách tên file ảnh tập Train
            valid_files.txt   ← danh sách tên file ảnh tập Validation
            test_files.txt    ← danh sách tên file ảnh tập Test

Kết quả phân chia (trước augmentation):
    Train      : 5.391 ảnh (79,9%) | ✓ 67/67 classes
    Validation :   678 ảnh (10,1%) | ✓ 67/67 classes
    Test       :   677 ảnh (10,0%) | ✓ 67/67 classes
    Tổng       : 6.746 ảnh

Yêu cầu: Python 3.6+, không cần thư viện ngoài.
=============================================================
"""

import os
import glob
import random
from collections import defaultdict

# ============================================================
#  CẤU HÌNH – chỉnh theo dataset của bạn
# ============================================================

# Thư mục gốc chứa toàn bộ dữ liệu
BASE_DIR = r"D:/Dataset/Final"

# Thư mục chứa file nhãn .txt (sau khi đã gộp 2 bộ bằng merge_dataset.py)
LABEL_DIR     = os.path.join(BASE_DIR, "split_dataset/labels")

IMAGE_EXT     = ".jpg"             # đuôi file ảnh (.jpg hoặc .png)

# Thư mục lưu kết quả (train_files.txt, valid_files.txt, test_files.txt)
OUTPUT_DIR    = os.path.join(BASE_DIR, "split_dataset")

TRAIN_RATIO   = 0.8                # 80% Train
VAL_RATIO     = 0.1                # 10% Validation
TEST_RATIO    = 0.1                # 10% Test

SEED          = 42                 # random seed để tái lập kết quả
# ============================================================


def load_dataset(label_dir, image_ext):
    """
    Đọc toàn bộ file nhãn và xây dựng:
      - file_classes : { filename.jpg → set(class_indices) }
      - class_files  : { class_idx   → [filename.jpg, ...] }
    """
    file_classes = {}
    class_files  = defaultdict(list)
    empty_files  = []

    label_files = glob.glob(os.path.join(label_dir, "*.txt"))
    if not label_files:
        raise FileNotFoundError(f"Không tìm thấy file nhãn nào trong: {label_dir}")

    for lf in label_files:
        fname = os.path.basename(lf).replace(".txt", image_ext)
        cls_set = set()
        lines = [l.strip() for l in open(lf, encoding="utf-8") if l.strip()]
        if not lines:
            empty_files.append(fname)
            continue
        for line in lines:
            parts = line.split()
            if parts:
                cls_set.add(int(parts[0]))
        file_classes[fname] = cls_set
        for c in cls_set:
            class_files[c].append(fname)

    return file_classes, class_files, empty_files


def stratified_split(file_classes, class_files, train_ratio, val_ratio, seed):
    """
    Phân chia theo Stratified Split:
      1. Shuffle ngẫu nhiên với seed cố định (seed=42)
      2. Chia theo tỉ lệ 80-10-10
      3. Fix: đảm bảo mỗi class có trong cả 3 tập
         - Nếu class vắng mặt trong Test hoặc Valid
         - Chuyển 1 file từ Train sang tập thiếu
         - Ưu tiên file có ít class nhất (giảm ảnh hưởng đến Train)
    """
    random.seed(seed)
    all_fnames = list(file_classes.keys())
    random.shuffle(all_fnames)

    total   = len(all_fnames)
    n_train = int(total * train_ratio)
    n_test  = int(total * (1 - train_ratio - val_ratio))

    train_set = set(all_fnames[:n_train])
    test_set  = set(all_fnames[n_train:n_train + n_test])
    val_set   = set(all_fnames[n_train + n_test:])

    all_classes = sorted(class_files.keys())
    fixed = []

    # Fix: nếu class nào thiếu trong test hoặc valid → chuyển 1 file từ train
    for c in all_classes:
        for target_set, split_name in [(test_set, "test"), (val_set, "valid")]:
            if not any(f in target_set for f in class_files[c]):
                # Ưu tiên file có ít class nhất (ít ảnh hưởng đến train nhất)
                candidates = sorted(
                    [f for f in class_files[c] if f in train_set],
                    key=lambda f: len(file_classes[f])
                )
                if candidates:
                    chosen = candidates[0]
                    train_set.remove(chosen)
                    target_set.add(chosen)
                    fixed.append((c, chosen, split_name))

    return train_set, val_set, test_set, fixed


def verify_splits(train_set, val_set, test_set, class_files):
    """Kiểm tra mỗi class có mặt đầy đủ trong cả 3 tập không."""
    all_classes = sorted(class_files.keys())
    missing = {"train": [], "valid": [], "test": []}
    for c in all_classes:
        if not any(f in train_set for f in class_files[c]):
            missing["train"].append(c)
        if not any(f in val_set for f in class_files[c]):
            missing["valid"].append(c)
        if not any(f in test_set for f in class_files[c]):
            missing["test"].append(c)
    return missing


def save_results(train_set, val_set, test_set, output_dir,
                 file_classes, class_files, fixed, label_dir, image_ext):
    """Ghi kết quả ra 3 file danh sách."""
    os.makedirs(output_dir, exist_ok=True)

    for fname, fset in [("train_files.txt", train_set),
                        ("valid_files.txt",  val_set),
                        ("test_files.txt",   test_set)]:
        with open(os.path.join(output_dir, fname), "w", encoding="utf-8") as f:
            f.write("\n".join(sorted(fset)))


# ============================================================
#  MAIN
# ============================================================
def main():
    print("=" * 60)
    print("  STRATIFIED SPLIT – YOLO Dataset")
    print("  Đảm bảo 67/67 classes có mặt đầy đủ trong cả 3 tập")
    print("=" * 60)

    # 1. Load dataset
    print(f"\n[1/4] Đọc nhãn từ: {LABEL_DIR}")
    file_classes, class_files, empty_files = load_dataset(LABEL_DIR, IMAGE_EXT)
    total = len(file_classes)
    n_cls = len(class_files)
    print(f"      → {total} files có nhãn | {n_cls} classes | {len(empty_files)} files rỗng")
    if empty_files:
        print(f"      Files rỗng (bỏ qua): {empty_files[:5]}{'...' if len(empty_files)>5 else ''}")

    # 2. Stratified split
    print(f"\n[2/4] Stratified Split ({int(TRAIN_RATIO*100)}-{int(VAL_RATIO*100)}-{int((1-TRAIN_RATIO-VAL_RATIO)*100)})...")
    train_set, val_set, test_set, fixed = stratified_split(
        file_classes, class_files, TRAIN_RATIO, VAL_RATIO, SEED
    )
    print(f"      Train   : {len(train_set):>6} files ({len(train_set)/total*100:.1f}%)")
    print(f"      Valid   : {len(val_set):>6} files ({len(val_set)/total*100:.1f}%)")
    print(f"      Test    : {len(test_set):>6} files ({len(test_set)/total*100:.1f}%)")
    if fixed:
        print(f"      → Điều chỉnh {len(fixed)} file từ train → test/valid (đảm bảo class hiếm)")

    # 3. Verify
    print(f"\n[3/4] Kiểm tra classes trong từng tập...")
    missing = verify_splits(train_set, val_set, test_set, class_files)
    all_ok = all(len(v) == 0 for v in missing.values())
    for split_name, miss in missing.items():
        if miss:
            print(f"      ⚠️  {split_name}: thiếu class {miss}")
        else:
            print(f"      ✓  {split_name}: đủ {n_cls}/{n_cls} classes")

    if not all_ok:
        print("\n  ⚠️  Một số class vẫn thiếu (có thể do class chỉ có 1 file).")

    # 4. Lưu kết quả
    print(f"\n[4/4] Ghi kết quả vào: {OUTPUT_DIR}/")
    save_results(
        train_set, val_set, test_set, OUTPUT_DIR,
        file_classes, class_files, fixed, LABEL_DIR, IMAGE_EXT
    )
    print(f"      ✓  train_files.txt  ({len(train_set)} dòng)")
    print(f"      ✓  valid_files.txt  ({len(val_set)} dòng)")
    print(f"      ✓  test_files.txt   ({len(test_set)} dòng)")
    print("\nHoàn thành!")


if __name__ == "__main__":
    main()

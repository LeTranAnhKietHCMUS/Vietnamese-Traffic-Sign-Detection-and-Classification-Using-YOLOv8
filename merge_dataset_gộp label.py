"""
=============================================================
  MERGE DATASET – Gộp 2 bộ dữ liệu YOLO
  Remap nhãn bộ mới (0–14 → 52–66) và cập nhật classes files
=============================================================

Bộ dữ liệu sử dụng trong đề tài được tổng hợp từ hai nguồn độc lập:
  - Bộ 1: Biển báo giao thông Việt Nam (52 classes, index 0–51)
           Thu thập thực tế, gán nhãn theo chuẩn QCVN 41:2019/BGTVT
           3.216 ảnh | 8.334 annotations
  - Bộ 2: Đèn giao thông & Biển tốc độ quốc tế – Roboflow Universe
           Project "Self Driving Cars" (selfdriving-car-qtywx), Version 6
           License: CC BY 4.0
           3.530 ảnh | 4.298 annotations | 15 classes (index 0–14 → remap 52–66)

Vì hai bộ dữ liệu dùng hệ thống đánh số class riêng biệt, cần remap
nhãn bộ 2 trước khi gộp để tránh xung đột class index.

Cấu trúc thư mục yêu cầu:
    D:/Dataset/Final/
        dataset/
            labels/         ← nhãn bộ cũ (index 0–51)
        images/             ← ảnh bộ mới
        labels/             ← nhãn bộ mới (index 0–14)
        classes.txt         ← tên class bộ cũ (52 dòng, mã biển)
        classes_en.txt      ← tên class bộ cũ (52 dòng, tiếng Anh)
        classes_vie.txt     ← tên class bộ cũ (52 dòng, tiếng Việt)

Kết quả:
    D:/Dataset/Final/
        split_dataset/
            labels/         ← nhãn đã gộp (index 0–66)
            images/         ← ảnh bộ mới (copy qua)
            classes.txt     ← 67 classes (mã biển)
            classes_en.txt  ← 67 classes (tiếng Anh)
            classes_vie.txt ← 67 classes (tiếng Việt)
            data.yaml       ← file cấu hình YOLO

Kết quả sau khi gộp:
    Bộ cũ   : 3.216 ảnh | 8.334 annotations | 52 classes (index 0–51)
    Bộ mới  : 3.530 ảnh | 4.298 annotations | 15 classes (index 0–14 → 52–66)
    Sau gộp : 6.746 ảnh | 12.632 annotations | 67 classes (index 0–66)

Yêu cầu: Python 3.6+, không cần thư viện ngoài.
=============================================================
"""

import os
import glob
import shutil

# ============================================================
#  CẤU HÌNH – chỉnh theo dataset của bạn
# ============================================================

# Thư mục gốc chứa toàn bộ dữ liệu
BASE_DIR = r"D:/Dataset/Final"

# Thư mục nhãn bộ CŨ (giữ nguyên index 0–51)
# Bộ 1: Biển báo giao thông Việt Nam – 52 classes, thu thập thực tế
OLD_LABEL_DIR   = os.path.join(BASE_DIR, "dataset/labels")

# Thư mục ảnh + nhãn bộ MỚI (index 0–14 → sẽ remap thành 52–66)
# Bộ 2: Roboflow "Self Driving Cars" – 15 classes (đèn + biển tốc độ)
NEW_LABEL_DIR   = os.path.join(BASE_DIR, "labels")
NEW_IMAGE_DIR   = os.path.join(BASE_DIR, "images")

# File classes bộ CŨ (52 dòng mỗi file)
CLASSES_OLD     = os.path.join(BASE_DIR, "classes.txt")       # mã biển
CLASSES_OLD_EN  = os.path.join(BASE_DIR, "classes_en.txt")    # tiếng Anh
CLASSES_OLD_VIE = os.path.join(BASE_DIR, "classes_vie.txt")   # tiếng Việt

# 15 class bộ MỚI (theo đúng thứ tự index 0–14, sẽ remap thành 52–66)
NEW_CLASS_CODES = [
    "TL.Green", "TL.Red",
    "SL.10", "SL.100", "SL.110", "SL.120",
    "SL.20", "SL.30", "SL.40", "SL.50",
    "SL.60", "SL.70", "SL.80", "SL.90",
    "R.Stop"
]
NEW_CLASS_EN = [
    "Green Light", "Red Light",
    "Speed Limit 10", "Speed Limit 100", "Speed Limit 110", "Speed Limit 120",
    "Speed Limit 20", "Speed Limit 30", "Speed Limit 40", "Speed Limit 50",
    "Speed Limit 60", "Speed Limit 70", "Speed Limit 80", "Speed Limit 90",
    "Stop"
]
NEW_CLASS_VIE = [
    "Đèn xanh", "Đèn đỏ",
    "Giới hạn tốc độ 10km/h", "Giới hạn tốc độ 100km/h",
    "Giới hạn tốc độ 110km/h", "Giới hạn tốc độ 120km/h",
    "Giới hạn tốc độ 20km/h", "Giới hạn tốc độ 30km/h",
    "Giới hạn tốc độ 40km/h", "Giới hạn tốc độ 50km/h",
    "Giới hạn tốc độ 60km/h", "Giới hạn tốc độ 70km/h",
    "Giới hạn tốc độ 80km/h", "Giới hạn tốc độ 90km/h",
    "Dừng lại"
]

# Thư mục đầu ra
OUTPUT_DIR = os.path.join(BASE_DIR, "split_dataset")

# Số class bộ cũ (offset để remap: 0–14 → 52–66)
OLD_CLASS_COUNT = 52
# ============================================================


def remap_label_file(src_path, dst_path, offset):
    """
    Đọc file nhãn YOLO, cộng offset vào class_id, ghi ra file mới.
    Mỗi dòng: <class_id> <cx> <cy> <w> <h>
    Ví dụ: class_id 0 → 52, class_id 14 → 66
    """
    lines = open(src_path, encoding="utf-8").readlines()
    new_lines = []
    for line in lines:
        parts = line.strip().split()
        if parts:
            new_idx = int(parts[0]) + offset
            new_lines.append(f"{new_idx} {' '.join(parts[1:])}\n")
    open(dst_path, "w", encoding="utf-8").writelines(new_lines)


def merge_classes(old_file, new_list, out_file):
    """Nối classes bộ cũ + bộ mới, ghi ra file."""
    if old_file and os.path.exists(old_file):
        old_list = open(old_file, encoding="utf-8").read().strip().split("\n")
    else:
        old_list = []
    merged = old_list + new_list
    open(out_file, "w", encoding="utf-8").write("\n".join(merged))
    return len(merged)


def main():
    print("=" * 60)
    print("  MERGE DATASET – Remap nhãn & Gộp classes")
    print("  Bộ 1 (VN, 52 classes) + Bộ 2 (Roboflow, 15 classes)")
    print("  → 67 classes (index 0–66)")
    print("=" * 60)

    out_labels = os.path.join(OUTPUT_DIR, "labels")
    out_images = os.path.join(OUTPUT_DIR, "images")
    os.makedirs(out_labels, exist_ok=True)
    os.makedirs(out_images, exist_ok=True)

    # ── BƯỚC 1: Copy nhãn bộ CŨ (giữ nguyên index 0–51) ────────
    print(f"\n[1/4] Copy nhãn bộ cũ (index 0–{OLD_CLASS_COUNT-1} giữ nguyên)...")
    print(f"      Nguồn: {OLD_LABEL_DIR}")
    old_files = glob.glob(os.path.join(OLD_LABEL_DIR, "*.txt"))
    for f in old_files:
        shutil.copy(f, os.path.join(out_labels, os.path.basename(f)))
    print(f"      ✓ {len(old_files)} file nhãn")

    # ── BƯỚC 2: Remap + Copy nhãn bộ MỚI (0–14 → 52–66) ────────
    print(f"\n[2/4] Remap nhãn bộ mới (0–{len(NEW_CLASS_CODES)-1} → "
          f"{OLD_CLASS_COUNT}–{OLD_CLASS_COUNT + len(NEW_CLASS_CODES)-1})...")
    print(f"      Nguồn: {NEW_LABEL_DIR}")
    new_label_files = glob.glob(os.path.join(NEW_LABEL_DIR, "*.txt"))
    for f in new_label_files:
        dst = os.path.join(out_labels, os.path.basename(f))
        remap_label_file(f, dst, offset=OLD_CLASS_COUNT)
    print(f"      ✓ {len(new_label_files)} file nhãn đã remap")

    # ── BƯỚC 3: Copy ảnh bộ MỚI ─────────────────────────────────
    print(f"\n[3/4] Copy ảnh bộ mới...")
    print(f"      Nguồn: {NEW_IMAGE_DIR}")
    new_images = glob.glob(os.path.join(NEW_IMAGE_DIR, "*"))
    for img in new_images:
        shutil.copy(img, os.path.join(out_images, os.path.basename(img)))
    print(f"      ✓ {len(new_images)} ảnh")

    # ── BƯỚC 4: Cập nhật file classes ───────────────────────────
    print(f"\n[4/4] Cập nhật file classes...")

    # classes.txt (mã biển)
    n = merge_classes(
        CLASSES_OLD,
        NEW_CLASS_CODES,
        os.path.join(OUTPUT_DIR, "classes.txt")
    )
    print(f"      ✓ classes.txt     → {n} classes")

    # classes_en.txt
    if CLASSES_OLD_EN and os.path.exists(CLASSES_OLD_EN):
        merge_classes(
            CLASSES_OLD_EN,
            NEW_CLASS_EN,
            os.path.join(OUTPUT_DIR, "classes_en.txt")
        )
        print(f"      ✓ classes_en.txt  → {n} classes")

    # classes_vie.txt
    if CLASSES_OLD_VIE and os.path.exists(CLASSES_OLD_VIE):
        merge_classes(
            CLASSES_OLD_VIE,
            NEW_CLASS_VIE,
            os.path.join(OUTPUT_DIR, "classes_vie.txt")
        )
        print(f"      ✓ classes_vie.txt → {n} classes")

    # data.yaml
    all_names = []
    if CLASSES_OLD_EN and os.path.exists(CLASSES_OLD_EN):
        all_names = open(CLASSES_OLD_EN, encoding="utf-8").read().strip().split("\n")
    else:
        all_names = open(os.path.join(OUTPUT_DIR, "classes.txt"), encoding="utf-8").read().strip().split("\n")
    all_names += NEW_CLASS_EN

    yaml_content = (
        f"train: ../train/images\n"
        f"val: ../valid/images\n"
        f"test: ../test/images\n\n"
        f"nc: {n}\n"
        f"names: {all_names}\n"
    )
    open(os.path.join(OUTPUT_DIR, "data.yaml"), "w", encoding="utf-8").write(yaml_content)
    print(f"      ✓ data.yaml        → nc={n}")

    # ── Tổng kết ─────────────────────────────────────────────────
    total_labels = len(glob.glob(os.path.join(out_labels, "*.txt")))
    total_images = len(glob.glob(os.path.join(out_images, "*")))

    print("\n" + "=" * 60)
    print("  KẾT QUẢ")
    print("=" * 60)
    print(f"  Nhãn bộ cũ   : {len(old_files):>6} files  (index 0–{OLD_CLASS_COUNT-1})")
    print(f"  Nhãn bộ mới  : {len(new_label_files):>6} files  (index {OLD_CLASS_COUNT}–{OLD_CLASS_COUNT+len(NEW_CLASS_CODES)-1})")
    print(f"  Tổng nhãn    : {total_labels:>6} files")
    print(f"  Tổng ảnh mới : {total_images:>6} files")
    print(f"  Tổng classes : {n}")
    print(f"  Thư mục out  : {OUTPUT_DIR}")
    print("=" * 60)
    print("\nHoàn thành!")


if __name__ == "__main__":
    main()

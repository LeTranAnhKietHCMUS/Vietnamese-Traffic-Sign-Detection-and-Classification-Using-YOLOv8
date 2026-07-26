# 🚦 Vietnamese Traffic Sign Detection and Classification Using YOLOv8

<p align="center">
  <img src="https://img.shields.io/badge/Model-YOLOv8-blueviolet" />
  <img src="https://img.shields.io/badge/Classes-67-orange" />
  <img src="https://img.shields.io/badge/Images-12.305-green" />
  <img src="https://img.shields.io/badge/mAP%4050-98.2%25-success" />
  <img src="https://img.shields.io/badge/Python-3.9%2B-blue" />
  <img src="https://img.shields.io/badge/License-MIT-lightgrey" />
</p>

Course project — **Introduction to Artificial Intelligence**

---

# 🇬🇧 English Version

## Overview

This project builds a system for **detecting** and **classifying** Vietnamese traffic signs in images, based on the **YOLOv8** architecture. Four model variants — **YOLOv8n / s / m / l** — were trained and compared on the same dataset to find the optimal trade-off between accuracy, inference speed, and deployment feasibility.

Traffic sign recognition plays a key role in real-world applications such as autonomous vehicles, Advanced Driver Assistance Systems (ADAS), and smart traffic monitoring. Recognizing Vietnamese traffic signs is particularly challenging due to the diversity of sign categories defined by **QCVN 41:2019/BGTVT** (5 main groups: prohibition, warning, mandatory, guide, and additional signs) and non-ideal real-world capture conditions (low light, partial occlusion, tilted angles, environmental noise).

## Objectives

- Build a detection + classification system for Vietnamese traffic signs using YOLOv8.
- Objectively train and compare 4 variants: **YOLOv8n, YOLOv8s, YOLOv8m, YOLOv8l** on the same dataset.
- Evaluate using multiple criteria: mAP, Precision, Recall, IoU, training time, inference speed, and deployment feasibility.
- Select the optimal model for each deployment scenario (high-end GPU server vs. embedded/edge device).

> **Scope:** the project is limited to the YOLOv8 family of architectures and does not extend to other architectures. The system was only tested/evaluated on a personal computer and was not deployed on actual embedded hardware.

## Dataset

The dataset was merged from **2 independent sources**, then unified and re-labeled into a single YOLO-format dataset with **67 classes**.

| Dataset | Source | Images | Annotations | Classes |
|---|---|---|---|---|
| Set 1 — Vietnamese traffic signs | Real-world collection, per QCVN 41:2019/BGTVT | 3,216 | 8,334 | 52 (index 0–51) |
| Set 2 — Traffic lights & international speed signs | [Roboflow Universe – Self Driving Cars](https://roboflow.com/selfdriving-car-qtywx/self-driving-cars-lfjou) (CC BY 4.0) | 3,530 | 4,298 | 15 (remapped → 52–66) |
| **Merged total** | — | **6,746** | **12,632** | **67 (0–66)** |

**Preprocessing pipeline:**
1. Remap labels of the new dataset (0–14 → 52–66) via a Python script (`os`, `glob`, `shutil`).
2. Merge labels + images from both datasets into a unified folder.
3. Update `classes.txt` / `data.yaml` (52 + 15 = 67 classes).

**Data split (Stratified Split, seed = 42):** **80% – 10% – 10%** for Train / Validation / Test, ensuring all 67 classes are represented in every split (including rare classes with only 1–3 original images).

| Split | Images (pre-augmentation) | Ratio |
|---|---|---|
| Train | 5,391 | 79.9% |
| Validation | 678 | 10.1% |
| Test | 677 | 10.0% |
| **Total** | **6,746** | 100% |

## Data Augmentation

The original dataset was **severely imbalanced** across classes (e.g., "No Stopping and No Parking" had 825 images, while "Other Danger" had only **1**). An automated augmentation pipeline was built with `albumentations` + `OpenCV`, applied only to minority classes (< 80 images/class in train), targeting a minimum of 150 images/class.

| Technique group | Methods | Purpose |
|---|---|---|
| Geometric | Affine, Perspective, HorizontalFlip | Simulate camera angle, distance, sign position |
| Color & lighting | RandomBrightnessContrast, HueSaturationValue, CLAHE, RGBShift | Simulate day/night lighting, white balance shifts |
| Noise & blur | GaussNoise/ISONoise/MultiplicativeNoise, MotionBlur/GaussianBlur | Simulate sensor noise, camera shake |
| Weather | RandomRain, RandomFog, RandomSunFlare | Simulate rain, fog, sun glare — typical of Vietnamese climate |
| Occlusion | CoarseDropout | Simulate objects partially blocking the sign |

**Results after augmentation:**
- The imbalance ratio between the most and least common class dropped from **825:1** to about **21:1**.
- All 67 classes reached a minimum of ~108 images in the training set, ensuring sufficient and fair data for learning and evaluation.
- Final augmented dataset: **12,305 images**, used to officially train the 4 models.

## YOLOv8 Model Variants

| Variant | Type | Parameters | GFLOPs | Strength |
|---|---|---|---|---|
| **YOLOv8n** | Nano | 3,042,041 | 8.3 | Extremely fast, suited for embedded devices |
| **YOLOv8s** | Small | 11,151,513 | 28.6 | Good balance between speed and accuracy |
| **YOLOv8m** | Medium | 25,878,553 | 78.9 | High accuracy, suited for mid-range servers |
| **YOLOv8l** | Large | 43,658,265 | 165.1 | Best accuracy, handles hard cases well |

All 4 models were trained for **50 epochs** on a **Tesla T4 GPU (Kaggle)**, with early stopping (`patience=10`), optimized with SGD/AdamW, using **Box Loss (CIoU)**, **Classification Loss (BCE)**, and **Distribution Focal Loss (DFL)**.

## Experimental Results

**Test set comparison (67 classes):**

| Model | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall | Inference (Tesla T4) | Size |
|---|---|---|---|---|---|---|
| YOLOv8n | 92.47% | 72.71% | 95.33% | 92.16% | 2.7 ms (~370 FPS) | 6.2 MB |
| YOLOv8s | 96.66% | 78.61% | 96.93% | 96.98% | 6.6 ms | 21.5 MB |
| YOLOv8m | 97.15% | 81.65% | 97.54% | 97.71% | 16.4 ms | 49.7 MB |
| YOLOv8l | 97.31% | 81.32% | 97.42% | 97.45% | 30.0 ms | 83.7 MB |

**Key findings:**
- None of the 4 models showed **overfitting** — validation loss tracked and decreased alongside training loss.
- Share of classes reaching AP@0.5 ≥ 95%: YOLOv8n (55.2%) → YOLOv8s (80.6%) → **YOLOv8m (88.1%, highest)** → YOLOv8l (85.1%).
- **YOLOv8l** was the only model with no class falling into the "weak" zone (AP < 70%), showing the most consistent performance across all 67 classes.
- The most common confusions occurred within the **speed limit sign group** (Speed Limit 10↔20, 110↔120, etc.) due to their visual similarity (red circle, differing only by number).

## Trade-off Analysis

| Criterion | YOLOv8n | YOLOv8s | YOLOv8m | YOLOv8l | Best |
|---|---|---|---|---|---|
| Training time (50 epochs) | 1.55 h | 2.47 h | 4.95 h | 7.69 h | YOLOv8n |
| Inference speed | 2.7 ms | 6.6 ms | 16.4 ms | 30.0 ms | YOLOv8n |
| mAP@50 (Test) | 95.9% | 97.9% | 98.2% | 98.2% | YOLOv8m/l |
| mAP@50-95 (Test) | 74.7% | 79.4% | 81.9% | 82.3% | YOLOv8l |
| Recall (Test) | 91.6% | 96.3% | 96.8% | 97.1% | YOLOv8l |

**Deployment recommendations:**
- 📡 **Embedded / edge devices** (Raspberry Pi, Jetson Nano, AI cameras): prefer **YOLOv8n** (or YOLOv8s on stronger hardware).
- 🖥️ **Mid-to-high-end GPU servers**: **YOLOv8m** is recommended as the optimal balance — mAP@50-95 = 81.9%, Precision 97.7%, Recall 96.8% — while YOLOv8l only adds 0.4% mAP@50-95 but nearly doubles inference time (diminishing returns).

## Project Structure

```
├── data/
│   ├── raw/                     # Raw data from 2 sources (Kaggle + Roboflow)
│   ├── merged/                  # Data after merging & label remapping (67 classes)
│   ├── augmented/               # Data after augmentation (12,305 images)
│   └── data.yaml                # Dataset config for Ultralytics
├── scripts/
│   ├── merge_labels.py          # Merge & remap labels of the 2 datasets
│   ├── stratified_split.py      # Stratified Train/Valid/Test split
│   └── augmentation_pipeline.py # Data augmentation pipeline (albumentations)
├── notebooks/
│   ├── train_yolov8n.ipynb
│   ├── train_yolov8s.ipynb
│   ├── train_yolov8m.ipynb
│   └── train_yolov8l.ipynb
├── runs/                        # Training outputs (weights, results.csv, charts)
├── results/                     # Comparison charts, confusion matrices, per-class bar charts
├── requirements.txt
└── README.md
```

> 💡 The structure above is a suggested layout — adjust folder/file names to match your actual source code before pushing to GitHub.



## Usage

### 1. Prepare the data
```bash
python scripts/merge_labels.py            # Merge & remap labels of both datasets → 67 classes
python scripts/stratified_split.py        # Split Train/Valid/Test (80/10/10)
python scripts/augmentation_pipeline.py   # Augment minority classes
```

### 2. Train a model
```bash
# Example: training YOLOv8m
yolo detect train \
  model=yolov8m.pt \
  data=data/data.yaml \
  epochs=50 \
  imgsz=640 \
  patience=10 \
  project=runs/train \
  name=yolov8m_run
```
Replace `yolov8m.pt` with `yolov8n.pt`, `yolov8s.pt`, or `yolov8l.pt` to train the other variants.

### 3. Evaluate on the test set
```bash
yolo detect val \
  model=runs/train/yolov8m_run/weights/best.pt \
  data=data/data.yaml \
  split=test
```

### 4. Inference on new images
```bash
yolo detect predict \
  model=runs/train/yolov8m_run/weights/best.pt \
  source=path/to/image_or_folder \
  conf=0.5 \
  save=True
```

### 5. Python API
```python
from ultralytics import YOLO

model = YOLO("runs/train/yolov8m_run/weights/best.pt")
results = model("path/to/image.jpg", conf=0.5)
results[0].show()
```

## Limitations & Future Work

**Current limitations:**
- Dataset scale is still limited (12,305 images for 67 classes); some classes still have few original samples.
- Not yet deployed or tested on real embedded/production systems.
- The speed-limit sign group (visually similar shapes) remains a common weakness across all 4 models.

**Future directions:**
- Expand the dataset, especially collecting more real-world images for rare classes.
- Optimize models (pruning, quantization) for embedded deployment (Jetson Nano/Xavier).
- Deploy and test in real-world environments (traffic cameras, ADAS devices).
- Experiment with newer YOLO architectures (YOLOv9/v10/v11) for further comparison.


---

# 🇻🇳 Bản Tiếng Việt

Đồ án môn học — **Nhập môn Trí tuệ nhân tạo**

## Giới thiệu

Dự án xây dựng hệ thống **phát hiện (detection)** và **phân loại (classification)** biển báo giao thông Việt Nam trong ảnh, dựa trên kiến trúc **YOLOv8**. Bốn biến thể mô hình — **YOLOv8n / s / m / l** — được huấn luyện và so sánh trên cùng một bộ dữ liệu nhằm tìm ra mô hình tối ưu giữa độ chính xác, tốc độ suy luận và khả năng triển khai thực tế.

Nhận diện biển báo giao thông đóng vai trò quan trọng trong các ứng dụng thực tiễn như xe tự hành, hệ thống hỗ trợ lái xe nâng cao (ADAS), và giám sát giao thông thông minh. Việc nhận diện biển báo tại Việt Nam có nhiều thách thức đặc thù do sự đa dạng của các nhóm biển báo theo **QCVN 41:2019/BGTVT** (5 nhóm chính: biển cấm, biển nguy hiểm, biển hiệu lệnh, biển chỉ dẫn, biển phụ) và điều kiện thu thập thực tế không lý tưởng (ánh sáng yếu, che khuất một phần, góc chụp nghiêng, nhiễu môi trường).

## Mục tiêu đề tài

- Xây dựng hệ thống phát hiện và phân loại biển báo giao thông Việt Nam dựa trên YOLOv8.
- Huấn luyện và so sánh khách quan 4 biến thể: **YOLOv8n, YOLOv8s, YOLOv8m, YOLOv8l** trên cùng một tập dữ liệu.
- Đánh giá theo nhiều tiêu chí: mAP, Precision, Recall, IoU, thời gian huấn luyện, tốc độ suy luận và khả năng triển khai.
- Lựa chọn mô hình tối ưu phù hợp với từng kịch bản ứng dụng (server GPU mạnh hay thiết bị nhúng/edge).

> **Phạm vi nghiên cứu:** đề tài chỉ giới hạn trong họ kiến trúc YOLOv8, không mở rộng sang các kiến trúc khác; hệ thống chỉ dừng ở mức thử nghiệm/đánh giá trên máy tính cá nhân, chưa triển khai trên thiết bị nhúng thực tế.

## Bộ dữ liệu

Bộ dữ liệu được tổng hợp từ **2 nguồn độc lập**, sau đó gộp và chuẩn hóa nhãn theo định dạng YOLO thành một bộ dữ liệu thống nhất với **67 classes**.

| Bộ dữ liệu | Nguồn | Số ảnh | Số annotations | Số classes |
|---|---|---|---|---|
| Bộ 1 — Biển báo giao thông VN | Thu thập thực tế theo chuẩn QCVN 41:2019/BGTVT | 3.216 | 8.334 | 52 (index 0–51) |
| Bộ 2 — Đèn tín hiệu & Biển tốc độ quốc tế | [Roboflow Universe – Self Driving Cars](https://roboflow.com/selfdriving-car-qtywx/self-driving-cars-lfjou) (CC BY 4.0) | 3.530 | 4.298 | 15 (remap → 52–66) |
| **Sau khi gộp** | — | **6.746** | **12.632** | **67 (0–66)** |

**Quy trình tiền xử lý:**
1. Remap nhãn của bộ dữ liệu mới (0–14 → 52–66) bằng script Python (`os`, `glob`, `shutil`).
2. Gộp nhãn + ảnh của hai bộ dữ liệu vào một thư mục thống nhất.
3. Cập nhật file `classes.txt` / `data.yaml` (52 + 15 = 67 lớp).

**Phân chia dữ liệu (Stratified Split, seed = 42):** tỉ lệ **80% – 10% – 10%** cho Train / Validation / Test, đảm bảo cả 67 class đều xuất hiện trong cả ba tập (kể cả các lớp cực hiếm chỉ có 1–3 ảnh gốc).

| Tập | Số ảnh (trước augmentation) | Tỉ lệ |
|---|---|---|
| Train | 5.391 | 79,9% |
| Validation | 678 | 10,1% |
| Test | 677 | 10,0% |
| **Tổng** | **6.746** | 100% |

## Tăng cường dữ liệu (Data Augmentation)

Bộ dữ liệu gốc bị **mất cân bằng nghiêm trọng** giữa các lớp (ví dụ lớp *"No Stopping and No Parking"* có 825 ảnh, trong khi lớp *"Other Danger"* chỉ có **1 ảnh** duy nhất). Một pipeline augmentation tự động được xây dựng bằng `albumentations` + `OpenCV`, chỉ áp dụng cho các lớp thiểu số (< 80 ảnh/lớp trong tập train), với mục tiêu tối thiểu 150 ảnh/lớp.

| Nhóm kỹ thuật | Phương pháp | Mục đích |
|---|---|---|
| Hình học | Affine, Perspective, HorizontalFlip | Mô phỏng góc camera, khoảng cách, vị trí biển báo |
| Màu sắc & ánh sáng | RandomBrightnessContrast, HueSaturationValue, CLAHE, RGBShift | Mô phỏng điều kiện sáng ngày/tối, sai lệch white balance |
| Nhiễu & Mờ | GaussNoise/ISONoise/MultiplicativeNoise, MotionBlur/GaussianBlur | Mô phỏng nhiễu cảm biến, rung camera |
| Thời tiết | RandomRain, RandomFog, RandomSunFlare | Mô phỏng mưa, sương mù, nắng chói — đặc trưng khí hậu Việt Nam |
| Che khuất | CoarseDropout | Mô phỏng vật cản che một phần biển báo |

**Kết quả sau augmentation:**
- Tỉ lệ mất cân bằng giữa lớp nhiều nhất/ít nhất giảm từ **825:1** xuống còn khoảng **21:1**.
- Toàn bộ 67 class đều đạt tối thiểu ~108 ảnh trong tập train, đảm bảo đủ dữ liệu để học và đánh giá công bằng.
- Tổng dữ liệu sau augmentation: **12.305 ảnh**, sử dụng để huấn luyện chính thức 4 mô hình.

## Các mô hình YOLOv8 sử dụng

| Phiên bản | Đặc điểm | Tham số | GFLOPs | Ưu điểm |
|---|---|---|---|---|
| **YOLOv8n** | Nano | 3.042.041 | 8.3 | Tốc độ cực cao, phù hợp thiết bị nhúng |
| **YOLOv8s** | Small | 11.151.513 | 28.6 | Cân bằng tốt giữa tốc độ và độ chính xác |
| **YOLOv8m** | Medium | 25.878.553 | 78.9 | Độ chính xác cao, phù hợp server tầm trung |
| **YOLOv8l** | Large | 43.658.265 | 165.1 | Độ chính xác tối ưu, xử lý tốt các trường hợp khó |

Cả 4 mô hình được huấn luyện **50 epochs** trên GPU **Tesla T4 (Kaggle)**, với early stopping (`patience=10`), tối ưu hóa bằng SGD/AdamW, sử dụng các thành phần loss: **Box Loss (CIoU)**, **Classification Loss (BCE)**, **Distribution Focal Loss (DFL)**.

## Kết quả thực nghiệm

**So sánh trên tập test (67 classes):**

| Mô hình | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall | Inference (Tesla T4) | Kích thước |
|---|---|---|---|---|---|---|
| YOLOv8n | 92,47% | 72,71% | 95,33% | 92,16% | 2,7 ms (~370 FPS) | 6,2 MB |
| YOLOv8s | 96,66% | 78,61% | 96,93% | 96,98% | 6,6 ms | 21,5 MB |
| YOLOv8m | 97,15% | 81,65% | 97,54% | 97,71% | 16,4 ms | 49,7 MB |
| YOLOv8l | 97,31% | 81,32% | 97,42% | 97,45% | 30,0 ms | 83,7 MB |

**Nhận xét nổi bật:**
- Cả 4 mô hình đều **không xảy ra overfitting** — validation loss bám sát và giảm đồng bộ với training loss.
- Tỉ lệ số lớp đạt AP@0.5 ≥ 95%: YOLOv8n (55,2%) → YOLOv8s (80,6%) → **YOLOv8m (88,1%, cao nhất)** → YOLOv8l (85,1%).
- **YOLOv8l** là mô hình duy nhất không có lớp nào rơi vào vùng "yếu" (AP < 70%) — thể hiện tính ổn định toàn diện trên cả 67 lớp.
- Nhầm lẫn phổ biến nhất tập trung ở nhóm **biển báo giới hạn tốc độ** (Speed Limit 10↔20, 110↔120,...) do hình dạng tương đồng (hình tròn viền đỏ, chỉ khác con số).

## Đánh giá Trade-off

| Tiêu chí | YOLOv8n | YOLOv8s | YOLOv8m | YOLOv8l | Tốt nhất |
|---|---|---|---|---|---|
| Thời gian train (50 epochs) | 1,55 giờ | 2,47 giờ | 4,95 giờ | 7,69 giờ | YOLOv8n |
| Tốc độ inference | 2,7 ms | 6,6 ms | 16,4 ms | 30,0 ms | YOLOv8n |
| mAP@50 (Test) | 95,9% | 97,9% | 98,2% | 98,2% | YOLOv8m/l |
| mAP@50-95 (Test) | 74,7% | 79,4% | 81,9% | 82,3% | YOLOv8l |
| Recall (Test) | 91,6% | 96,3% | 96,8% | 97,1% | YOLOv8l |

**Khuyến nghị triển khai:**
- 📡 **Thiết bị nhúng / edge** (Raspberry Pi, Jetson Nano, camera AI): ưu tiên **YOLOv8n** (hoặc YOLOv8s nếu phần cứng khá hơn).
- 🖥️ **Server / GPU tầm trung–cao**: **YOLOv8m** được khuyến nghị là lựa chọn cân bằng tối ưu — mAP@50-95 = 81,9%, Precision 97,7%, Recall 96,8%, trong khi YOLOv8l chỉ cải thiện thêm 0,4% mAP@50-95 nhưng tăng gần gấp đôi thời gian suy luận (hiện tượng *diminishing returns*).

## Cấu trúc thư mục

```
├── data/
│   ├── raw/                     # Dữ liệu gốc từ 2 nguồn (Kaggle + Roboflow)
│   ├── merged/                  # Dữ liệu sau khi gộp & remap nhãn (67 classes)
│   ├── augmented/               # Dữ liệu sau augmentation (12.305 ảnh)
│   └── data.yaml                # Cấu hình dataset cho Ultralytics
├── scripts/
│   ├── merge_labels.py          # Gộp & remap nhãn 2 bộ dữ liệu
│   ├── stratified_split.py      # Chia Train/Valid/Test theo Stratified Split
│   └── augmentation_pipeline.py # Pipeline tăng cường dữ liệu (albumentations)
├── notebooks/
│   ├── train_yolov8n.ipynb
│   ├── train_yolov8s.ipynb
│   ├── train_yolov8m.ipynb
│   └── train_yolov8l.ipynb
├── runs/                        # Kết quả huấn luyện (weights, results.csv, biểu đồ)
├── results/                     # Biểu đồ so sánh, confusion matrix, bar chart per-class
├── requirements.txt
└── README.md
```

> 💡 Cấu trúc trên là gợi ý tổ chức repo — hãy điều chỉnh lại cho khớp với cách bạn tổ chức mã nguồn thực tế trước khi push lên GitHub.



## Hướng dẫn sử dụng

### 1. Chuẩn bị dữ liệu
```bash
python scripts/merge_labels.py            # Gộp & remap nhãn 2 bộ dữ liệu → 67 classes
python scripts/stratified_split.py        # Chia Train/Valid/Test (80/10/10)
python scripts/augmentation_pipeline.py   # Tăng cường dữ liệu cho lớp thiểu số
```

### 2. Huấn luyện mô hình
```bash
# Ví dụ huấn luyện YOLOv8m
yolo detect train \
  model=yolov8m.pt \
  data=data/data.yaml \
  epochs=50 \
  imgsz=640 \
  patience=10 \
  project=runs/train \
  name=yolov8m_run
```
Thay `yolov8m.pt` bằng `yolov8n.pt`, `yolov8s.pt`, `yolov8l.pt` để huấn luyện các biến thể khác.

### 3. Đánh giá mô hình trên tập test
```bash
yolo detect val \
  model=runs/train/yolov8m_run/weights/best.pt \
  data=data/data.yaml \
  split=test
```

### 4. Suy luận (Inference) trên ảnh mới
```bash
yolo detect predict \
  model=runs/train/yolov8m_run/weights/best.pt \
  source=path/to/image_or_folder \
  conf=0.5 \
  save=True
```

### 5. Sử dụng bằng Python API
```python
from ultralytics import YOLO

model = YOLO("runs/train/yolov8m_run/weights/best.pt")
results = model("path/to/image.jpg", conf=0.5)
results[0].show()
```

## Hạn chế & Hướng phát triển

**Hạn chế hiện tại:**
- Quy mô dữ liệu chưa đủ lớn (12.305 ảnh cho 67 classes), một số lớp vẫn còn ít mẫu gốc.
- Chưa triển khai và kiểm thử trên thiết bị nhúng/hệ thống thực tế.
- Nhóm biển báo giới hạn tốc độ (hình dạng tương đồng) vẫn là điểm yếu chung của cả 4 mô hình.

**Hướng phát triển tương lai:**
- Mở rộng quy mô dataset, đặc biệt thu thập thêm ảnh thực tế cho các lớp hiếm.
- Tối ưu hóa mô hình (pruning, quantization) để triển khai trên thiết bị nhúng (Jetson Nano/Xavier).
- Triển khai thử nghiệm trong môi trường thực tế (camera giao thông, thiết bị ADAS).
- Thử nghiệm các kiến trúc YOLO mới hơn (YOLOv9/v10/v11) để so sánh thêm.


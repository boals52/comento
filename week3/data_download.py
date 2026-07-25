"""Google Colab 노트북에서 변환된 Python 스크립트."""


# %% [Code cell 1]
!pip install roboflow

# %% [Code cell 2]
from roboflow import Roboflow
from google.colab import drive
from getpass import getpass

import shutil
import os
import yaml

# %% [Code cell 3]
drive.mount('/content/drive')

# %% [Code cell 4]
ROBOFLOW_API_KEY = getpass("Roboflow API Key를 입력하세요: ")

rf = Roboflow(api_key=ROBOFLOW_API_KEY)

project = rf.workspace("prediction-of-traffic-situations-using-realtime-traffic-density-estimation").project("traffic-density-prediction")
version = project.version(9)
dataset = version.download("yolov8")

# %% [Code cell 5]
source_path = "/content/Traffic-Density-Prediction-9"
drive_path = (
    "/content/drive/MyDrive/2026/comento/"
    "Traffic-Density-Prediction-9"
)

shutil.copytree(
    source_path,
    drive_path,
    dirs_exist_ok=True
)

print("Drive 저장 완료")
print(os.listdir(drive_path))

# %% [Code cell 6]
# data.yaml 확인
yaml_path = "/content/drive/MyDrive/2026/comento//Traffic-Density-Prediction-9/data.yaml"

with open(yaml_path, "r", encoding="utf-8") as f:
    data_config = yaml.safe_load(f)

print(type(data_config))
print("nc:", data_config.get("nc"))
print("names:", data_config.get("names"))
print("클래스 수:", len(data_config.get("names", [])))

# %% [Code cell 7]
# 이미지와 라벨 수 확인
from pathlib import Path

dataset_root = Path("/content/drive/MyDrive/2026/comento/Traffic-Density-Prediction-9")
image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

for split in ["train", "valid", "test"]:
    image_dir = dataset_root / split / "images"
    label_dir = dataset_root / split / "labels"

    image_files = [
        path for path in image_dir.iterdir()
        if path.suffix.lower() in image_extensions
    ]
    label_files = list(label_dir.glob("*.txt"))

    print(
        f"{split:5s} | "
        f"images={len(image_files):5d} | "
        f"labels={len(label_files):5d}"
    )

# %% [Code cell 8]
# 이미지와 라벨 짝 검사
for split in ["train", "valid", "test"]:
    image_dir = dataset_root / split / "images"
    label_dir = dataset_root / split / "labels"

    image_stems = {
        path.stem
        for path in image_dir.iterdir()
        if path.suffix.lower() in image_extensions
    }
    label_stems = {
        path.stem
        for path in label_dir.glob("*.txt")
    }

    missing_labels = image_stems - label_stems
    orphan_labels = label_stems - image_stems

    print(f"\n[{split}]")
    print("라벨 없는 이미지:", len(missing_labels))
    print("이미지 없는 라벨:", len(orphan_labels))
    print("라벨 없는 이미지 예시:", list(missing_labels)[:5])
    print("이미지 없는 라벨 예시:", list(orphan_labels)[:5])

# %% [Code cell 9]
# 라벨 형식과 좌표 범위 검사
import math

names = data_config["names"]
num_classes = len(names)

invalid_labels = []
empty_labels = []
class_counts = [0] * num_classes

for split in ["train", "valid", "test"]:
    label_dir = dataset_root / split / "labels"

    for label_path in label_dir.glob("*.txt"):
        lines = label_path.read_text(encoding="utf-8").strip().splitlines()

        if not lines:
            empty_labels.append(str(label_path))
            continue

        for line_number, line in enumerate(lines, start=1):
            values = line.split()

            if len(values) != 5:
                invalid_labels.append(
                    (str(label_path), line_number, "열 개수 오류", line)
                )
                continue

            try:
                class_id = int(float(values[0]))
                x_center, y_center, width, height = map(float, values[1:])
            except ValueError:
                invalid_labels.append(
                    (str(label_path), line_number, "숫자 변환 오류", line)
                )
                continue

            if not 0 <= class_id < num_classes:
                invalid_labels.append(
                    (str(label_path), line_number, "클래스 ID 오류", line)
                )
                continue

            coordinates = [x_center, y_center, width, height]

            if not all(math.isfinite(value) for value in coordinates):
                invalid_labels.append(
                    (str(label_path), line_number, "NaN/Inf 좌표", line)
                )
                continue

            if not all(0 <= value <= 1 for value in coordinates):
                invalid_labels.append(
                    (str(label_path), line_number, "좌표 범위 오류", line)
                )
                continue

            if width <= 0 or height <= 0:
                invalid_labels.append(
                    (str(label_path), line_number, "박스 크기 오류", line)
                )
                continue

            class_counts[class_id] += 1

print("잘못된 라벨 수:", len(invalid_labels))
print("빈 라벨 파일 수:", len(empty_labels))
print("오류 예시:", invalid_labels[:5])

# %% [Code cell 10]
# 클래스 정의와 클래스 불균형 확인
import pandas as pd

class_distribution = pd.DataFrame({
    "class_id": range(num_classes),
    "class_name": names,
    "box_count": class_counts,
})

class_distribution["ratio"] = (
    class_distribution["box_count"]
    / class_distribution["box_count"].sum()
)

class_distribution = class_distribution.sort_values(
    "box_count",
    ascending=False,
)

display(class_distribution)

# %% [Code cell 11]
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.bar(
    class_distribution["class_name"],
    class_distribution["box_count"],
)
plt.xlabel("Class")
plt.ylabel("Number of Bounding Boxes")
plt.title("Class Distribution")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# %% [Code cell 12]
max_count = class_distribution["box_count"].max()
min_count = class_distribution["box_count"].min()

print("최대 클래스 개수:", max_count)
print("최소 클래스 개수:", min_count)

if min_count > 0:
    print("불균형 비율:", max_count / min_count)

# %% [Code cell 13]
# split별 클래스 분포 확인
from collections import Counter

split_class_counts = {}

for split in ["train", "valid", "test"]:
    counter = Counter()
    label_dir = dataset_root / split / "labels"

    for label_path in label_dir.glob("*.txt"):
        for line in label_path.read_text().strip().splitlines():
            values = line.split()

            if len(values) == 5:
                class_id = int(float(values[0]))
                counter[class_id] += 1

    split_class_counts[split] = counter

split_df = pd.DataFrame({
    split: [
        split_class_counts[split].get(class_id, 0)
        for class_id in range(num_classes)
    ]
    for split in ["train", "valid", "test"]
})

split_df.insert(0, "class_name", names)
display(split_df)

# %% [Code cell 14]
# Bounding Box 크기 분석
box_areas = []
box_widths = []
box_heights = []
objects_per_image = []

for split in ["train", "valid", "test"]:
    label_dir = dataset_root / split / "labels"

    for label_path in label_dir.glob("*.txt"):
        lines = label_path.read_text().strip().splitlines()
        objects_per_image.append(len(lines))

        for line in lines:
            values = line.split()

            if len(values) != 5:
                continue

            width = float(values[3])
            height = float(values[4])

            box_widths.append(width)
            box_heights.append(height)
            box_areas.append(width * height)

print("Bounding Box 수:", len(box_areas))
print("평균 정규화 면적:", sum(box_areas) / len(box_areas))
print("최소 정규화 면적:", min(box_areas))
print("최대 정규화 면적:", max(box_areas))

# %% [Code cell 15]
plt.figure(figsize=(8, 5))
plt.hist(box_areas, bins=50)
plt.xlabel("Normalized Bounding Box Area")
plt.ylabel("Count")
plt.title("Bounding Box Area Distribution")
plt.show()

# %% [Code cell 16]
# 실제 라벨 시각화

import cv2
import random
import matplotlib.pyplot as plt

def draw_yolo_labels(image_path, label_path, class_names):
    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(f"이미지를 읽을 수 없습니다: {image_path}")

    height, width = image.shape[:2]

    if label_path.exists():
        for line in label_path.read_text().strip().splitlines():
            values = line.split()

            if len(values) != 5:
                continue

            class_id = int(float(values[0]))
            x_center, y_center, box_width, box_height = map(
                float,
                values[1:],
            )

            x1 = int((x_center - box_width / 2) * width)
            y1 = int((y_center - box_height / 2) * height)
            x2 = int((x_center + box_width / 2) * width)
            y2 = int((y_center + box_height / 2) * height)

            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            cv2.putText(
                image,
                class_names[class_id],
                (x1, max(y1 - 5, 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )

    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# %% [Code cell 17]
train_images = list(
    (dataset_root / "train" / "images").glob("*")
)

sample_paths = random.sample(
    train_images,
    min(9, len(train_images)),
)

plt.figure(figsize=(15, 15))

for index, image_path in enumerate(sample_paths, start=1):
    label_path = (
        dataset_root
        / "train"
        / "labels"
        / f"{image_path.stem}.txt"
    )

    image = draw_yolo_labels(
        image_path,
        label_path,
        names,
    )

    plt.subplot(3, 3, index)
    plt.imshow(image)
    plt.axis("off")
    plt.title(image_path.name[:25])

plt.tight_layout()
plt.show()

# %% [Code cell 18]
# 중복 이미지와 데이터 누수 확인

import hashlib
from collections import defaultdict

def file_hash(path):
    hasher = hashlib.md5()

    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)

    return hasher.hexdigest()

hash_to_items = defaultdict(list)

for split in ["train", "valid", "test"]:
    image_dir = dataset_root / split / "images"

    for image_path in image_dir.iterdir():
        if image_path.suffix.lower() in image_extensions:
            hash_to_items[file_hash(image_path)].append(
                (split, image_path.name)
            )

cross_split_duplicates = []

for items in hash_to_items.values():
    splits = {split for split, _ in items}

    if len(splits) > 1:
        cross_split_duplicates.append(items)

print("Split 간 완전 중복 그룹:", len(cross_split_duplicates))
print("예시:", cross_split_duplicates[:5])

# %% [Code cell 19]
for split in ["train", "valid", "test"]:
    image_names = sorted(
        path.name
        for path in (dataset_root / split / "images").iterdir()
    )

    print(f"\n[{split}]")
    print(image_names[:20])

# %% [Code cell 20]
# Resize 방식 확인

from PIL import Image
from collections import Counter

image_shapes = Counter()

for image_path in (
    dataset_root / "train" / "images"
).iterdir():

    if image_path.suffix.lower() not in image_extensions:
        continue

    with Image.open(image_path) as image:
        image_shapes[image.size] += 1

print(image_shapes.most_common(10))

# %% [Code cell 21]
import re
from pathlib import Path
from collections import defaultdict

dataset_root = Path("/content/Traffic-Density-Prediction-9")
image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def extract_source_id(filename: str) -> str:
    """
    Roboflow 해시와 영상 프레임 번호를 제거해
    원본 영상 또는 원본 이미지 그룹 ID를 추출한다.
    """
    stem = Path(filename).stem

    # Roboflow가 붙인 .rf.<hash> 제거
    stem = re.sub(r"\.rf\.[a-f0-9]+$", "", stem)

    # 예: 1-10-1-20_mp4-0331_jpg → 1-10-1-20_mp4
    match = re.match(r"(.+_mp4)-\d+_jpg$", stem)

    if match:
        return match.group(1)

    # 일반 이미지도 Roboflow 해시를 제거한 이름으로 그룹화
    return stem


source_splits = defaultdict(set)
source_examples = defaultdict(list)

for split in ["train", "valid", "test"]:
    image_dir = dataset_root / split / "images"

    for image_path in image_dir.iterdir():
        if image_path.suffix.lower() not in image_extensions:
            continue

        source_id = extract_source_id(image_path.name)

        source_splits[source_id].add(split)
        source_examples[source_id].append(
            (split, image_path.name)
        )

overlapping_sources = {
    source_id: splits
    for source_id, splits in source_splits.items()
    if len(splits) > 1
}

print("여러 split에 포함된 원본 그룹 수:", len(overlapping_sources))

for source_id, splits in list(overlapping_sources.items())[:10]:
    print(f"\n원본 그룹: {source_id}")
    print("포함된 split:", sorted(splits))

    for item in source_examples[source_id][:10]:
        print(" ", item)

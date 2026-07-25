"""Google Colab 노트북에서 변환된 Python 스크립트."""


# %% [Code cell 1]
from google.colab import drive
drive.mount("/content/drive")

# %% [Code cell 2]
!pip install ultralytics

# %% [Code cell 3]
from pathlib import Path
from functools import reduce
from pathlib import Path
from ultralytics import YOLO

import cv2
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# %% [Code cell 4]
RESULTS_ROOT = Path(
    "/content/drive/MyDrive/2026/comento/experiment_results"
)

summary_files = [
    RESULTS_ROOT / f"exp{i}_summary.csv"
    for i in range(1, 6)
]

for path in summary_files:
    print(path.name, path.exists())

# %% [Code cell 5]
summary_dfs = [
    pd.read_csv(path)
    for path in summary_files
    if path.exists()
]

comparison_df = pd.concat(
    summary_dfs,
    ignore_index=True,
)

comparison_columns = [
    "experiment",
    "model",
    "epochs",
    "precision",
    "recall",
    "mAP50",
    "mAP50-95",
    "parameter_count",
]

comparison_df = comparison_df[
    [
        column
        for column in comparison_columns
        if column in comparison_df.columns
    ]
]

display(comparison_df)

# %% [Code cell 6]
# 반올림 결과
display_df = comparison_df.copy()

metric_columns = [
    "precision",
    "recall",
    "mAP50",
    "mAP50-95",
]

for column in metric_columns:
    if column in display_df.columns:
        display_df[column] = (
            display_df[column] * 100
        ).round(2)

display(display_df)

# %% [Code cell 7]
clean_comparison_df = comparison_df[
    comparison_df["experiment"].isin(
        ["exp2", "exp3", "exp4", "exp5"]
    )
].copy()

clean_comparison_df["mAP50-95_rank"] = (
    clean_comparison_df["mAP50-95"]
    .rank(ascending=False, method="min")
)

clean_comparison_df["mAP50_rank"] = (
    clean_comparison_df["mAP50"]
    .rank(ascending=False, method="min")
)

clean_comparison_df["recall_rank"] = (
    clean_comparison_df["recall"]
    .rank(ascending=False, method="min")
)

# %% [Code cell 8]
# 클래스별 AP 비교
class_ap_dfs = []

for experiment_number in range(2, 6):
    path = (
        RESULTS_ROOT
        / f"exp{experiment_number}_class_ap.csv"
    )

    if not path.exists():
        print("파일 없음:", path)
        continue

    dataframe = pd.read_csv(path)

    dataframe = dataframe.rename(
        columns={
            "mAP50-95": f"exp{experiment_number}"
        }
    )

    class_ap_dfs.append(
        dataframe[
            [
                "class_id",
                "class_name",
                f"exp{experiment_number}",
            ]
        ]
    )

class_ap_comparison = reduce(
    lambda left, right: pd.merge(
        left,
        right,
        on=["class_id", "class_name"],
        how="outer",
    ),
    class_ap_dfs,
)

class_ap_display = class_ap_comparison.copy()

for experiment in ["exp2", "exp3", "exp4", "exp5"]:
    if experiment in class_ap_display.columns:
        class_ap_display[experiment] = (
            class_ap_display[experiment] * 100
        ).round(2)

display(class_ap_display)

# %% [Code cell 9]
RUNS_ROOT = Path(
    "/content/drive/MyDrive/2026/comento/runs"
)

MODEL_PATHS = {
    "exp1": RUNS_ROOT / "exp1_yolov8n_baseline/weights/best.pt",
    "exp2": RUNS_ROOT / "exp2_yolov8n_clean_split/weights/best.pt",
    "exp3": RUNS_ROOT / "exp3_yolov8n_50epochs/weights/best.pt",
    "exp4": RUNS_ROOT / "exp4_yolov8n_augmentation/weights/best.pt",
    "exp5": RUNS_ROOT / "exp5_yolov8s_baseline/weights/best.pt",
}

for name, path in MODEL_PATHS.items():
    print(name, path.exists(), path)

# %% [Code cell 10]
CLEAN_ROOT = Path(
    "/content/drive/MyDrive/2026/comento/Traffic-Density-Prediction-9-cleaned"
)

COMMON_IMAGE_DIR = CLEAN_ROOT / "valid/images"
COMMON_LABEL_DIR = CLEAN_ROOT / "valid/labels"

# %% [Code cell 11]
def load_yolo_ground_truth(label_path, image_width, image_height):
    boxes = []
    classes = []

    if not label_path.exists():
        return np.empty((0, 4)), np.empty((0,), dtype=int)

    lines = label_path.read_text(
        encoding="utf-8"
    ).strip().splitlines()

    for line in lines:
        values = line.split()

        if len(values) != 5:
            continue

        class_id = int(float(values[0]))
        x_center, y_center, width, height = map(
            float,
            values[1:]
        )

        x1 = (x_center - width / 2) * image_width
        y1 = (y_center - height / 2) * image_height
        x2 = (x_center + width / 2) * image_width
        y2 = (y_center + height / 2) * image_height

        boxes.append([x1, y1, x2, y2])
        classes.append(class_id)

    return (
        np.array(boxes, dtype=float),
        np.array(classes, dtype=int),
    )

# %% [Code cell 12]
# IoU 계산
def compute_iou(box_a, box_b):
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    intersection_width = max(0, x2 - x1)
    intersection_height = max(0, y2 - y1)
    intersection = intersection_width * intersection_height

    area_a = max(0, box_a[2] - box_a[0]) * max(
        0,
        box_a[3] - box_a[1],
    )

    area_b = max(0, box_b[2] - box_b[0]) * max(
        0,
        box_b[3] - box_b[1],
    )

    union = area_a + area_b - intersection

    if union == 0:
        return 0.0

    return intersection / union

# %% [Code cell 13]
# 모델 예측이 Ground Truth를 성공적으로 탐지했는지 판단
def evaluate_image_detection(
    prediction_result,
    gt_boxes,
    gt_classes,
    iou_threshold=0.5,
):
    if prediction_result.boxes is None:
        pred_boxes = np.empty((0, 4))
        pred_classes = np.empty((0,), dtype=int)
        pred_confidences = np.empty((0,))
    else:
        pred_boxes = (
            prediction_result.boxes.xyxy
            .cpu()
            .numpy()
        )

        pred_classes = (
            prediction_result.boxes.cls
            .cpu()
            .numpy()
            .astype(int)
        )

        pred_confidences = (
            prediction_result.boxes.conf
            .cpu()
            .numpy()
        )

    matched_gt = set()
    true_positive_count = 0

    for pred_box, pred_class, pred_conf in zip(
        pred_boxes,
        pred_classes,
        pred_confidences,
    ):
        best_iou = 0.0
        best_gt_index = None

        for gt_index, (gt_box, gt_class) in enumerate(
            zip(gt_boxes, gt_classes)
        ):
            if gt_index in matched_gt:
                continue

            if pred_class != gt_class:
                continue

            iou = compute_iou(pred_box, gt_box)

            if iou > best_iou:
                best_iou = iou
                best_gt_index = gt_index

        if (
            best_gt_index is not None
            and best_iou >= iou_threshold
        ):
            matched_gt.add(best_gt_index)
            true_positive_count += 1

    false_negative_count = len(gt_boxes) - len(matched_gt)

    return {
        "tp": true_positive_count,
        "fn": false_negative_count,
        "gt_count": len(gt_boxes),
        "detected_all": (
            len(gt_boxes) > 0
            and false_negative_count == 0
        ),
        "detected_any": true_positive_count > 0,
    }

# %% [Code cell 14]
models = {
    name: YOLO(str(path))
    for name, path in MODEL_PATHS.items()
}

# %% [Code cell 15]
from tqdm.auto import tqdm
# 공통 이미지에서 자동 탐색
image_extensions = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}

image_paths = sorted([
    path
    for path in COMMON_IMAGE_DIR.iterdir()
    if path.suffix.lower() in image_extensions
])

comparison_records = []

for image_path in tqdm(image_paths):
    image = cv2.imread(str(image_path))

    if image is None:
        continue

    height, width = image.shape[:2]

    label_path = (
        COMMON_LABEL_DIR
        / f"{image_path.stem}.txt"
    )

    gt_boxes, gt_classes = load_yolo_ground_truth(
        label_path,
        width,
        height,
    )

    if len(gt_boxes) == 0:
        continue

    model_results = {}

    for experiment_name, model in models.items():
        prediction = model.predict(
            source=str(image_path),
            imgsz=640,
            conf=0.25,
            iou=0.7,
            verbose=False,
        )[0]

        model_results[experiment_name] = (
            evaluate_image_detection(
                prediction,
                gt_boxes,
                gt_classes,
                iou_threshold=0.5,
            )
        )

    record = {
        "image_path": str(image_path),
        "gt_count": len(gt_boxes),
    }

    for experiment_name, result in model_results.items():
        record[f"{experiment_name}_tp"] = result["tp"]
        record[f"{experiment_name}_fn"] = result["fn"]
        record[
            f"{experiment_name}_detected_all"
        ] = result["detected_all"]

    comparison_records.append(record)

# %% [Code cell 16]
# 실험별 성공 사례 찾기

comparison_results_df = pd.DataFrame(
    comparison_records
)

success_cases = {}

for experiment_name in [
    "exp2",
    "exp3",
    "exp4",
    "exp5",
]:
    cases = comparison_results_df[
        (
            comparison_results_df["exp1_fn"] > 0
        )
        &
        (
            comparison_results_df[
                f"{experiment_name}_tp"
            ]
            >
            comparison_results_df["exp1_tp"]
        )
    ].copy()

    success_cases[experiment_name] = cases

    print(
        experiment_name,
        "개선 사례 수:",
        len(cases),
    )

# %% [Code cell 17]
# 대표 이미지 한 장씩 선택

selected_cases = {}

for experiment_name, cases in success_cases.items():
    if len(cases) == 0:
        print(
            experiment_name,
            "개선 사례 없음",
        )
        continue

    cases = cases.copy()

    cases["tp_gain"] = (
        cases[f"{experiment_name}_tp"]
        - cases["exp1_tp"]
    )

    selected_row = cases.sort_values(
        ["tp_gain", "gt_count"],
        ascending=[False, False],
    ).iloc[0]

    selected_cases[experiment_name] = Path(
        selected_row["image_path"]
    )

    print(
        experiment_name,
        "선택 이미지:",
        selected_row["image_path"],
        "TP 증가:",
        selected_row["tp_gain"],
    )

# %% [Code cell 18]
# GT 시각화 함수

def draw_ground_truth(
    image_path,
    label_path,
    class_names,
):
    image = cv2.imread(str(image_path))

    height, width = image.shape[:2]

    gt_boxes, gt_classes = load_yolo_ground_truth(
        label_path,
        width,
        height,
    )

    for box, class_id in zip(
        gt_boxes,
        gt_classes,
    ):
        x1, y1, x2, y2 = map(int, box)

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2,
        )

        cv2.putText(
            image,
            class_names[class_id],
            (x1, max(y1 - 5, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )

    return cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB,
    )

# %% [Code cell 19]
# 비교 시각화 함수
def visualize_all_experiments(
    image_path,
    models,
    class_names,
    save_path=None,
):
    label_path = (
        COMMON_LABEL_DIR
        / f"{image_path.stem}.txt"
    )

    gt_image = draw_ground_truth(
        image_path,
        label_path,
        class_names,
    )

    figures = [
        ("Ground Truth", gt_image)
    ]

    for experiment_name in [
        "exp1",
        "exp2",
        "exp3",
        "exp4",
        "exp5",
    ]:
        result = models[
            experiment_name
        ].predict(
            source=str(image_path),
            imgsz=640,
            conf=0.25,
            iou=0.7,
            verbose=False,
        )[0]

        plotted_image = result.plot()

        plotted_image = cv2.cvtColor(
            plotted_image,
            cv2.COLOR_BGR2RGB,
        )

        figures.append(
            (
                experiment_name.upper(),
                plotted_image,
            )
        )

    plt.figure(figsize=(18, 10))

    for index, (title, image) in enumerate(
        figures,
        start=1,
    ):
        plt.subplot(2, 3, index)
        plt.imshow(image)
        plt.title(title)
        plt.axis("off")

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight",
        )

    plt.show()

# %% [Code cell 20]
OUTPUT_DIR = (
    "/content/drive/MyDrive/2026/comento/"
    "experiment_results"
)

class_names = [
    "Bicycle",
    "E-bike",
    "Jeepney",
    "Motorcycle",
    "Pedestrian",
    "Tricycle",
    "Truck",
    "Vehicle",
]

for target_exp in ["exp2", "exp3", "exp4", "exp5"]:
    example_image = selected_cases[target_exp]

    print(f"\n{target_exp} 대표 사례")
    print("이미지:", example_image.name)

    visualize_all_experiments(
        image_path=example_image,
        models=models,
        class_names=class_names,
        save_path=(
            f"{OUTPUT_DIR}/"
            f"comparison_{target_exp}_example.png"
        ),
    )

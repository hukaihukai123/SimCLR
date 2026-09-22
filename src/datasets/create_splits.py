import json
from collections import Counter
from pathlib import Path

import numpy as np
from torchvision.datasets import CIFAR10


SEED = 42
NUM_CLASSES = 10

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data" / "raw"
SPLIT_ROOT = PROJECT_ROOT / "splits"
OUTPUT_PATH = SPLIT_ROOT / f"cifar10_seed{SEED}.json"


def class_counts(indices: list[int], targets: np.ndarray) -> dict[int, int]:
    return dict(sorted(Counter(targets[indices].tolist()).items()))


def main() -> None:
    dataset = CIFAR10(
        root=str(DATA_ROOT),
        train=True,
        download=False,
        transform=None,
    )

    targets = np.asarray(dataset.targets)
    rng = np.random.default_rng(SEED)

    train_by_class: dict[int, np.ndarray] = {}
    val_indices: list[int] = []

    # CIFAR-10 每类 5000 张：
    # 每类取 500 张作为验证集，剩余 4500 张作为训练池。
    for class_id in range(NUM_CLASSES):
        class_indices = np.flatnonzero(targets == class_id)
        rng.shuffle(class_indices)

        val_indices.extend(class_indices[:500].tolist())
        train_by_class[class_id] = class_indices[500:]

    # 使用嵌套采样：
    # 1% 标签子集包含于 10% 标签子集。
    labeled_1_percent: list[int] = []
    labeled_10_percent: list[int] = []
    labeled_100_percent: list[int] = []

    for class_id in range(NUM_CLASSES):
        indices = train_by_class[class_id]

        # 45 / 4500 = 1%
        labeled_1_percent.extend(indices[:45].tolist())

        # 450 / 4500 = 10%
        labeled_10_percent.extend(indices[:450].tolist())

        labeled_100_percent.extend(indices.tolist())

    ssl_train_indices = labeled_100_percent.copy()

    # 只打乱最终顺序，不改变样本成员。
    rng.shuffle(ssl_train_indices)
    rng.shuffle(val_indices)
    rng.shuffle(labeled_1_percent)
    rng.shuffle(labeled_10_percent)
    rng.shuffle(labeled_100_percent)

    # 完整性检查
    assert len(ssl_train_indices) == 45_000
    assert len(val_indices) == 5_000
    assert len(labeled_1_percent) == 450
    assert len(labeled_10_percent) == 4_500
    assert len(labeled_100_percent) == 45_000

    assert set(ssl_train_indices).isdisjoint(val_indices)
    assert set(labeled_1_percent).issubset(labeled_10_percent)
    assert set(labeled_10_percent).issubset(labeled_100_percent)

    split_data = {
        "dataset": "CIFAR-10",
        "seed": SEED,
        "ssl_train": ssl_train_indices,
        "validation": val_indices,
        "labeled": {
            "1_percent": labeled_1_percent,
            "10_percent": labeled_10_percent,
            "100_percent": labeled_100_percent,
        },
    }

    SPLIT_ROOT.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(split_data, file, indent=2)

    print(f"Split saved to: {OUTPUT_PATH.resolve()}")
    print()
    print("SSL train:", len(ssl_train_indices))
    print("Validation:", len(val_indices))
    print("Labeled 1%:", len(labeled_1_percent))
    print("Labeled 10%:", len(labeled_10_percent))
    print("Labeled 100%:", len(labeled_100_percent))
    print()
    print("Validation class counts:")
    print(class_counts(val_indices, targets))
    print("1% labeled class counts:")
    print(class_counts(labeled_1_percent, targets))
    print("10% labeled class counts:")
    print(class_counts(labeled_10_percent, targets))


if __name__ == "__main__":
    main()
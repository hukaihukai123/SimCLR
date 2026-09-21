from pathlib import Path

from torchvision.datasets import CIFAR10


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data" / "raw"


def main() -> None:
    DATA_ROOT.mkdir(parents=True, exist_ok=True)

    train_dataset = CIFAR10(
        root=str(DATA_ROOT),
        train=True,
        download=True,
        transform=None,
    )

    test_dataset = CIFAR10(
        root=str(DATA_ROOT),
        train=False,
        download=True,
        transform=None,
    )

    image, label = train_dataset[0]

    print("CIFAR-10 download completed.")
    print(f"Data root: {DATA_ROOT.resolve()}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples:  {len(test_dataset)}")
    print(f"Image type:    {type(image).__name__}")
    print(f"Image size:    {image.size}")
    print(f"First label:   {label}")
    print(f"Class name:    {train_dataset.classes[label]}")
    print(f"Classes:       {train_dataset.classes}")


if __name__ == "__main__":
    main()
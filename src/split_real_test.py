from pathlib import Path
import random
import shutil

CLASSES = ["football", "basketball", "tennis", "boxing", "swimming"]
TEST_PER_CLASS = 4
SEED = 42

def list_images(folder: Path):
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    return [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts]

def main():
    random.seed(SEED)

    project_root = Path(__file__).resolve().parents[1]
    train_root = project_root / "data" / "train_real"
    test_root  = project_root / "data" / "test_real"

    for cls in CLASSES:
        src_dir = train_root / cls
        dst_dir = test_root / cls
        dst_dir.mkdir(parents=True, exist_ok=True)

        imgs = list_images(src_dir)
        if len(imgs) != 22:
            raise ValueError(f"{cls}: expected 22 images in train_real, got {len(imgs)}")

        random.shuffle(imgs)
        test_imgs = imgs[:TEST_PER_CLASS]

        for img in test_imgs:
            shutil.move(str(img), str(dst_dir / img.name))

        print(f"{cls}: moved {TEST_PER_CLASS} to test_real")

    # quick summary
    for cls in CLASSES:
        tr = len(list_images(train_root / cls))
        te = len(list_images(test_root / cls))
        print(f"{cls}: train_real={tr}, test_real={te}")

if __name__ == "__main__":
    main()

from pathlib import Path
from PIL import Image, ImageEnhance
import random

CLASSES = ["football", "basketball", "tennis", "boxing", "swimming"]
SYNTH_PER_CLASS = 22
SEED = 42

def list_images(folder: Path):
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    return [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts]

def random_transform(img: Image.Image):
    # losowa rotacja
    angle = random.uniform(-20, 20)
    img = img.rotate(angle, expand=True)

    # losowa jasność
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(random.uniform(0.7, 1.3))

    # losowy kontrast
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(random.uniform(0.7, 1.3))

    return img

def main():
    random.seed(SEED)

    project_root = Path(__file__).resolve().parents[1]
    train_real = project_root / "data" / "train_real"
    train_synth = project_root / "data" / "train_synth"

    for cls in CLASSES:
        src_dir = train_real / cls
        dst_dir = train_synth / cls
        dst_dir.mkdir(parents=True, exist_ok=True)

        imgs = list_images(src_dir)
        if len(imgs) == 0:
            raise ValueError(f"No images found in {src_dir}")

        for i in range(SYNTH_PER_CLASS):
            src_img_path = random.choice(imgs)
            img = Image.open(src_img_path).convert("RGB")

            img = random_transform(img)

            out_path = dst_dir / f"{cls}_synth_{i}.jpg"
            img.save(out_path, quality=90)

        print(f"{cls}: generated {SYNTH_PER_CLASS} synthetic images")

if __name__ == "__main__":
    main()

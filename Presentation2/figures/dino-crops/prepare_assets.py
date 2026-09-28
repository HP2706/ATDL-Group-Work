"""Prepare the explicit crop/resolution examples; no model inference is performed."""
from pathlib import Path
from PIL import Image


def main() -> None:
    root = Path(__file__).resolve().parent
    with Image.open(root / 'dog.jpg') as source:
        crops: dict[str, tuple[int, int, int, int]] = {
            'global-a': (0, 0, 1213, 1213),
            'local-a': (220, 70, 640, 490),
        }
        for name, bounds in crops.items():
            size = 112 if name.startswith('local') else 256
            source.crop(bounds).resize((size, size), Image.Resampling.BICUBIC).save(root / f'{name}.png')
        source.crop(crops['global-a']).resize((512, 512), Image.Resampling.BICUBIC).save(root / 'global-a-hires.png')


if __name__ == '__main__':
    main()

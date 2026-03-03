'''
This module is used to: discover image files, build a simple index DataFrame,
validate readability and sizes, and optionally load an external labels CSV to merge with the image index.
'''

from pathlib import Path
from typing import List, Optional
import pandas as pd
import cv2


class MessidorDataset:
    '''Minimal handler for a local Messidor-2 dataset directory.

    Usage:
        ds = MessidorDataset('data/messidor-2')
        ds.discover_images()
        df = ds.to_dataframe()
        ds.validate(sample=5)
    '''

    def __init__(self, data_dir: str = "data/messidor-2"):
        self.data_dir = Path(data_dir)
        self.images: List[Path] = []
        self.index: Optional[pd.DataFrame] = None

    def discover_images(self, exts=None) -> List[Path]:
        if exts is None:
            exts = {'.tif', '.tiff', '.png', '.jpg', '.jpeg'}

        self.images = [p for p in sorted(self.data_dir.rglob('*')) if p.suffix.lower() in exts]
        return self.images

    def to_dataframe(self) -> pd.DataFrame:
        '''Return a DataFrame index with columns: `image_id`, `file_path`.

        `image_id` is the filename stem (e.g. 20051020_43808_0100_PP).
        '''
        if not self.images:
            self.discover_images()

        rows = []
        for p in self.images:
            rows.append({'image_id': p.stem, 'file_path': str(p)})

        self.index = pd.DataFrame(rows)
        return self.index

    def load_labels(self, labels_csv: str, image_id_col: Optional[str] = None) -> Optional[pd.DataFrame]:
        '''Load a labels CSV and merge with the image index if possible.

        If `image_id_col` is None the method will attempt to find a matching
        column name containing 'image' or 'name'.
        '''
        if self.index is None:
            self.to_dataframe()

        df = pd.read_csv(labels_csv)

        if image_id_col is None:
            candidates = [c for c in df.columns if 'image' in c.lower() or 'name' in c.lower()]
            image_id_col = candidates[0] if candidates else df.columns[0]

        merged = pd.merge(self.index, df, left_on='image_id', right_on=image_id_col, how='left')
        self.index = merged
        return merged

    def validate(self, sample: int = 5) -> dict:
        '''Quick validation: try to open up to `sample` images and report sizes.

        Returns a small report dict.
        '''
        if not self.images:
            self.discover_images()

        report = {'total_images': len(self.images), 'checked': 0, 'failed': 0, 'sizes': []}

        for p in self.images[:sample]:
            img = cv2.imread(str(p))
            report['checked'] += 1
            if img is None:
                report['failed'] += 1
                report['sizes'].append(None)
            else:
                report['sizes'].append(img.shape[:2])

        return report


def validate_dataset_dir(data_dir: str = "data/messidor-2") -> bool:
    ds = MessidorDataset(data_dir)
    images = ds.discover_images()
    if not images:
        print(f"✗ No images found in {data_dir}")
        return False

    print(f"V Found {len(images)} images in {data_dir}")
    print("Sample validation:")
    print(ds.validate())
    return True


if __name__ == '__main__':
    validate_dataset_dir()

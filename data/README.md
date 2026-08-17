# Data Directory

This project uses HAM10000 for supervised training and PAD-UFES-20 for
cross-domain evaluation. Real datasets require manual download and are not
bundled in the repository.

## HAM10000 (real training data)

1. Download `HAM10000_images_part_1.zip` and `HAM10000_images_part_2.zip` from https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T
2. Download `HAM10000_metadata.csv` from the same source
3. Download `HAM10000_segmentations_lesion_tschandl.zip` for lesion masks
4. Extract all to `data/ham10000/`

Directory structure after extraction:
```
data/ham10000/
  HAM10000_images_part_1/
    ISIC_0024306.jpg
    ...
  HAM10000_images_part_2/
    ISIC_0024307.jpg
    ...
  HAM10000_metadata.csv
  HAM10000_segmentations_lesion_tschandl/
    ISIC_0024306_segmentation.png
    ...
```

Then run:
```bash
python scripts/precompute_abcd_targets.py --data data/ham10000
python src/train.py --data data/ham10000
```

## PAD-UFES-20 (cross-domain evaluation)

1. Download from https://data.mendeley.com/datasets/zr7vgbcyr2/1
2. Extract to `data/pad_ufes20/`

Then run:
```bash
python src/evaluate.py --checkpoint models/best.pt --data data/pad_ufes20
```

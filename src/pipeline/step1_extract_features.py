import os
import sys
import glob
import argparse
import yaml
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.data.clahe_preprocessor import CLAHEPreprocessor
from src.models.vit_extractor import FrozenViTExtractor
from src.utils.seed import set_seed


class SimpleOCTFolderDataset(Dataset):
    """Dataset returning preprocessed OCT B-scan tensors and class labels."""

    def __init__(self, root_dir: str, class_names: list, preprocessor: CLAHEPreprocessor):
        self.samples = []
        self.labels = []
        self.preprocessor = preprocessor
        self.class_to_idx = {cls: idx for idx, cls in enumerate(class_names)}

        for cls_name in class_names:
            cls_dir = os.path.join(root_dir, cls_name)
            if not os.path.exists(cls_dir):
                continue
            image_paths = sorted(
                glob.glob(os.path.join(cls_dir, "*.jpeg"))
                + glob.glob(os.path.join(cls_dir, "*.jpg"))
            )
            for path in image_paths:
                self.samples.append(path)
                self.labels.append(self.class_to_idx[cls_name])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_tensor = self.preprocessor.preprocess_image(self.samples[idx])
        return torch.from_numpy(img_tensor).float(), self.labels[idx]


def extract_features_for_mode(config: dict, model: torch.nn.Module, device: torch.device, use_clahe: bool = True):
    """Run single-pass frozen ViT extraction for train and test splits."""
    mode_tag = "CLAHE" if use_clahe else "No-CLAHE"
    suffix = "" if use_clahe else "_noclahe"

    preprocessor = CLAHEPreprocessor(
        clip_limit=config["data"]["clahe"]["clip_limit"],
        tile_grid_size=tuple(config["data"]["clahe"]["tile_grid_size"]),
        target_size=tuple(config["data"]["image_size"]),
        use_clahe=use_clahe,
    )

    data_dir = config["data"]["data_dir"]
    classes = config["data"]["classes"]
    batch_size = config["feature_extraction"]["batch_size"]
    num_workers = config["feature_extraction"].get("num_workers", 0)
    if sys.platform == "win32" and device.type == "cpu":
        num_workers = 0

    export_cfg = config["feature_extraction"]["export_paths"]
    use_amp = device.type == "cuda"

    for split in ["test", "train"]:
        split_dir = os.path.join(data_dir, split)
        if not os.path.exists(split_dir):
            print(f"Directory not found: {split_dir}")
            continue

        out_feat = export_cfg.get(f"{split}_features{suffix}", f"outputs/features/X_{split}_raw{suffix}.npy")
        out_lbl = export_cfg.get(f"{split}_labels{suffix}", f"outputs/features/y_{split}_raw{suffix}.npy")

        dataset = SimpleOCTFolderDataset(split_dir, classes, preprocessor)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=use_amp,
        )
        print(f"Extracting {split} ({mode_tag}, {len(dataset)} images)...")

        all_feats, all_labels = [], []
        with torch.no_grad():
            for imgs, lbls in tqdm(loader, desc=f"{split} ({mode_tag})"):
                imgs = imgs.to(device)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    feats = model(imgs)
                all_feats.append(feats.cpu().numpy().astype(np.float32))
                all_labels.append(lbls.numpy())

        X = np.vstack(all_feats)
        y = np.concatenate(all_labels)

        os.makedirs(os.path.dirname(out_feat), exist_ok=True)
        np.save(out_feat, X)
        np.save(out_lbl, y)
        print(f"Saved {out_feat} {X.shape}")


def extract_and_export(config_path: str = "configs/config.yaml", mode: str = "clahe"):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    set_seed(config["training"]["random_seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = config["feature_extraction"]["backbone"]
    print(f"Device: {device}, Backbone: {backbone}")

    model = FrozenViTExtractor(
        model_name=backbone,
        pretrained=config["feature_extraction"]["pretrained"],
    ).to(device)

    if mode in ("clahe", "both"):
        extract_features_for_mode(config, model, device, use_clahe=True)
    if mode in ("noclahe", "both"):
        extract_features_for_mode(config, model, device, use_clahe=False)


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()

    parser = argparse.ArgumentParser(description="Single-pass frozen ViT feature extraction.")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to YAML config.")
    parser.add_argument(
        "--mode",
        choices=["clahe", "noclahe", "both"],
        default="clahe",
        help="Extraction mode: 'clahe', 'noclahe' (M0 ablation), or 'both'.",
    )
    args = parser.parse_args()
    extract_and_export(config_path=args.config, mode=args.mode)

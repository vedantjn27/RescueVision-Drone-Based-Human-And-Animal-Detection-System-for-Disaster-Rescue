import os
import yaml
import cv2
import numpy as np
from pathlib import Path
from datasets import load_dataset
from ultralytics import YOLO
from PIL import Image
import torch
from tqdm import tqdm
import json
from datetime import datetime
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from io import BytesIO
import logging

# ---------- Logging ----------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------- Trainer ----------
class DisasterRescueYOLOTrainer:
    def __init__(self, output_dir="disaster_drone_models", model_size="yolov8n.pt"):
        self.output_dir = Path(output_dir)
        self.model_size = model_size
        self.setup_directories()

        # Two classes, fixed ids
        self.classes = {'human': 0, 'animal': 1}

        # COCO → our 2-class mapping (0-based COCO ids)
        self.coco_to_disaster = {
            0: 0,   # person -> human
            14: 1,  # bird -> animal
            15: 1,  # cat -> animal
            16: 1,  # dog -> animal
            17: 1,  # horse -> animal
            18: 1,  # sheep -> animal
            19: 1,  # cow -> animal
            20: 1,  # elephant -> animal
            21: 1,  # bear -> animal
            22: 1,  # zebra -> animal
            23: 1,  # giraffe -> animal
        }

        # Robust, script-free dataset candidates (all Parquet/auto-converted)
        self.dataset_groups = {
            # 1) COCO Parquet (preferred)
            "coco_parquet": [
                {
                    "name": "coco_parquet_detection",
                    "dataset_id": "detection-datasets/coco",
                    "split": "train",
                    "max_samples": 3000,
                    "bbox_fmt": "xyxy",          # viewer shows [x1,y1,x2,y2]
                    "cat_field": "category",      # ints 0..79, 0=person
                    "filter": "coco"              # use coco_to_disaster
                },
                {
                    "name": "coco2017_padilla",
                    "dataset_id": "rafaelpadilla/coco2017",
                    "split": "train",
                    "max_samples": 3000,
                    "bbox_fmt": "xywh",           # dataset card: x,y,w,h
                    "cat_field": "label",         # 1..(inc) → subtract 1
                    "filter": "coco"
                },
            ],
            # 2) OpenImages bboxes (HUGE → stream tiny subset)
            "openimages_bbox": [
                {
                    "name": "openimages_bbox",
                    "dataset_id": "vikhyatk/openimages-bbox",
                    "split": "train",
                    "max_samples": 600,     # keep small
                    "bbox_fmt": "auto",     # we’ll detect per sample
                    "cat_field": None,      # can be str or id; we infer
                    "filter": "openimages"
                }
            ],
            # 3) Tiny sanity dataset (fallback)
            "ultralytics_mini": [
                {
                    "name": "coco8_ultralytics",
                    "dataset_id": "Ultralytics/COCO8",
                    "split": "train",
                    "max_samples": 64,
                    "bbox_fmt": "xywh",     # COCO8 uses xywh in many mirrors; we auto-fix anyway
                    "cat_field": None,
                    "filter": "generic"
                }
            ]
            # Note: We intentionally skip Pascal VOC mirrors on HF because common ones
            # expose segmentation masks, not detection boxes (would yield 0 samples).
        }

        self.training_config = {
            'epochs': 100, 'imgsz': 960, 'batch': 16, 'patience': 15,
            'save_period': 10, 'workers': 4,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'augment': True, 'mixup': 0.0, 'copy_paste': 0.0, 'mosaic': 1.0,
            'hsv_h': 0.015, 'hsv_s': 0.7, 'hsv_v': 0.4, 'degrees': 10.0,
            'translate': 0.1, 'scale': 0.9, 'shear': 2.0, 'perspective': 0.0,
            'flipud': 0.0, 'fliplr': 0.5,
        }

    # ---------- FS ----------
    def setup_directories(self):
        for p in [
            self.output_dir,
            self.output_dir / 'datasets' / 'images' / 'train',
            self.output_dir / 'datasets' / 'images' / 'val',
            self.output_dir / 'datasets' / 'labels' / 'train',
            self.output_dir / 'datasets' / 'labels' / 'val',
            self.output_dir / 'models',
            self.output_dir / 'logs'
        ]:
            p.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory structure at {self.output_dir}")

    def create_data_yaml(self):
        data_config = {
            'path': str(self.output_dir / 'datasets'),
            'train': 'images/train',
            'val': 'images/val',
            'nc': len(self.classes),
            'names': list(self.classes.keys())
        }
        yaml_path = self.output_dir / 'disaster_rescue.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(data_config, f)
        logger.info(f"Created data YAML at {yaml_path}")
        return yaml_path

    # ---------- Dataset streaming ----------
    def _load_streaming(self, ds_id, split):
        # No trust_remote_code anywhere; rely on Parquet auto-converted datasets
        try:
            return load_dataset(ds_id, split=split, streaming=True)
        except Exception as e:
            logger.warning(f"Streaming failed for {ds_id}:{split} ({e})")
            try:
                d = load_dataset(ds_id, split=split)  # non-streaming fallback
                return iter(d)
            except Exception as e2:
                logger.error(f"Failed to load {ds_id}:{split} ({e2})")
                return None

    def _iter_group_candidates(self, group_key):
        for cfg in self.dataset_groups.get(group_key, []):
            ds = self._load_streaming(cfg['dataset_id'], cfg['split'])
            if ds is not None:
                yield cfg, ds

    # ---------- Parsing helpers ----------
    @staticmethod
    def _to_pil(sample_image):
        if isinstance(sample_image, Image.Image):
            return sample_image.convert('RGB')
        if isinstance(sample_image, dict) and 'bytes' in sample_image:
            return Image.open(BytesIO(sample_image['bytes'])).convert('RGB')
        return None

    def _infer_disaster_class(self, ann, group_filter, cat_field):
        # Returns 0 (human), 1 (animal), or None
        # 1) COCO-like (numeric)
        if cat_field and cat_field in ann:
            val = ann[cat_field]
            try:
                cid = int(val)
            except Exception:
                cid = None

            if group_filter == 'coco' and cid is not None:
                # rafaelpadilla uses label 1..; shift to 0-based
                if cat_field == 'label' and cid >= 1:
                    cid = cid - 1
                return self.coco_to_disaster.get(cid, None)

        # 2) OpenImages / generic (strings)
        #   We use broad name matching.
        for key in ('category', 'category_name', 'name', 'label', 'class'):
            if key in ann and isinstance(ann[key], str):
                cname = ann[key].lower()
                if any(k in cname for k in ['person', 'human', 'people']):
                    return 0
                if any(k in cname for k in ['animal', 'dog', 'cat', 'bird', 'horse', 'cow', 'sheep', 'zebra', 'giraffe', 'bear', 'elephant']):
                    return 1
        return None

    @staticmethod
    def _bbox_to_xyxy(bbox, fmt, width, height):
        """
        Accepts list/tuple of 4 numbers.
        fmt: 'xyxy', 'xywh', or 'auto'
        Returns clamped [x1,y1,x2,y2] in pixels or None.
        """
        if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
            return None

        x1 = y1 = x2 = y2 = None
        if fmt == 'xyxy':
            x1, y1, x2, y2 = bbox[:4]
        elif fmt == 'xywh':
            x, y, w, h = bbox[:4]
            x1, y1, x2, y2 = x, y, x + w, y + h
        else:  # auto heuristic
            a, b, c, d = bbox[:4]
            # If c>d and c,d look like max coords bigger than a,b, assume xyxy
            if c > a and d > b and (c <= width + 1) and (d <= height + 1):
                x1, y1, x2, y2 = a, b, c, d
            else:
                x1, y1, x2, y2 = a, b, a + c, b + d

        # Clamp and validate
        x1 = max(0.0, min(float(width), float(x1)))
        y1 = max(0.0, min(float(height), float(y1)))
        x2 = max(0.0, min(float(width), float(x2)))
        y2 = max(0.0, min(float(height), float(y2)))
        if x2 <= x1 or y2 <= y1:
            return None
        return [x1, y1, x2, y2]

    @staticmethod
    def _xyxy_to_yolo(x1, y1, x2, y2, w, h):
        cx = (x1 + x2) / 2.0 / w
        cy = (y1 + y2) / 2.0 / h
        bw = (x2 - x1) / w
        bh = (y2 - y1) / h
        # clamp to [0,1]
        cx = max(0.0, min(1.0, cx))
        cy = max(0.0, min(1.0, cy))
        bw = max(0.0, min(1.0, bw))
        bh = max(0.0, min(1.0, bh))
        return [cx, cy, bw, bh]

    def _convert_sample(self, sample, cfg, image_idx):
        # Get image
        pil = None
        if 'image' in sample:
            pil = self._to_pil(sample['image'])
        elif 'image_url' in sample:
            try:
                r = requests.get(sample['image_url'], timeout=10)
                pil = Image.open(BytesIO(r.content)).convert('RGB')
            except Exception:
                return None
        if pil is None:
            return None

        w, h = pil.size
        # Extract annotations
        objects = None
        for key in ('objects', 'annotations', 'labels'):
            if key in sample:
                objects = sample[key]
                break
        if objects is None:
            return None

        # Objects may be dict of lists or list of dicts—normalize
        anns = []
        if isinstance(objects, dict):
            # Try to zip dict-of-lists
            lens = [len(v) for v in objects.values() if isinstance(v, list)]
            if lens:
                L = min(lens)
                for i in range(L):
                    item = {k: (v[i] if isinstance(v, list) and len(v) > i else v) for k, v in objects.items()}
                    anns.append(item)
        elif isinstance(objects, list):
            anns = objects
        else:
            return None

        yolo_anns = []
        for ann in anns:
            # Which class?
            cls = self._infer_disaster_class(ann, cfg.get('filter'), cfg.get('cat_field'))
            if cls is None:
                continue

            # Which bbox?
            bbox = ann.get('bbox') or ann.get('box') or ann.get('bboxes')
            if bbox is None:
                continue

            # Some datasets store nested per-annotation lists (e.g., lists of lists)
            if isinstance(bbox[0], (list, tuple)):
                # take each candidate box
                for bb in bbox:
                    xyxy = self._bbox_to_xyxy(bb, cfg['bbox_fmt'], w, h)
                    if xyxy:
                        cx, cy, bw, bh = self._xyxy_to_yolo(*xyxy, w, h)
                        yolo_anns.append([cls, cx, cy, bw, bh])
            else:
                xyxy = self._bbox_to_xyxy(bbox, cfg['bbox_fmt'], w, h)
                if xyxy:
                    cx, cy, bw, bh = self._xyxy_to_yolo(*xyxy, w, h)
                    yolo_anns.append([cls, cx, cy, bw, bh])

        if not yolo_anns:
            return None

        return {
            "image": np.array(pil),
            "annotations": yolo_anns,
            "dataset": cfg['name'],
            "idx": image_idx
        }

    # ---------- End-to-end data prep ----------
    def stream_and_prepare_data(self, train_split=0.8, stop_after_first_success=True, min_samples_to_stop=1500):
        logger.info("Starting data streaming and preparation...")
        train_samples, val_samples = [], []
        total = 0
        finished = False

        group_order = ["coco_parquet", "openimages_bbox", "ultralytics_mini"]

        for g in group_order:
            if finished:
                break
            logger.info(f"=== Trying group: {g} ===")
            loaded_any = False

            for cfg, ds in self._iter_group_candidates(g):
                loaded_any = True
                processed = 0
                max_n = cfg['max_samples']
                with ThreadPoolExecutor(max_workers=4) as pool:
                    futures = []
                    try:
                        for idx, sample in enumerate(ds):
                            if processed >= max_n:
                                break
                            futures.append(pool.submit(self._convert_sample, sample, cfg, idx))
                            if len(futures) >= 24:
                                for fut in as_completed(futures):
                                    r = fut.result()
                                    if r:
                                        (train_samples if random.random() < train_split else val_samples).append(r)
                                        processed += 1
                                        total += 1
                                futures.clear()
                                logger.info(f"{cfg['name']}: {processed}/{max_n}")
                    except Exception as e:
                        # <- this prevents OpenImages network hiccups from killing the run
                        logger.warning(f"{cfg['name']} interrupted: {e}. Keeping {processed} samples and continuing.")
                    finally:
                        for fut in as_completed(futures):
                            r = fut.result()
                            if r:
                                (train_samples if random.random() < train_split else val_samples).append(r)
                                processed += 1
                                total += 1

                logger.info(f"Completed {cfg['name']}: {processed} samples")

                # If we’ve already got enough data, bail out of remaining groups
                if stop_after_first_success and total >= min_samples_to_stop and processed > 0:
                    logger.info(f"Collected {total} samples; skipping remaining groups.")
                    finished = True
                    break

            if not loaded_any:
                logger.warning(f"No candidates loaded for group {g} (skipping).")

        logger.info(f"Total processed: {total} (Train: {len(train_samples)}, Val: {len(val_samples)})")
        self.save_processed_data(train_samples, 'train')
        self.save_processed_data(val_samples, 'val')
        return len(train_samples), len(val_samples)


    def save_processed_data(self, samples, split):
        logger.info(f"Saving {len(samples)} {split} samples...")
        images_dir = self.output_dir / 'datasets' / 'images' / split
        labels_dir = self.output_dir / 'datasets' / 'labels' / split
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        for i, s in enumerate(tqdm(samples, desc=f"Saving {split} data")):
            img_path = images_dir / f"{s['dataset']}_{s['idx']:06d}.jpg"
            Image.fromarray(s['image']).save(img_path, quality=95)
            lab_path = labels_dir / f"{s['dataset']}_{s['idx']:06d}.txt"
            with open(lab_path, 'w') as f:
                for a in s['annotations']:
                    f.write(f"{a[0]} {a[1]:.6f} {a[2]:.6f} {a[3]:.6f} {a[4]:.6f}\n")
        logger.info(f"Saved {len(samples)} {split} samples")

    # ---------- Training / validation ----------
    def train_model(self):
        logger.info("Starting model training...")
        data_yaml_path = self.create_data_yaml()
        model = YOLO(self.model_size)
        args = {
            'data': str(data_yaml_path),
            'epochs': self.training_config['epochs'],
            'imgsz': self.training_config['imgsz'],
            'batch': self.training_config['batch'],
            'device': self.training_config['device'],
            'patience': self.training_config['patience'],
            'save_period': self.training_config['save_period'],
            'workers': self.training_config['workers'],
            'project': str(self.output_dir / 'models'),
            'name': 'disaster_rescue_v1',
            'exist_ok': True,
            'pretrained': True,
            'optimizer': 'AdamW',
            'lr0': 0.01, 'lrf': 0.01, 'momentum': 0.937, 'weight_decay': 5e-4,
            'warmup_epochs': 3, 'warmup_momentum': 0.8, 'warmup_bias_lr': 0.1,
            'box': 7.5, 'cls': 0.5, 'dfl': 1.5, 'pose': 12.0, 'kobj': 1.0,
            'label_smoothing': 0.0, 'nbs': 64,
            'hsv_h': self.training_config['hsv_h'],
            'hsv_s': self.training_config['hsv_s'],
            'hsv_v': self.training_config['hsv_v'],
            'degrees': self.training_config['degrees'],
            'translate': self.training_config['translate'],
            'scale': self.training_config['scale'],
            'shear': self.training_config['shear'],
            'perspective': self.training_config['perspective'],
            'flipud': self.training_config['flipud'],
            'fliplr': self.training_config['fliplr'],
            'mosaic': self.training_config['mosaic'],
            'mixup': self.training_config['mixup'],
            'copy_paste': self.training_config['copy_paste'],
            'verbose': True, 'seed': 42, 'deterministic': True,
            'single_cls': False, 'rect': False, 'cos_lr': False,
            'close_mosaic': 10, 'resume': False, 'amp': True,
            'fraction': 1.0, 'profile': False, 'overlap_mask': True,
            'mask_ratio': 4, 'dropout': 0.0, 'val': True,
        }
        results = model.train(**args)
        self.save_training_results(results)
        return model, results

    def save_training_results(self, results):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = {
            'timestamp': ts,
            'model_size': self.model_size,
            'classes': self.classes,
            'training_config': self.training_config,
            'results': str(results) if results else None
        }
        p = self.output_dir / 'logs' / f'training_results_{ts}.json'
        with open(p, 'w') as f:
            json.dump(out, f, indent=2)
        logger.info(f"Training results saved to {p}")

    def validate_model(self, model_path=None):
        if model_path is None:
            model_path = self.output_dir / 'models' / 'disaster_rescue_v1' / 'weights' / 'best.pt'
        if not model_path.exists():
            logger.error(f"Model not found at {model_path}")
            return None
        logger.info(f"Validating model: {model_path}")
        model = YOLO(str(model_path))
        return model.val(
            data=str(self.output_dir / 'disaster_rescue.yaml'),
            imgsz=640, batch=16, save_json=True, conf=0.001, iou=0.6,
            max_det=300, half=True,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            plots=True, verbose=True
        )

    # ---------- Synthetic fallback ----------
    def create_synthetic_disaster_data(self):
        logger.info("Creating synthetic disaster rescue training data...")
        train_n, val_n = 800, 160  # smaller to avoid long runs
        for split, n in [('train', train_n), ('val', val_n)]:
            images_dir = self.output_dir / 'datasets' / 'images' / split
            labels_dir = self.output_dir / 'datasets' / 'labels' / split
            images_dir.mkdir(parents=True, exist_ok=True)
            labels_dir.mkdir(parents=True, exist_ok=True)
            for i in tqdm(range(n), desc=f"Synth {split}"):
                img = self.generate_synthetic_disaster_image()
                anns = self.generate_synthetic_annotations()
                if not anns:
                    continue
                cv2.imwrite(str(images_dir / f"synthetic_{split}_{i:06d}.jpg"), img)
                with open(labels_dir / f"synthetic_{split}_{i:06d}.txt", "w") as f:
                    for a in anns:
                        f.write(f"{a[0]} {a[1]:.6f} {a[2]:.6f} {a[3]:.6f} {a[4]:.6f}\n")
        logger.info("Synthetic data generation completed")

    @staticmethod
    def generate_synthetic_disaster_image():
        h, w = 640, 640
        img = np.random.randint(80, 120, (h, w, 3), dtype=np.uint8)
        noise = np.random.normal(0, 20, (h, w, 3)).astype(np.int32)
        img = np.clip(img.astype(np.int32) + noise, 0, 255).astype(np.uint8)
        for _ in range(random.randint(2, 5)):
            pt1 = (random.randint(0, w//2), random.randint(0, h//2))
            pt2 = (random.randint(pt1[0], w), random.randint(pt1[1], h))
            color = tuple(int(random.randint(60, 100)) for _ in range(3))
            cv2.rectangle(img, pt1, pt2, color, -1)
        for _ in range(random.randint(1, 3)):
            center = (random.randint(0, w), random.randint(0, h))
            radius = random.randint(10, 50)
            color = tuple(int(random.randint(50, 90)) for _ in range(3))
            cv2.circle(img, center, radius, color, -1)
        return img

    @staticmethod
    def generate_synthetic_annotations():
        anns = []
        k = random.randint(1, 4)
        for _ in range(k):
            cls = 0 if random.random() < 0.7 else 1
            cx = random.uniform(0.15, 0.85)
            cy = random.uniform(0.15, 0.85)
            if cls == 0:
                bw = random.uniform(0.08, 0.25)
                bh = random.uniform(0.15, 0.4)
            else:
                bw = random.uniform(0.05, 0.2)
                bh = random.uniform(0.08, 0.25)
            cx = max(bw/2, min(1 - bw/2, cx))
            cy = max(bh/2, min(1 - bh/2, cy))
            anns.append([cls, cx, cy, bw, bh])
        return anns

    def count_synthetic_data(self):
        train_dir = self.output_dir / 'datasets' / 'images' / 'train'
        val_dir = self.output_dir / 'datasets' / 'images' / 'val'
        return len(list(train_dir.glob('*.jpg'))), len(list(val_dir.glob('*.jpg')))

    # ---------- Orchestration ----------
    def run_complete_training_pipeline(self):
        logger.info("=== DISASTER RESCUE YOLO TRAINING PIPELINE ===")
        logger.info(f"Output dir: {self.output_dir} | Model: {self.model_size} | Device: {self.training_config['device']}")
        try:
            logger.info("\n--- Step 1: Data Preparation ---")
            tr, va = self.stream_and_prepare_data()
            if tr == 0:
                logger.warning("No data gathered from HF. Falling back to synthetic data...")
                self.create_synthetic_disaster_data()
                tr, va = self.count_synthetic_data()
            if tr == 0:
                logger.error("No training data available. Exiting.")
                return None

            logger.info("\n--- Step 2: Model Training ---")
            model, _ = self.train_model()

            logger.info("\n--- Step 3: Model Validation ---")
            self.validate_model()

            final_model_path = self.output_dir / 'models' / 'disaster_rescue_v1' / 'weights' / 'best.pt'
            if final_model_path.exists():
                logger.info("✅ Training completed successfully!")
                logger.info(f"Best model: {final_model_path}")
                logger.info(f"Training samples: {tr} | Validation samples: {va}")
                return final_model_path
            else:
                logger.error("Training finished but best.pt not found")
                return None

        except Exception as e:
            logger.error(f"Training pipeline failed: {e}", exc_info=True)
            return None

# ---------- CLI ----------
def main():
    print("=== DISASTER RESCUE YOLO MODEL TRAINING ===")
    model_options = ['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt']
    for i, m in enumerate(model_options, 1): print(f"  {i}. {m}")
    choice = input(f"\nSelect model size (1-{len(model_options)}, default=1): ").strip() or '1'
    selected_model = model_options[max(0, min(len(model_options)-1, int(choice)-1))]
    trainer = DisasterRescueYOLOTrainer(output_dir="disaster_drone_models", model_size=selected_model)

    print("\nTraining Configuration:")
    ep = input("Number of epochs (default=100): ").strip()
    bs = input("Batch size (default=16): ").strip()
    trainer.training_config['epochs'] = int(ep) if ep else 100
    trainer.training_config['batch'] = int(bs) if bs else 16
    print(f"\nFinal config: model={selected_model}, epochs={trainer.training_config['epochs']}, batch={trainer.training_config['batch']}, device={trainer.training_config['device']}")
    if input("\nStart training? (y/N): ").strip().lower() != 'y':
        print("Training cancelled.")
        return

    result = trainer.run_complete_training_pipeline()
    if result:
        print("\n" + "="*60)
        print("🎉 TRAINING COMPLETED SUCCESSFULLY! 🎉")
        print("="*60)
        print(f"Trained model: {result}")
        print(f"Run detection with: python your_detection_script.py --model \"{result}\"")
    else:
        print("\n❌ Training failed. Check logs for details.")

if __name__ == "__main__":
    main()

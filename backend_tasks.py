# -*- coding: utf-8 -*-
"""
@file: backend_tasks.py
@description: Core scientific processing logic. Contains classes for:
              1. PdfConverter: Multiprocess PDF rendering.
              2. ObjectDetector: YOLOv8/11 inference logic.
              3. ImageClassifier: ResNet/CNN filtering logic.
@author: Yaxuan Shen
@date: 2025-10-01
"""

import os
import shutil
import csv
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError
from PIL import Image

# Third-party scientific libraries
import fitz  # PyMuPDF
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from ultralytics import YOLO

from config import Config
from backend_signals import WorkerSignals


class TaskRunner:
    """Base class for all background tasks."""

    def __init__(self, signals: WorkerSignals):
        self.signals = signals
        self.stop_requested = False

    def request_stop(self):
        self.stop_requested = True


class PdfConverter(TaskRunner):
    """Handles high-performance PDF to Image conversion."""

    def __init__(self, signals: WorkerSignals, dpi: int = 3):
        super().__init__(signals)
        self.dpi = dpi

    @staticmethod
    def _convert_page_worker(pdf_path, pages_to_convert, dpi, output_folder):
        """Static worker method for ProcessPoolExecutor context."""
        count = 0
        error_msg = None
        try:
            with fitz.open(pdf_path) as doc:
                for page_info in pages_to_convert:
                    page = doc.load_page(page_info['page_num'])
                    # Set resolution matrix
                    mat = fitz.Matrix(dpi, dpi)
                    try:
                        # Render to pixmap (RGB)
                        pix = page.get_pixmap(matrix=mat, alpha=False, colorspace=fitz.csRGB)
                        pix.save(page_info['img_path'])
                        count += 1
                    except Exception as e:
                        error_msg = f"Page {page_info['page_num'] + 1} error: {e}"
        except Exception as e:
            error_msg = f"File error: {e}"
        return count, error_msg

    def run(self):
        self.signals.log(f"▶️ Starting PDF Conversion Engine (DPI: {self.dpi * 100})")

        # 1. Scan Phase
        all_pdfs = [os.path.join(Config.UPLOAD_FOLDER, f) for f in sorted(os.listdir(Config.UPLOAD_FOLDER))
                    if f.lower().endswith('.pdf')]

        if not all_pdfs:
            self.signals.log("⚠️ No PDF files found in Source folder.", "warn")
            self.signals.progress("conversion", 0)
            return

        tasks_by_pdf = {}
        total_pages_found = 0

        self.signals.log("🔍 Scanning PDFs for incremental update...", "info")
        for pdf_path in all_pdfs:
            if self.stop_requested: break
            try:
                doc = fitz.open(pdf_path)
                num_pages = len(doc)
                doc.close()

                base_name = os.path.splitext(os.path.basename(pdf_path))[0]
                needed_pages = []

                for p_num in range(num_pages):
                    img_name = f"{base_name}_{p_num + 1}.jpg"
                    img_path = os.path.join(Config.PAGES_FOLDER, img_name)
                    if not os.path.exists(img_path):
                        needed_pages.append({'page_num': p_num, 'img_path': img_path})

                if needed_pages:
                    tasks_by_pdf[pdf_path] = needed_pages
                    total_pages_found += len(needed_pages)
            except Exception as e:
                self.signals.log(f"❌ Scan failed for {os.path.basename(pdf_path)}: {e}", "error")

        if total_pages_found == 0:
            self.signals.log("✅ All pages are up to date.", "success")
            self.signals.progress("conversion", 100)
            return

        # 2. Execution Phase
        self.signals.log(f"🚀 Processing {total_pages_found} pages using {Config.PDF_MAX_WORKERS} cores...", "info")
        processed_count = 0
        total_pdfs = len(tasks_by_pdf)

        with ProcessPoolExecutor(max_workers=Config.PDF_MAX_WORKERS) as executor:
            futures = {
                executor.submit(self._convert_page_worker, pdf, pages, self.dpi, Config.PAGES_FOLDER): pdf
                for pdf, pages in tasks_by_pdf.items()
            }

            for future in as_completed(futures):
                if self.stop_requested:
                    executor.shutdown(wait=False, cancel_futures=True)
                    break

                pdf_name = os.path.basename(futures[future])
                try:
                    # Dynamic timeout based on workload
                    timeout = 120 + len(tasks_by_pdf[futures[future]]) * 15
                    converted, err = future.result(timeout=timeout)

                    if err:
                        self.signals.log(f"⚠️ {pdf_name}: {err}", "warn")
                    else:
                        processed_count += 1
                        progress = int((processed_count / total_pdfs) * 100)
                        self.signals.progress("conversion", progress)
                        self.signals.log(f"✅ Converted {pdf_name} ({converted} pages)", "info")

                except TimeoutError:
                    self.signals.log(f"❌ Timeout: {pdf_name} took too long.", "error")
                except Exception as e:
                    self.signals.log(f"❌ Critical error {pdf_name}: {e}", "error")

        if not self.stop_requested:
            self.signals.log("🎉 PDF Conversion Pipeline Completed.", "success")
            self.signals.progress("conversion", 100)
        else:
            self.signals.log("🛑 Operation terminated by user.", "warn")


class ObjectDetector(TaskRunner):
    """Runs YOLO inference to extract fossil regions of interest."""

    def __init__(self, signals: WorkerSignals, model_name: str, confidence: float):
        super().__init__(signals)
        self.model_name = model_name
        self.conf = confidence
        self.model_path = Config.YOLO_MODELS.get(model_name)

    def run(self):
        if not os.path.exists(self.model_path):
            self.signals.log(f"❌ Model file missing: {self.model_path}", "error")
            return

        self.signals.log(f"▶️ Initializing YOLO Detector ({self.model_name}) on {Config.DEVICE}...")

        try:
            model = YOLO(self.model_path)
            self.signals.log("🧠 Model loaded successfully.", "info")
        except Exception as e:
            self.signals.log(f"❌ Failed to load model: {e}", "error")
            return

        # Prepare Logging / Resume logic
        det_log_path = os.path.join(Config.LOGS_FOLDER, "detection_log.csv")
        scanned_log_path = os.path.join(Config.LOGS_FOLDER, "scanned_pages_log.txt")

        processed_files = set()
        if os.path.exists(scanned_log_path):
            with open(scanned_log_path, 'r', encoding='utf-8') as f:
                processed_files = set(line.strip() for line in f)

        # ID Counter recovery
        rad_counter = 1
        if os.path.exists(det_log_path) and os.path.getsize(det_log_path) > 0:
            try:
                df = pd.read_csv(det_log_path)
                if 'temp_image_id' in df.columns:
                    ids = df['temp_image_id'].str.extract(r'rad_(\d+)')[0].dropna().astype(int)
                    if not ids.empty:
                        rad_counter = ids.max() + 1
            except Exception:
                pass

        # Identify pending work
        all_imgs = sorted([f for f in os.listdir(Config.PAGES_FOLDER) if f.lower().endswith(('.jpg', '.png'))])
        to_process = [f for f in all_imgs if f not in processed_files]

        if not to_process:
            self.signals.log("✅ Detection is up to date.", "success")
            self.signals.progress("detection", 100)
            return

        self.signals.log(f"Processing {len(to_process)} new page images...", "info")

        # File handles
        f_det = open(det_log_path, 'a', newline='', encoding='utf-8')
        writer = csv.writer(f_det)
        if os.path.getsize(det_log_path) == 0:
            writer.writerow(['temp_image_id', 'source_paper', 'source_page', 'bbox', 'confidence'])
        f_scan = open(scanned_log_path, 'a', encoding='utf-8')

        # Batch Processing
        batch_size = Config.DETECTION_BATCH_SIZE
        processed_count = 0

        for i in range(0, len(to_process), batch_size):
            if self.stop_requested: break

            batch_files = to_process[i: i + batch_size]
            batch_paths = [os.path.join(Config.PAGES_FOLDER, f) for f in batch_files]

            try:
                results = model(batch_paths, verbose=False, device=Config.DEVICE, stream=False)

                # Iterate results
                img_cache = {}  # Lazy load images for cropping

                for j, res in enumerate(results):
                    filename = batch_files[j]
                    src_path = batch_paths[j]

                    # Mark scanned
                    f_scan.write(f"{filename}\n")

                    # Metadata extraction
                    parts = os.path.splitext(filename)[0].rsplit('_', 1)
                    s_paper = parts[0]
                    s_page = parts[1] if len(parts) > 1 else '?'

                    # Process boxes
                    for box in res.boxes:
                        conf = float(box.conf[0])
                        if conf >= self.conf:
                            coords = [int(x) for x in box.xyxy[0].tolist()]

                            if src_path not in img_cache:
                                img_cache[src_path] = Image.open(src_path)

                            # Crop
                            crop_id = f"rad_{rad_counter:07d}"
                            rad_counter += 1

                            try:
                                crop_img = img_cache[src_path].crop(coords)
                                crop_img.save(os.path.join(Config.CROPPED_FOLDER, f"{crop_id}.jpg"))
                                writer.writerow([crop_id, s_paper, s_page, str(coords), f"{conf:.4f}"])
                            except Exception as e:
                                print(f"Crop error: {e}")

            except Exception as e:
                self.signals.log(f"❌ Batch Inference Error: {e}", "error")

            processed_count += len(batch_files)
            self.signals.progress("detection", int((processed_count / len(to_process)) * 100))

            # Flush buffers
            if i % (batch_size * 2) == 0:
                f_det.flush()
                f_scan.flush()

        f_det.close()
        f_scan.close()

        if not self.stop_requested:
            self.signals.log("🎉 Detection Pipeline Completed.", "success")
        else:
            self.signals.log("🛑 Detection Aborted.", "warn")


class ImageClassifier(TaskRunner):
    """Runs ResNet/MobileNet to filter out noise from extracted images."""

    def __init__(self, signals: WorkerSignals, model_name: str):
        super().__init__(signals)
        self.model_name = model_name
        self.model_path = Config.CLASSIFIER_MODELS.get(model_name)

        # Standard ImageNet normalization
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def run(self):
        if not os.path.exists(self.model_path):
            self.signals.log(f"❌ Model file missing: {self.model_path}", "error")
            return

        self.signals.log(f"▶️ Initializing Classifier ({self.model_name})...")

        try:
            num_classes = len(Config.CLASSIFIER_CLASS_NAMES)
            # Initialize Architecture
            model = models.resnet50(weights=None)
            model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

            # Load Weights
            state_dict = torch.load(self.model_path, map_location=Config.DEVICE)
            model.load_state_dict(state_dict)
            model.to(Config.DEVICE)
            model.eval()
            self.signals.log("🧠 Classifier loaded.", "info")
        except Exception as e:
            self.signals.log(f"❌ Classifier Load Error: {e}", "error")
            return

        imgs = [f for f in os.listdir(Config.CROPPED_FOLDER) if f.lower().endswith('.jpg')]
        total = len(imgs)

        if total == 0:
            self.signals.log("⚠️ No candidate images to filter.", "warn")
            self.signals.progress("classification", 100)
            return

        self.signals.log(f"Filtering {total} candidates...", "info")

        batch_size = Config.CLASSIFICATION_BATCH_SIZE
        rejected_count = 0
        processed_count = 0
        rejected_ids = []

        with torch.no_grad():
            for i in range(0, total, batch_size):
                if self.stop_requested: break

                batch_files = imgs[i: i + batch_size]
                batch_tensors = []
                valid_files = []

                # Preprocessing
                for fname in batch_files:
                    try:
                        p = os.path.join(Config.CROPPED_FOLDER, fname)
                        img = Image.open(p).convert('RGB')
                        batch_tensors.append(self.transform(img))
                        valid_files.append(fname)
                    except Exception:
                        continue

                if not batch_tensors: continue

                # Inference
                input_tensor = torch.stack(batch_tensors).to(Config.DEVICE)
                outputs = model(input_tensor)
                _, preds = torch.max(outputs, 1)

                # Post-processing
                for idx, class_idx in enumerate(preds):
                    cls_name = Config.CLASSIFIER_CLASS_NAMES[class_idx.item()]
                    fname = valid_files[idx]

                    if cls_name != Config.POSITIVE_CLASS_NAME:
                        # Reject: Move to rejected folder
                        src = os.path.join(Config.CROPPED_FOLDER, fname)
                        dst = os.path.join(Config.REJECTED_FOLDER, fname)
                        try:
                            shutil.move(src, dst)
                            rejected_count += 1
                            rejected_ids.append(os.path.splitext(fname)[0])
                        except OSError:
                            pass

                processed_count += len(batch_files)
                self.signals.progress("classification", int((processed_count / total) * 100))

        # Cleanup Logs
        if rejected_ids:
            self._clean_csv_log(rejected_ids)

        if not self.stop_requested:
            self.signals.log(f"🎉 Classification Complete. Removed {rejected_count} artifacts.", "success")
        else:
            self.signals.log("🛑 Classification Stopped.", "warn")

    def _clean_csv_log(self, rejected_ids):
        log_path = os.path.join(Config.LOGS_FOLDER, "detection_log.csv")
        if os.path.exists(log_path):
            try:
                df = pd.read_csv(log_path)
                clean_df = df[~df['temp_image_id'].isin(rejected_ids)]
                clean_df.to_csv(log_path, index=False)
                self.signals.log("📄 Updated metadata log.", "info")
            except Exception as e:
                self.signals.log(f"⚠️ Log update failed: {e}", "warn")
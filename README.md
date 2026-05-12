# Radiolarian Image Extraction & Analysis System

An automated, end-to-end desktop application designed to extract and analyze radiolarian fossil images from academic PDF literature. Integrates PDF rendering, YOLO object detection, and CNN image classification into a user-friendly GUI.

---

## Project Structure

```
Radiolarian-Image-Extractor/
├── main.py                      # Application entry point
├── config.py                    # Global configuration & paths
├── backend_tasks.py             # Core processing logic
├── backend_signals.py           # Thread communication
├── ui_main.py                   # GUI controller
├── ui_panels.py                 # Tab interface
├── ui_widgets.py                # Custom UI components
├── utils.py                     # File system utilities
├── models/                      # Model weights
│   ├── yolo/                    # YOLO detection models
│   └── classifiers/             # Image classification models
└── model_training/              # Model training pipeline
    └── image_classification/     # Image classification module
```

---

## Architecture

### Core Logic
- Multithreaded processing for PDF conversion and AI inference
- Incremental processing with resume capability
- Thread-safe queue-based communication

### UI Layer
- Built with tkinter
- Responsive dashboard with progress tracking
- Paginated results gallery

### Backend Components

| Component | Description |
|-----------|-------------|
| PdfConverter | Renders high-resolution images from PDFs using PyMuPDF |
| ObjectDetector | Uses YOLOv11/v8 for fossil localization |
| ImageClassifier | Filters noise using ResNet50/MobileNet |

---

## Installation

### Prerequisites
- Python 3.9+
- CUDA-capable GPU (Recommended)

### Steps

```bash
# Clone the repository
git clone https://github.com/syxscott/Radiolarian-Image-Extractor.git
cd Radiolarian-Image-Extractor

# Install dependencies
pip install -r requirements.txt

# Setup model directories
mkdir -p models/yolo models/classifiers

# Place your YOLO weights (e.g., yolo11x.pt) in models/yolo/
# Place your classifier weights (e.g., resnet50.pth) in models/classifiers/
```

---

## Usage

### GUI Application

```bash
python main.py
```

### Workflow

1. **Import** - Load PDF documents in the "Pipeline" tab
2. **Convert** - Convert PDF pages to images (multi-core processing)
3. **Detect** - Run YOLO detection to crop fossil candidates
4. **Filter** - Apply classifier to remove non-fossil noise
5. **Analyze** - Review results in the "Gallery" tab

---

## Model Training

Train custom classifiers with class weights and multi-metric evaluation:

```bash
cd model_training/image_classification

# Step 1: Prepare data
# Place images in classification_dataset/<class_name>/ folders

# Step 2: Split data
python data_split.py

# Step 3: Train models
python train.py

# Step 4: Evaluate models
python evaluate.py
```

### Supported Models
- ResNet (18, 50)
- EfficientNet (B0, B2)
- MobileNet (V2, V3)
- VGG16
- DenseNet121
- ResNeXt50
- ShuffleNet
- SqueezeNet

### Configuration
Edit `config.py` to adjust hyperparameters:
- Epochs, batch size, learning rate
- Model list to train
- Train/validation split ratio

---

## Data Directories

| Directory | Description |
|-----------|-------------|
| `01_Source_PDFs/` | Original PDF files |
| `02_Processed_Pages/` | Converted page images |
| `03_Candidate_Images/` | Detected fossil candidates |
| `03a_Rejected_Images/` | Filtered noise |
| `05_Logs/` | Processing logs |

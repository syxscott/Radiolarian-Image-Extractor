Radiolarian Image Extraction & Analysis System

An automated, end-to-end desktop application designed to extract and analyze radiolarian fossil images from academic PDF literature. This tool integrates PDF rendering, YOLO object detection, and ResNet image classification into a user-friendly GUI.

🏗️ Architecture

The project is structured into modular components for scalability and maintainability:

Core Logic: Multithreaded processing for PDF conversion and AI inference.

UI Layer: Built with tkinter, featuring a responsive dashboard and results gallery.

Backend:

PdfConverter: Renders high-resolution images from PDFs.

ObjectDetector: Uses YOLOv11/v8 for fossil localization.

ImageClassifier: Filters noise using ResNet50/MobileNet.

🚀 Installation

Prerequisites

Python 3.9 or higher

CUDA-capable GPU (Recommended for faster inference)

Steps

Clone the repository

git clone [https://github.com/syxscott/Radiolarian-Image-Extractor.git](https://github.com/YourUsername/Radiolarian-Image-Extractor.git)
cd Radiolarian-Image-Extractor


Install dependencies

pip install -r requirements.txt


Setup Models

Create a models/yolo directory and place your YOLO weights (e.g., yolo11x.pt) inside.

Create a models/classifiers directory and place your classifier weights (e.g., resnet50.pth) inside.

Note: The application will verify these paths on startup.

🖥️ Usage

Run the main entry point:

python main.py


Workflow

Import: Load your PDF documents in the "Processing Pipeline" tab.

Convert: Convert PDF pages to images (supports multi-core processing).

Detect: Run YOLO detection to crop fossil candidates.

Filter: Apply the classifier to remove non-fossil noise (text, graphs).

Analyze: Review the clean dataset in the "Results Gallery".

📂 Project Structure

├── main.py              # Application Entry Point
├── config.py            # Global Configuration & Paths
├── backend_tasks.py     # Scientific Processing Logic
├── backend_signals.py   # Thread Communication
├── ui_main.py           # GUI Controller
├── ui_panels.py         # Tab Interface Implementation
├── ui_widgets.py        # Custom UI Components
└── utils.py             # File System Utilities



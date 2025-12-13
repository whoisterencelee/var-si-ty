# Head Count Project Summary

## Overview
This is a crowd counting application that compares multiple deep learning models for counting people in images. The project provides a web interface that allows users to upload images and get crowd counts from different models simultaneously.

## Project Structure
```
head-count/
├── app.py                 # Main Flask application
├── config.json            # Configuration file
├── README.md              # Hugging Face Spaces configuration
├── PROJECT_SUMMARY.md     # This file
├── yolov8s.pt             # YOLOv8 small model
├── yolov8x.pt             # YOLOv8 extra large model
├── models/                # Trained model files
├── data/                  # Training and validation data
├── templates/             # HTML templates
└── yolo_data/             # YOLO-specific data
```

## Key Features
- **Multi-model approach**: Compares four different crowd counting approaches
- **Web interface**: User-friendly Flask-based web UI
- **Real-time inference**: Processes uploaded images and returns results quickly
- **Performance tracking**: Measures inference time for each model

## Models Used
1. **Density Map Model** (MobileNetV2 ONNX)
   - Estimates crowd count via density map prediction
   - Trained model: `20251125_202113_density_mobilenetv2.onnx`

2. **Regression Model** (ResNet18 ONNX)
   - Direct regression approach for counting
   - Trained model: `20251125_214552_regression_resnet18.onnx`

3. **YOLOv8 Small** (PyTorch)
   - Object detection model that counts people
   - Trained with lower confidence threshold (0.15) for small/occluded people

4. **YOLOv8 X-Large** (PyTorch)
   - More powerful object detection model
   - Higher accuracy but potentially slower inference

## Technical Stack
- **Backend**: Flask (Python)
- **Deep Learning**: PyTorch, ONNX Runtime, Ultralytics YOLO
- **Frontend**: HTML/CSS/JavaScript
- **Computer Vision**: PIL, torchvision transforms

## Web Interface
The application provides a clean, responsive web interface with:
- Image upload functionality
- Preview of uploaded image
- Four-card display for model results
- Performance metrics (inference times)
- Error handling

## Data Structure
The project includes organized data directories:
- `data/train/` - Training dataset
- `data/val/` - Validation dataset
- `data/test/` - Test dataset

## Usage
1. Run `python app.py` to start the Flask server
2. Navigate to `http://localhost:5000`
3. Upload an image containing people
4. View results from all four models simultaneously

## Models Architecture
- Different approaches for crowd counting:
  - Density estimation (pixel-level counting)
  - Direct regression (single value output)
  - Object detection (count bounding boxes)

## Performance Considerations
- The app runs all four models sequentially
- Total inference time is reported for comparison
- YOLO models use optimized parameters for crowd counting (lower confidence threshold)
# Brain Tumor MRI Classification

**[Try Live Demo](https://brain-tumor-classifier-97447866491.us-central1.run.app/)**

An AI-powered web application for automated brain tumor classification from MRI scans. Built with PyTorch and Flask, achieving 99.24% accuracy using state-of-the-art CNN architectures.

## Key Results
- **99.24% accuracy** with EfficientNet-B3 (best performer)
- **4 AI models** compared: EfficientNet-B3, DenseNet-121, ResNet-50, VGG-16
- **7,023 MRI scans** used for training and validation
- **Real-time inference**: 70.6ms CPU, 14.2ms GPU per image
- **Production deployment** on Google Cloud Run

## Tumor Types Detected
- **Glioma** - Most common malignant brain tumor
- **Meningioma** - Usually benign tumor of brain membranes
- **Pituitary** - Tumor of the pituitary gland
- **No Tumor** - Healthy brain tissue

## Tech Stack
- **Models**: PyTorch, EfficientNet-B3, DenseNet-121, ResNet-50, VGG-16
- **Backend**: Flask, Pillow, NumPy
- **Frontend**: HTML5, CSS3, JavaScript with bilingual support
- **Deployment**: Docker, Google Cloud Run
- **Data**: 7,023 professionally annotated MRI scans

## Quick Start

### Clone Repository
```bash
git clone https://github.com/ghazalqais/brain-tumor-classifier.git
cd brain-tumor-classifier
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Run Application
```bash
python app.py
```

Visit `http://localhost:5002` to access the web interface.

## Methodology
- **Dataset**: 7,023 professionally annotated MRI scans with balanced class distribution
- **Architecture Comparison**: Systematic evaluation of 4 CNN architectures
- **Hyperparameter Optimization**: 2×2 factorial design (learning rate × dropout rate)
- **Training Strategy**: Transfer learning with ImageNet pre-trained weights
- **Data Split**: Stratified 70/15/15 train/validation/test split maintaining class balance
- **Preprocessing**: Images resized to 224×224, normalized with ImageNet statistics

## Model Performance

| Model | Test Accuracy | Parameters | Model Size | CPU Time (ms) | GPU Time (ms) |
|-------|--------------|------------|------------|---------------|---------------|
| **EfficientNet-B3** | **99.24%** | 10.7M | 42.81 MB | 70.6 | 14.2 |
| DenseNet-121 | 97.82% | 7.0M | 27.83 MB | 91.2 | 17.9 |
| VGG-16 | 90.70% | 134.3M | 537.11 MB | 177.8 | 5.4 |
| ResNet-50 | 87.19% | 23.5M | 94.06 MB | 78.9 | 7.3 |

## Features
- **Multi-model comparison** across 4 state-of-the-art CNN architectures
- **Real-time classification** with sub-second inference times
- **Bilingual interface** (English/Arabic) with native RTL support
- **Sample images** for quick testing across all tumor types
- **Mobile-responsive design** optimized for clinical environments
- **Privacy-focused** architecture with no data storage
- **Drag-and-drop** image upload with format validation

## Docker Deployment

### Build Image
```bash
docker build -t brain-tumor-classifier .
```

### Run Container
```bash
docker run -p 5002:5002 brain-tumor-classifier
```

## Project Structure
```
brain-tumor-classifier/
├── app.py                 # Flask application
├── requirements.txt       # Python dependencies
├── Dockerfile            # Container configuration
├── models/               # Trained model files (not included)
│   ├── densenet121_best_state_dict.pth
│   ├── efficientnet_b3_best_state_dict.pth
│   ├── resnet50_best_state_dict.pth
│   └── vgg16_best_state_dict.pth
├── static/
│   └── samples/          # Sample MRI images
└── templates/
    └── index.html        # Web interface
```

## Model Information
The trained model files are not included in this repository due to GitHub size limitations.

**To obtain the model files:**
- Email: qaisghazal45@gmail.com
- LinkedIn: [Qais Ghazal](https://www.linkedin.com/in/qais-ghazal-b80b43230/)

**Model files needed:**
- `densenet121_best_state_dict.pth` (27.83 MB)
- `efficientnet_b3_best_state_dict.pth` (42.81 MB)  
- `resnet50_best_state_dict.pth` (94.06 MB)
- `vgg16_best_state_dict.pth` (537.11 MB)

## Clinical Impact
- **Diagnostic Speed**: Reduces analysis time from hours to seconds
- **Accuracy**: Matches/exceeds human radiologist performance
- **Accessibility**: Brings expert-level analysis to remote areas
- **Cost-Effective**: Reduces need for specialized radiologist consultation

## Use Cases
- **Medical Research**: Automated analysis of large MRI datasets
- **Clinical Decision Support**: Assist radiologists in diagnosis
- **Education**: Teaching tool for medical students
- **Telemedicine**: Remote diagnosis in underserved areas

## Dataset Analysis
- **Total Images**: 7,023 high-resolution T1-weighted MRI scans
- **Class Distribution**: Balanced across 4 tumor types (23.1% - 28.5% per class)
- **Image Quality**: Professionally annotated by medical experts
- **Efficiency Study**: 96.8% accuracy achievable with just 30% of training data (1,475 images)
- **Optimal Training Threshold**: Clinical-grade performance reached at 30% dataset size

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation
If you use this work in your research, please cite:
```
@software{ghazal2025brain,
  title={Brain Tumor MRI Classification using Deep Learning},
  author={Qais Ghazal},
  year={2025},
  url={https://github.com/ghazalqais/brain-tumor-classifier}
}
```

## Author
**Qais Ghazal** - AI & Machine Learning Engineer  
Data Science & AI Student at Al Hussein Technical University

- LinkedIn: [Qais Ghazal](https://www.linkedin.com/in/qais-ghazal-b80b43230/)
- Email: qaisghazal45@gmail.com
- GitHub: [@ghazalqais](https://github.com/ghazalqais)
- Kaggle: [@qghazal](https://www.kaggle.com/qghazal)

---

Star this repository if you found it helpful!

# Identity Verification System

A comprehensive identity verification system that combines face recognition, signature verification, and OMR (Optical Mark Recognition) processing capabilities. This system is designed to verify student identities and process examination papers efficiently.

## Project Members

- **Alana Fathima** (alanafathima66@gmail.com)
- **Amal Naveen** (amalnaveen001@gmail.com)
- **Fathima Z** (fathimazzz15@gmail.com)
- **Gowthami V R** (gowthamivr13@gmail.com)

## Features

- **Face Recognition**: Real-time face detection and identity verification using DeepFace
- **Signature Verification**: Digital signature verification using TensorFlow Lite model
- **OMR Processing**: Automated processing of Optical Mark Recognition sheets
- **Web Interface**: User-friendly Flask-based web application
- **Student Data Management**: Integrated student information system

## Technical Stack

- **Backend**: Python 3.x with Flask
- **Machine Learning**:
  - TensorFlow 2.9.0
  - DeepFace
  - OpenCV
  - scikit-learn
- **Frontend**: HTML, CSS, JavaScript
- **Image Processing**: OpenCV, PIL
- **Data Processing**: NumPy, Pandas

## Project Structure

```
├── app.py                 # Main application file
├── requirements.txt       # Project dependencies
├── models/               # Trained ML models
├── templates/            # HTML templates
├── static/              # Static assets
├── uploads/             # Temporary file storage
├── data/                # Data files
├── training/            # Training scripts
├── test_images/         # Test image samples
└── reports/             # Generated reports
```

## Prerequisites

- Python 3.8 or higher (recommended: Python 3.8-3.11)
- Virtual environment (recommended)
- CUDA-capable GPU (optional, for faster processing)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd identity_project/app
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Application

1. Start the Flask server:
```bash
python app.py
```

2. Open your web browser and navigate to:
```
http://localhost:5000
```

## Core Components

### Face Recognition
- Uses DeepFace library for face detection and recognition
- Supports multiple face detection backends
- Real-time face verification

### Signature Verification
- TensorFlow Lite model for signature verification
- Preprocessing pipeline for signature images
- Confidence-based verification system

### OMR Processing
- Automated bubble detection and marking
- Answer key management system
- Grade calculation and reporting

## API Endpoints

- `/`: Main application page
- `/answer_key`: Answer key management interface
- `/signature`: Signature verification interface
- `/verify_signature`: Signature verification endpoint
- `/process`: OMR processing endpoint

## Model Details

### Face Recognition Model
- Architecture: DeepFace (VGG-Face)
- Input: RGB face images
- Output: Identity prediction with confidence score

### Signature Verification Model
- Architecture: TensorFlow Lite model
- Input: Preprocessed signature images
- Output: Signature verification result with confidence score

## Performance Considerations

- GPU acceleration is disabled by default (configurable in app.py)
- Image size limits: 16MB per upload
- Supported image formats: JPG, PNG
- Recommended image resolution: 1920x1080 or higher

## Security Features

- File size restrictions
- Input validation
- Secure file handling
- Temporary file cleanup

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request
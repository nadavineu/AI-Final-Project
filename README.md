# AI-Final-Project
mood detector in python with pytorch
# Emotion Detection System

## Project Overview

This project is an emotion detection system that utilizes a convolutional neural network (CNN) model to analyze facial expressions from images. The system includes a frontend for user interaction and a backend for processing and prediction.

## Project Structure

```
C:.
├───Frontend
│   ├───models
│   │   └─── ModelFinal_detection_model.pth
│   ├───templates
│   │   └─── index.html (index.htm_)
│   └─── app.py
└───Model Training
```

## Components

- **Frontend**: Contains the web application files, including the model and templates.
  - `models/ModelFinal_detection_model.pth`: The trained deep learning model for emotion detection.
  - `templates/index.html`: The main HTML file for the web interface.
  - `app.py`: The Flask application to serve the frontend and handle predictions.
- **Model Training**: Contains scripts and data for training the emotion detection model.

## Requirements

- Python 3.x
- Flask
- PyTorch
- OpenCV
- NumPy
- Other dependencies listed in `requirements.txt`

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   ```
2. Navigate to the project directory:
   ```bash
   cd <project-directory>
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the application:
   ```bash
   python app.py
   ```

## Usage

- Open a web browser and navigate to `http://localhost:5000`.
- Upload an image containing a face.
- The model will process the image and display the detected emotion.

## License

This project is licensed under the terms specified in the `LICENSE` file.

## Author

Developed by Nadav.


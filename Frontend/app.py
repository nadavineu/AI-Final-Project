import torch
import torch.nn as nn
import torch.nn.functional as F
from flask import Flask, render_template, Response
import cv2
import numpy as np

# Define the CNNModel class with the exact same structure as when you saved it
class CNNModel(nn.Module):
    def __init__(self, num_classes=7):
        super(CNNModel, self).__init__()

        # Convolutional Block 1
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)  # padding=1 for "same"
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)  # padding=1 for "same"
        self.bn2 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(0.2)

        # Convolutional Block 2
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)  # padding=1 for "same"
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)  # padding=1 for "same"
        self.bn4 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout(0.2)

        # Convolutional Block 3
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)  # padding=1 for "same"
        self.bn5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)  # padding=1 for "same"
        self.bn6 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout3 = nn.Dropout(0.2)

        # Convolutional Block 4
        self.conv7 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)  # padding=1 for "same"
        self.bn7 = nn.BatchNorm2d(256)
        self.conv8 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)  # padding=1 for "same"
        self.bn8 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout4 = nn.Dropout(0.2)

        # Fully Connected Layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(256 * 3 * 3, 64)  # Adjust input size based on the final feature map size
        self.bn9 = nn.BatchNorm1d(64)
        self.dropout5 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(64, 64)
        self.bn10 = nn.BatchNorm1d(64)
        self.dropout6 = nn.Dropout(0.5)

        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x):
        # Block 1
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        x = self.dropout1(x)

        # Block 2
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)
        x = self.dropout2(x)

        # Block 3
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.pool3(x)
        x = self.dropout3(x)

        # Block 4
        x = F.relu(self.bn7(self.conv7(x)))
        x = F.relu(self.bn8(self.conv8(x)))
        x = self.pool4(x)
        x = self.dropout4(x)

        # Fully Connected Layers
        x = self.flatten(x)
        x = F.relu(self.bn9(self.fc1(x)))
        x = self.dropout5(x)

        x = F.relu(self.bn10(self.fc2(x)))
        x = self.dropout6(x)

        x = self.fc3(x)
        return F.softmax(x, dim=1)

# Initialize the model
model = CNNModel(num_classes=7)

# Load the model weights (use the correct path to the model weights file)
model_path = "C:/Program Files/EmotionCNN/models/ModelFinal_detection_model.pth"
model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))

# Set the model to evaluation mode
model.eval()

# Create Flask app
app = Flask(__name__)
import cv2
import numpy as np
import torch
import cv2
import numpy as np
import torch

# Define emotion labels
emotion_labels = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
import cv2
import numpy as np
import torch

# Define emotion labels
emotion_labels = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']

# Load OpenCV's pre-trained face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def generate_frames():
    cap = cv2.VideoCapture(0)  # Open webcam

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            # Extract the face ROI (region of interest)
            face = gray[y:y+h, x:x+w]

            # Resize face to (48x48) for model input
            face = cv2.resize(face, (48, 48))
            face = face / 255.0  # Normalize pixel values

            # Convert to tensor (1, 1, 48, 48)
            input_tensor = torch.tensor(face, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

            # Perform emotion detection
            with torch.no_grad():
                outputs = model(input_tensor)  # Get raw model output
                probabilities = torch.nn.functional.softmax(outputs, dim=1)  # Convert to probabilities
                predicted_idx = torch.argmax(probabilities, dim=1).item()  # Get predicted class index
                emotion = emotion_labels[predicted_idx]  # Convert index to label

            # Draw a rectangle around the face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # Display the predicted emotion
            cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Encode frame as JPEG
        _, jpeg = cv2.imencode('.jpg', frame)
        frame = jpeg.tobytes()

        # Yield the frame
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

    cap.release()




@app.route('/')
def index():
    return render_template('index.html')  # Make sure the 'index.html' file exists in the 'templates' folder

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)

# Automated Attendance System Using Face Recognition (MTCNN + FaceNet)

This project is an automated facial recognition-based attendance system implemented in a Jupyter Notebook. It utilizes MTCNN for face detection and FaceNet for embedding generation. A deep neural network is trained on these embeddings to classify individuals. Once identified, the system logs their attendance in a timestamped CSV file.

## Project Highlights

- Detects faces from images using MTCNN.
- Extracts 512-dimensional embeddings with FaceNet.
- Augments training data using `ImageDataGenerator`.
- Trains a neural network classifier on embeddings.
- Captures real-time face data via webcam (Google Colab compatible).
- Logs attendance in a CSV with timestamp and status ("present").

## Notebook
The full workflow is implemented in:

Automated_Attendance_System.ipynb


## Dataset Requirements

- **Image Folder:** A directory containing labeled face images (e.g., `/content/gdrive/MyDrive/images/`).
- **CSV File:** A CSV file with two columns:
  - `id`: Image filename (e.g., `person1.jpg`)
  - `label`: Identity of the person

Example CSV structure:

| id         | label     |
|------------|-----------|
| 001.jpg    | person_1  |
| 002.jpg    | person_2  |

## Setup Instructions

1. **Mount Google Drive in Colab:**

from google.colab import drive
drive.mount('/content/gdrive')
Install required packages in Colab:


!pip install mtcnn keras-facenet
Install additional dependencies:

pip install -r requirements.txt
Run all cells in Automated_Attendance_System.ipynb.

Output
Predicted label of the person in real-time webcam image.

Log file your_file.csv with the following columns:

Time
Date
Label
Mark

Example row:

2025-08-01 10:15:23, 2025-08-01, person_1, present


Technologies Used
Python
TensorFlow / Keras
MTCNN
FaceNet (keras-facenet)
OpenCV
Pandas, NumPy
Google Colab


License
This project is licensed under the MIT License.


Acknowledgments

FaceNet by Google

MTCNN Face Detector

keras-facenet

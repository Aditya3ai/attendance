import cv2
import numpy as np
import os

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to detect and crop face
def detect_and_crop_face(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
    if len(faces) > 0:
        x, y, w, h = faces[0]  # Use the first detected face
        return image[y:y+h, x:x+w]
    else:
        return None

# Load known faces and names from captured_images directory
dataset_dir = r"C:\Users\cex\Desktop\face_recognition_attendance\captured_images"
known_faces = []
known_names = []

for person_name in os.listdir(dataset_dir):
    person_dir = os.path.join(dataset_dir, person_name)
    if os.path.isdir(person_dir):  # Ensure it's a directory
        for image_file in os.listdir(person_dir):
            image_path = os.path.join(person_dir, image_file)
            image = cv2.imread(image_path)
            if image is not None:
                cropped_face = detect_and_crop_face(image)
                if cropped_face is not None:
                    resized_face = cv2.resize(cropped_face, (100, 100))
                    known_faces.append(resized_face)
                    known_names.append(person_name)
                else:
                    print(f"Face not detected in {image_path}")

if not known_faces:
    print("No known faces loaded. Exiting.")
    exit()

print(f"Loaded {len(known_faces)} faces from the dataset.")

# Start video capture from webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture video frame. Exiting.")
        break

    # Detect faces in the video frame
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detected_faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

    for (x, y, w, h) in detected_faces:
        detected_face = frame[y:y+h, x:x+w]
        resized_detected_face = cv2.resize(detected_face, (100, 100))

        # Compare with known faces using Mean Squared Error (MSE)
        min_mse = float('inf')
        recognized_name = "Unknown"
        for known_face, name in zip(known_faces, known_names):
            mse = np.mean((cv2.cvtColor(known_face, cv2.COLOR_BGR2GRAY) - 
                           cv2.cvtColor(resized_detected_face, cv2.COLOR_BGR2GRAY)) ** 2)
            if mse < min_mse and mse < 500:  # Adjust threshold as needed
                min_mse = mse
                recognized_name = name

        # Draw bounding box and label
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, recognized_name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the video frame with recognized faces
    cv2.imshow("Live Face Recognition", frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

import cv2
import os
import face_recognition
import csv
from datetime import datetime

# Display a loading message
print("Loading datasets...")

# Load known faces and names from the captured_images directory
dataset_dir = r"C:\Users\cex\Desktop\face_recognition_attendance\captured_images"
known_face_encodings = []
known_names = []

for person_name in os.listdir(dataset_dir):
    person_dir = os.path.join(dataset_dir, person_name)
    if os.path.isdir(person_dir):  # Ensure it's a directory
        for image_file in os.listdir(person_dir):
            image_path = os.path.join(person_dir, image_file)
            image = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(image)
            if encodings:  # Ensure the face was detected
                known_face_encodings.append(encodings[0])
                known_names.append(person_name)

print(f"Loaded {len(known_face_encodings)} faces from the dataset.")

# Create or open the CSV file to store attendance
csv_file = "attendance.csv"
if not os.path.exists(csv_file):
    with open(csv_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Name", "Date", "Time"])

# Start video capture from webcam
cap = cv2.VideoCapture(0)

# Track recognized faces to avoid duplicate entries within the same session
recognized_faces = set()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture video frame. Exiting.")
        break

    # Resize frame for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Detect faces and encode them
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for face_encoding, face_location in zip(face_encodings, face_locations):
        # Compare face encodings with known faces
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.5)
        name = "Unknown"

        if True in matches:
            match_index = matches.index(True)
            name = known_names[match_index]

            # Register the recognized face in CSV if not already registered during this session
            if name not in recognized_faces:
                recognized_faces.add(name)
                now = datetime.now()
                date = now.strftime("%Y-%m-%d")
                time = now.strftime("%H:%M:%S")

                with open(csv_file, mode="a", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow([name, date, time])

        # Scale back face location to original size
        top, right, bottom, left = face_location
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw bounding box and label
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the video frame with recognized faces
    cv2.imshow("Live Face Recognition", frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
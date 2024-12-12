import cv2
import os

# Load the pre-trained Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize the webcam
video_capture = cv2.VideoCapture(0)

# Prompt user to enter their name
name = input("Enter your name: ")

# Create a directory for the person to store captured images
if not os.path.exists('captured_images'):
    os.makedirs('captured_images')

person_directory = os.path.join('captured_images', name)
if not os.path.exists(person_directory):
    os.makedirs(person_directory)

print(f"Capturing images for {name}. Press 'q' to stop capturing.")

img_count = 0  # To count the number of images captured

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    if not ret:
        print("Failed to grab frame.")
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw rectangles around the faces and capture images
    for (x, y, w, h) in faces:
        # Draw rectangle around face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Crop the detected face
        face = frame[y:y+h, x:x+w]

        # Save the cropped face image with a unique filename
        img_count += 1
        img_filename = os.path.join(person_directory, f"{name}_{img_count}.jpg")
        cv2.imwrite(img_filename, face)

        # Show captured image number on the screen
        cv2.putText(frame, f"Capturing {img_count}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Face Capture', frame)

    # Stop capturing when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close any open windows
video_capture.release()
cv2.destroyAllWindows()

print(f"Images saved in 'captured_images/{name}/'")

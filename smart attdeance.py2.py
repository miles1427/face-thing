import cv2
import face_recognition
import numpy as np

# -----------------------
# Step 1: Load known faces
# -----------------------
known_image = face_recognition.load_image_file("bhavya 1.jpg")  # your saved image
known_encoding = face_recognition.face_encodings(known_image)[0]

known_face_encodings = [known_encoding]
known_face_names = ["Mandes"]  # name of the person

# -----------------------
# Step 2: Start phone camera via DroidCam
# -----------------------
video_capture = cv2.VideoCapture(4747)  # try 0 or 1, your phone should appear here
 # update URL with your feed

if not video_capture.isOpened():
    print("fuck you")
    exit()

# -----------------------
# Step 3: Real-time face recognition
# -----------------------
while True:
    ret, frame = video_capture.read()
    if not ret:
        print("Failed to grab frame! Check your connection.")
        break

    # Resize frame for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]  # BGR -> RGB

    # Detect faces and get encodings
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for face_encoding, face_location in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)

        name = "Unknown"
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        # Draw rectangle and name
        top, right, bottom, left = face_location
        top *= 4; right *= 4; bottom *= 4; left *= 4  # scale back up
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

    # Show the webcam feed
    cv2.imshow("Phone Camera Face Recognition", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# -----------------------
# Step 4: Release resources safely
# -----------------------
video_capture.release()
cv2.destroyAllWindows()

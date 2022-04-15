"""
This script makes used of face_recognition package to calculate the 128D descriptor to be used for face recognition
and compare the faces using some distance metrics
"""

# Import required packages:
import face_recognition

# Load known images (remember that these images are loaded in RGB order):
known_image_1 = face_recognition.load_image_file("jared_1.jpeg")
known_image_2 = face_recognition.load_image_file("jared_2.jpeg")
known_image_3 = face_recognition.load_image_file("jared_3.jpeg")
known_image_4 = face_recognition.load_image_file("obama.jpeg")

# Crate names for each loaded image:
names = ["jared_1.jpeg", "jared_2.jpeg", "jared_3.jpeg", "obama.jpeg"]

# Load unknown image (this image is going to be compared against all the previous loaded images):
unknown_image = face_recognition.load_image_file("jared_4.jpeg")

# Calculate the encodings for every of the images:
known_image_1_encoding = face_recognition.face_encodings(known_image_1)[0]
known_image_2_encoding = face_recognition.face_encodings(known_image_2)[0]
known_image_3_encoding = face_recognition.face_encodings(known_image_3)[0]
known_image_4_encoding = face_recognition.face_encodings(known_image_4)[0]
known_encodings = [known_image_1_encoding, known_image_2_encoding, known_image_3_encoding, known_image_4_encoding]
unknown_encoding = face_recognition.face_encodings(unknown_image)[0]

# Compare the faces:
results = face_recognition.compare_faces(known_encodings, unknown_encoding)

# Print the results:
print(results)
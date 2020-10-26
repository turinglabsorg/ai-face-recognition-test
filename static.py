#%%
import numpy as np
import face_recognition
from PIL import Image, ImageDraw
from IPython.display import display

# The program we will be finding faces on the example below
pil_im = Image.open('compare/three_people.jpg')
display(pil_im)

# Create arrays of known face encodings and their names
known_face_encodings = [
    face_recognition.face_encodings(face_recognition.load_image_file("learn/biden.jpg"))[0],
    face_recognition.face_encodings(face_recognition.load_image_file("learn/obama.jpg"))[0],
    face_recognition.face_encodings(face_recognition.load_image_file("learn/roberto.jpg"))[0]
]

known_face_names = [
    "The good",
    "The ugly",
    "The bad"
]

print('Learned encoding for', len(known_face_encodings), 'images.')

# Load an image with an unknown face
unknown_image = face_recognition.load_image_file("compare/three_people.jpg")

# Find all the faces and face encodings in the unknown image
face_locations = face_recognition.face_locations(unknown_image)
face_encodings = face_recognition.face_encodings(unknown_image, face_locations)

# Convert the image to a PIL-format image so that we can draw on top of it with the Pillow library
# See http://pillow.readthedocs.io/ for more about PIL/Pillow
pil_image = Image.fromarray(unknown_image)
# Create a Pillow ImageDraw Draw instance to draw with
draw = ImageDraw.Draw(pil_image)

# Loop through each face found in the unknown image
for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
    # See if the face is a match for the known face(s)
    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
    name = "Unknown"

    # Or instead, use the known face with the smallest distance to the new face
    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
    best_match_index = np.argmin(face_distances)
    if matches[best_match_index]:
        name = known_face_names[best_match_index]
        
    # Draw a box around the face using the Pillow module
    if best_match_index == 2:
        draw.rectangle(((left, top), (right, bottom)), outline=(255, 0, 0))
        # Draw a label with a name below the face
        text_width, text_height = draw.textsize(name)
        draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(255, 0, 0), outline=(255, 0, 0))
        draw.text((left + 6, bottom - text_height - 5), name, fill=(255, 255, 255, 255))
    else:
        draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))
        text_width, text_height = draw.textsize(name)
        draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
        draw.text((left + 6, bottom - text_height - 5), name, fill=(255, 255, 255, 255))

# Remove the drawing library from memory as per the Pillow docs
del draw

# Display the resulting image
display(pil_image)
import face_recognition
import os
import pickle
import numpy as np
from PIL import Image

dataset_path = "dataset"

encodings = []
names = []

for filename in os.listdir(dataset_path):
    path = os.path.join(dataset_path, filename)

    try:
        # Convert image to RGB properly
        pil_image = Image.open(path).convert("RGB")
        image = np.array(pil_image)

        face_enc = face_recognition.face_encodings(image)

        if len(face_enc) > 0:
            encodings.append(face_enc[0])
            names.append(filename.split(".")[0])
        else:
            print(f"No face found in {filename}")

    except Exception as e:
        print(f"Error processing {filename}: {e}")

data = {
    "encodings": encodings,
    "names": names
}

with open("encodings.pkl", "wb") as f:
    pickle.dump(data, f)

print(image.shape)
print("Encodings saved successfully!")
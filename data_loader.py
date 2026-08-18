import os
import cv2
import numpy as np

def load_images(df, base_path):
    images = []
    labels = []

    for i in range(len(df)):
        row = df.iloc[i]

        img_path = os.path.normpath(
            os.path.join(base_path, row["Path"])
        )

        if i < 5:
            print("Looking for:", os.path.abspath(img_path))

        if not os.path.exists(img_path):
            print("FILE NOT FOUND:", img_path)
            continue

        img = cv2.imread(img_path)

        if img is None:
            print("IMAGE CORRUPTED:", img_path)
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (32, 32))

        images.append(img)
        labels.append(row["ClassId"])

    print(f"Loaded {len(images)} images")
    print(f"Skipped {len(df) - len(images)} images")

    return np.array(images), np.array(labels)
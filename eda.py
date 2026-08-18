import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

base_path = "German_Traffic_sign"
def eda(train_data, base_path):
    """
    Perform Exploratory Data Analysis on Traffic Sign Dataset.
    """

    print("=" * 50)
    print("DATASET INFORMATION")
    print("=" * 50)
    train_data.info()

    print("\nSummary Statistics")
    print(train_data.describe())

    print("\nMissing Values")
    print(train_data.isnull().sum())

    print("\nClass Distribution")
    print(train_data["ClassId"].value_counts().sort_index())

    # ---------------- Class Distribution ---------------- #

    plt.figure(figsize=(14, 6))
    sns.countplot(x="ClassId", data=train_data)
    plt.title("Traffic Sign Class Distribution")
    plt.xlabel("Class ID")
    plt.ylabel("Number of Images")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig("Sample_images.png",dpi = 300)
    plt.show()
    #---------------- Visualize --------------------- #
    plt.figure(figsize=(10, 6))
    sample = train_data.sample(9)  # Randomly sample 9 images
    for i,(_,row) in enumerate(sample.iterrows()):
            img_path = os.path.join(base_path, row['Path'])
            img = cv2.imread(img_path)
    
            if img is None:
                print(f"Warning: Could not read image at {img_path}")
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for correct color display
            plt.subplot(3, 3, i+ 1)
            plt.imshow(img)
            plt.title(f"Class ID: {row['ClassId']}")
            plt.axis('off')
    
            #print(f"Image {i+1} - Class ID: {row['ClassId']} - Path: {img_path}")
    plt.tight_layout()        
    plt.show() 
    # ---------------- Sample Images ---------------- #

    plt.figure(figsize=(10, 10))

    sample = train_data.sample(
        n=min(9, len(train_data)),
        random_state=42
    )

    for i, (_, row) in enumerate(sample.iterrows()):

        img_path = os.path.normpath(
            os.path.join(base_path, row["Path"])
        )

        if not os.path.exists(img_path):
            print("File not found:", img_path)
            continue

        img = cv2.imread(img_path)

        if img is None:
            print("Cannot read:", img_path)
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        plt.subplot(3, 3, i + 1)
        plt.imshow(img)
        plt.title(f"Class {row['ClassId']}")
        plt.axis("off")

    plt.tight_layout()
    plt.savefig("Sample_images.png",dpi = 300)
    plt.show()


# ----------------------------------------------------- #

def check_images(train_data, base_path):
    """
    Verify all image paths.
    """

    print("\nChecking Images...\n")

    success = 0
    failed = 0

    for _, row in train_data.iterrows():

        img_path = os.path.normpath(
            os.path.join(base_path, row["Path"])
        )

        if not os.path.exists(img_path):
            print("Missing:", img_path)
            failed += 1
            continue

        img = cv2.imread(img_path)

        if img is None:
            print("Corrupted:", img_path)
            failed += 1
            continue

        success += 1

    print("\nImage Check Completed")
    print(f"Successfully Loaded : {success}")
    print(f"Missing/Corrupted   : {failed}")


# ----------------------------------------------------- #
if __name__ == "__main__":
   #load data
   train_df = pd.read_csv("German_Traffic_sign/Train.csv")

   eda(train_df,base_path)
   check_images(train_df, base_path)
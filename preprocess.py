import os
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from config import FILE_COL, IMG_DIR, RESIZED_DIR, TRAIN_CSV, TEST_CSV, IMG_SIZE


def resize_dataset():
    os.makedirs(RESIZED_DIR, exist_ok=True)

    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)
    all_ids = pd.concat([train_df[[FILE_COL]], test_df[[FILE_COL]]], ignore_index=True)

    success = skipped = failed = 0

    for _, row in all_ids.iterrows():
        filename = f"face-{int(row[FILE_COL])}.png"
        src_path = os.path.join(IMG_DIR, filename)
        dst_path = os.path.join(RESIZED_DIR, filename)

        if os.path.exists(dst_path):
            skipped += 1
            continue

        if not os.path.exists(src_path):
            print(f"Missing: {src_path}")
            failed += 1
            continue

        img = cv2.imread(src_path)
        if img is None:
            print(f"Cannot read: {src_path}")
            failed += 1
            continue

        resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
        cv2.imwrite(dst_path, resized)
        success += 1

    print(f"Done. Resized: {success} | Skipped: {skipped} | Failed: {failed}")


def verify_resize():
    df = pd.read_csv(TRAIN_CSV)
    sample = df.sample(9, random_state=0)

    fig, axes = plt.subplots(3, 3, figsize=(7, 7))

    for ax, (_, row) in zip(axes.flat, sample.iterrows()):
        filename = f"face-{int(row[FILE_COL])}.png"
        img_path = os.path.join(RESIZED_DIR, filename)
        img = cv2.imread(img_path)

        if img is None:
            ax.axis("off")
            continue

        h, w = img.shape[:2]
        if h != IMG_SIZE or w != IMG_SIZE:
            print(f"Wrong size {h}x{w}: {filename}")

        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(f"{w}x{h}", fontsize=8)
        ax.axis("off")

    plt.suptitle("Resize Verification", fontsize=10)
    plt.tight_layout()
    plt.savefig("resize_verification.png", dpi=100)
    plt.show()
    print("Verification grid saved to resize_verification.png")


resize_dataset()
verify_resize()
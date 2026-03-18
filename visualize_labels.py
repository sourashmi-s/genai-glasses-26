import os
import cv2
import pandas as pd
from config import FILE_COL, LABEL_COL, IMG_DIR, TRAIN_CSV, TEST_CSV


DISPLAY_SIZE = 512
DELAY_MS     = 3000


def verify_labels(csv_path):

    df = pd.read_csv(csv_path)
    flagged = []

    print(f"Viewing {len(df)} images from {csv_path}")
    print("f = flag mislabelled | q = quit | any key = next")

    for idx, row in df.iterrows():

        filename = f"face-{int(row[FILE_COL])}.png"
        img_path = os.path.join(IMG_DIR, filename)

        if not os.path.exists(img_path):
            print(f"Missing: {img_path}")
            continue

        img = cv2.imread(img_path)
        if img is None:
            print(f"Cannot read: {img_path}")
            continue

        img = cv2.resize(img, (DISPLAY_SIZE, DISPLAY_SIZE), interpolation=cv2.INTER_AREA)

        label_text = "GLASSES" if int(row[LABEL_COL]) == 1 else "NO GLASSES"
        cv2.rectangle(img, (0, 0), (img.shape[1], 30), (0, 0, 0), -1)
        cv2.putText(img, f"{label_text} | id={row[FILE_COL]}", (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)

        cv2.imshow("Label Check", img)
        key = cv2.waitKey(DELAY_MS) & 0xFF

        if key == ord("q"):
            break
        elif key == ord("f"):
            flagged.append({"index": idx, "id": row[FILE_COL], "label": row[LABEL_COL]})
            print(f"Flagged: id={row[FILE_COL]} label={row[LABEL_COL]}")

    cv2.destroyAllWindows()
    return flagged


def main():

    all_flagged = []

    for csv_path in [TRAIN_CSV, TEST_CSV]:
        flagged = verify_labels(csv_path)
        all_flagged.extend(flagged)

    if all_flagged:
        print(f"\n{len(all_flagged)} images flagged:")
        for item in all_flagged:
            print(f"  id={item['id']}  current_label={item['label']}")
        print("Open the CSV and fix these rows manually.")
    else:
        print("No issues found.")


main()
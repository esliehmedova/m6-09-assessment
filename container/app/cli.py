import sys, os, csv, json
from detector import CatDetector

APP_DIR = os.path.dirname(os.path.abspath(__file__))
STUDENT_JSON = "/app/STUDENT.json"
MODEL_PATH = os.path.join(APP_DIR, "..", "models", "best.onnx")
INPUT_DIR = "/data/input"
OUTPUT_CSV = "/data/output/predictions.csv"
EXTS = (".jpg", ".jpeg", ".png")

def cmd_info():
    with open(STUDENT_JSON) as f:
        print(f.read())

def cmd_predict():
    det = CatDetector(os.path.abspath(MODEL_PATH))
    rows = []
    for root, _, files in os.walk(INPUT_DIR):
        for fn in sorted(files):
            if not fn.lower().endswith(EXTS):
                continue
            full = os.path.join(root, fn)
            rel = os.path.relpath(full, INPUT_DIR).replace(os.sep, "/")
            boxes = det.predict(full)
            if not boxes:
                rows.append([rel, "", "", "", "", "", ""])
            for b in boxes:
                rows.append([rel, b["xmin"], b["ymin"], b["xmax"],
                             b["ymax"], b["confidence"], b["class"]])
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_path","xmin","ymin","xmax","ymax","confidence","class"])
        w.writerows(rows)
    print(f"Wrote {len(rows)} rows to {OUTPUT_CSV}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: cli.py [info|predict]"); sys.exit(1)
    if sys.argv[1] == "info":
        cmd_info()
    elif sys.argv[1] == "predict":
        cmd_predict()
    else:
        print(f"unknown command: {sys.argv[1]}"); sys.exit(1)

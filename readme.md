an AI image detection system that runs completely offline,
Goal Definition

You want an AI model that can:

Detect people (workers, visitors) in images or video.

Determine whether each person is wearing a safety helmet or not wearing one.

Run locally without internet.

This means:
✅ You need object detection, not just classification.
✅ You need two labels: helmet and no_helmet.
✅ You can fine-tune a lightweight YOLO model offline.

⚙️ 2. Model Choice (Offline, Free, High Performance)
Model	Offline	Speed	Accuracy	Notes
YOLOv8n / YOLOv5s	✅	⚡⚡⚡⚡	🔥🔥🔥	Best balance for CPU/GPU
YOLOv8n-seg	✅	⚡⚡⚡	🔥🔥	Detect + segment helmets
MobileNet-SSD	✅	⚡⚡⚡⚡	🔥	Ultra-light for embedded

💡 Recommended: Start with YOLOv8n.pt → fine-tune on helmet dataset.

📦 3. Dataset Options

You can use open datasets:

Safety Helmet Detection Dataset

~3,800 images

Labels: person, helmet, head, helmet-on, helmet-off

PPE Detection Dataset (Roboflow)

Variants for helmet, vest, mask, boots

Or capture your own site’s CCTV frames and label them with LabelImg or Roboflow Annotate.

Label format: YOLO text files (class_id x_center y_center width height).

🧰 4. Training (Offline)

Install dependencies:

pip install ultralytics


Organize dataset:

datasets/
 ├── images/
 │   ├── train/
 │   └── val/
 └── labels/
     ├── train/
     └── val/


Example data.yaml:

train: datasets/images/train
val: datasets/images/val

nc: 2
names: ['helmet', 'no_helmet']


Train locally:

yolo detect train data=data.yaml model=yolov8n.pt epochs=50 imgsz=640


✅ This will create a local model file like:

runs/detect/train/weights/best.pt

🧩 5. Local Inference (No Internet)
from ultralytics import YOLO
import cv2

model = YOLO("runs/detect/train/weights/best.pt")

img = cv2.imread("worker.jpg")
results = model(img)
annotated = results[0].plot()
cv2.imshow("Helmet Detection", annotated)
cv2.waitKey(0)


Runs fully offline using local weights.
For live camera:

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    results = model(frame)
    cv2.imshow("Detection", results[0].plot())
    if cv2.waitKey(1) == 27:  # ESC to exit
        break
cap.release()
cv2.destroyAllWindows()

💻 6. Optional: Host Locally (FastAPI REST Server)

To expose detection via API (still offline):

from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
from PIL import Image
import io

app = FastAPI()
model = YOLO("runs/detect/train/weights/best.pt")

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    img_bytes = await file.read()
    img = Image.open(io.BytesIO(img_bytes))
    results = model(img)
    return results[0].tojson()


Run with:

uvicorn app:app --reload


Then send POST /detect with an image file.

⚡ 7. Optimization Tips
Goal	How
Run on low-end PC	Use yolov8n.pt or quantize to INT8
Run on Jetson	Export to TensorRT: yolo export format=engine
Run on Raspberry Pi	Export to ONNX or TFLite
Improve accuracy	Add more no helmet images and diverse lighting angles
🧾 8. Output Example

Each detected person:

[
  {
    "class": "helmet",
    "confidence": 0.92,
    "bbox": [100, 120, 80, 90]
  },
  {
    "class": "no_helmet",
    "confidence": 0.88,
    "bbox": [300, 150, 85, 95]
  }
]


You can overlay a red box for “no helmet” and green for “helmet”.

✅ Summary
Step	Description
1	Use YOLOv8n (offline)
2	Train on Kaggle/Roboflow PPE dataset
3	Fine-tune locally with ultralytics
4	Detect via Python or FastAPI
5	Deploy offline on PC / Jetson / Pi
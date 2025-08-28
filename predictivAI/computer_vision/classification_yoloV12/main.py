import io
import requests
from PIL import Image
from ultralytics import YOLO

model = YOLO("yolo12n.pt")

url = "https://media.roboflow.com/notebooks/examples/dog-2.jpeg"

image = Image.open(io.BytesIO(requests.get(url).content))
predictions = model([image])
for prediction in predictions:
    prediction.show()
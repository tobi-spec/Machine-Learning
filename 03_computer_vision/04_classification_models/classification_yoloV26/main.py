import io
import requests
from PIL import Image
from ultralytics import YOLO

model = YOLO("yolo26n-cls.pt")

url = "https://ultralytics.com/images/bus.jpg"

image = Image.open(io.BytesIO(requests.get(url).content))
predictions = model([image])
for prediction in predictions:
    prediction.show()
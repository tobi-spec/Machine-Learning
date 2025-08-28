import io
import requests
import supervision as sv
from PIL import Image
from rfdetr import RFDETRBase

# Apache 2.0 license
model = RFDETRBase()
model.optimize_for_inference()

url = "https://media.roboflow.com/notebooks/examples/dog-2.jpeg"

image = Image.open(io.BytesIO(requests.get(url).content))
detections = model.predict(image)
labels = [model.class_names[i] for i in detections.class_id]

annotated_image = image.copy()
annotated_image = sv.BoxAnnotator().annotate(annotated_image, detections)
annotated_image = sv.LabelAnnotator().annotate(annotated_image, detections, labels)

sv.plot_image(annotated_image)

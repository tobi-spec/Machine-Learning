import io
import requests
from PIL import Image
import supervision as sv
from rfdetr import RFDETRKeypointPreview

url = "https://media.roboflow.com/notebooks/examples/dog-2.jpeg"

image = Image.open(io.BytesIO(requests.get(url).content))
model = RFDETRKeypointPreview()
key_points = model.predict(image, threshold=0.5)
annotated_image = sv.VertexAnnotator().annotate(image, key_points)

sv.plot_image(annotated_image)
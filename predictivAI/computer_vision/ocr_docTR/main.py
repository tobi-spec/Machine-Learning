from doctr.models import ocr_predictor
from doctr.io import DocumentFile

model = ocr_predictor(pretrained=True)
single_img_doc = DocumentFile.from_images("./img.png")

result = model(single_img_doc)
print(result)
result.show()
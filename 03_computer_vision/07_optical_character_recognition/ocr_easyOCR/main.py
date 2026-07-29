import easyocr
reader = easyocr.Reader(['en'])
result = reader.readtext('./img.png')
result_text = reader.readtext('./img.png', detail = 0)
print(result)
print(result_text)
from pathlib import Path

import matplotlib.pyplot as plt
from ultralytics import YOLO


IMAGE_SOURCE = "https://ultralytics.com/images/bus.jpg"

def create_results(model_name, output_name):
    model = YOLO(model_name)
    results = model(IMAGE_SOURCE)
    result_images = []

    for index, result in enumerate(results, start=1):
        title = f"{output_name}"
        result_images.append((title, result.plot()))

    return result_images


def show_comparison(result_images):
    columns = 3
    rows = 2
    fig, axes = plt.subplots(rows, columns, figsize=(15, 5 * rows))
    axes = axes.flatten()

    for ax, (title, image) in zip(axes, result_images):
        ax.imshow(image[..., ::-1]) # BGR -> RGB
        ax.set_title(title)
        ax.axis("off")

    for ax in axes[len(result_images):]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()


result_images = []
result_images.extend(create_results("yolo26n-cls.pt", "classification"))
result_images.extend(create_results("yolo26n.pt", "detection"))
result_images.extend(create_results("yolo26n-seg.pt", "segmentation"))
result_images.extend(create_results("yolo26n-pose.pt", "pose_estimation"))

show_comparison(result_images)

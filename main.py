import streamlit as st
import os
import cv2
import shutil
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image, ImageOps

PATH_MODEL = 'models/best.pt'
PATH_IMAGES = 'images'


def predict(image_path):
    return model(image_path).xywhn[0]


def get_labels(xywhn):
    return [str(model.names[int(index)]) for index in xywhn[:, 5].tolist()]


def get_bounding_boxes_coords(xywhn, img_height, img_width):
    boxes = xywhn[:, :4].tolist()

    pixel_bounding_boxes = [
        [x_center * img_width, y_center * img_height, width * img_width, height * img_height]
        for [x_center, y_center, width, height] in boxes
    ]

    boxes = []
    for bb in pixel_bounding_boxes:
        start = int(bb[0] - bb[2] / 2), int(bb[1] - bb[3] / 2)
        end = int(bb[0] + bb[2] / 2), int(bb[1] + bb[3] / 2)
        boxes.append([start, end])
    return boxes


def display_image(img):
    height, width = img.shape[:2]
    resized_image = cv2.resize(img, (3 * width, 3 * height), interpolation=cv2.INTER_CUBIC)

    fig = plt.gcf()
    fig.set_size_inches(18, 10)
    plt.axis("off")
    plt.imshow(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))


def cleanup():
    try:
        shutil.rmtree(PATH_IMAGES)
    except OSError as e:
        print(f'Error: {PATH_IMAGES} : {e.strerror}')


def setup_model():
    if not Path(PATH_IMAGES).exists():
        Path(PATH_IMAGES).mkdir(parents=True, exist_ok=True)

    return torch.hub.load('yolov5', 'custom', path=PATH_MODEL, force_reload=True, source='local')


model = setup_model()


if __name__ == '__main__':

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image = ImageOps.exif_transpose(image)
        file_path = os.path.join(PATH_IMAGES, uploaded_file.name)
        image.save(file_path)
        results = predict(file_path)

        im = cv2.imread(file_path)

        img_height, img_width = im.shape[0], im.shape[1]

        bounding_boxes = get_bounding_boxes_coords(results, im.shape[0], im.shape[1])
        category_names = get_labels(results)

        category_names, bounding_boxes = zip(*sorted(zip(category_names, bounding_boxes), key=lambda x: x[1]))
        for box, category in zip(bounding_boxes, category_names):
            start_point, end_point = box[0], box[1]
            cv2.putText(im, category, start_point, cv2.FONT_HERSHEY_SIMPLEX, 10, (255, 0, 255), 1)
            cv2.rectangle(im, start_point, end_point, (255, 255, 255), 2)

        message = "No numbers found!"
        if bounding_boxes:
            display_image(im)
            st.image(im)
            message = f"There are the following numbers on the picture: {', '.join(category_names)}"
        st.write(message)
    cleanup()

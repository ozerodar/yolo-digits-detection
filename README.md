# yolo-digits-detection
This project uses the You Only Look Once (YOLO) object detection algorithm to implement a digit recognition system. The system is able to detect and classify handwritten digits written on white paper.

## Dataset
The model was trained on the modified MNIST dataset with added digit bounding boxes (see https://universe.roboflow.com/min-david-gglzp/mnist-ihqky). It was also fine-tuned using the Caniverse numbers dataset (see https://universe.roboflow.com/caniverse/caniverse-numb) for better bounding boxes detection.

## Results
The model was tested on the modified MNIST testing dataset, achieving a mean average precision (mAP) of 74%. In real-time object detection scenarios, the system is able to detect and classify digits with relatively high precision. See the section below to check some of the scenarios.

## Usage
To use the digit recognition system first, you need to install some dependencies. For this, you need Python 3.10 and pipenv installed. Once you have this installed, run:

```
git clone https://github.com/ozerodar/yolo-digits-detection
cd yolo-digits-detection
pipenv shell --python 3.10
pipenv install --dev
git clone https://github.com/ultralytics/yolov5
pip install -r yolov5/requirements.txt 
```

Then simply run:

```
streamlit run main.py
```

You can find some sample images you can play with in the `demo` folder. Here you can notice that the system is sometimes unsure about a digit. This case can be solved by checking overlaps and selecting the class/box that has a higher probability.

## Limitations
The provided solution was implemented under time pressure and using a simple CPU. It requires more thorough dataset preparation for more general use (for example, using different backgrounds, hand writings, colors, etc.)

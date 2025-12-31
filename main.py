import cv2
import json
import numpy as np
import time
from openvino.inference_engine import IECore
from typing import Tuple

def predict(image_path: str) -> Tuple[int, float]:
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    img = img.transpose(2, 0, 1)
    img = img[np.newaxis, :] / 255.0

    input_blob = next(iter(net.input_info))
    output_blob = next(iter(net.outputs))

    start_time = time.time()
    res = exec_net.infer({input_blob: img})
    end_time = time.time()

    pred = res[output_blob]
    time_used = end_time - start_time

    return np.argmax(pred), time_used * 1000

if __name__ == '__main__':
    with open("./data/label_mapping.json", "r", encoding="utf-8") as f:
        labels = json.load(f)
    
    parsed = []

    with open("./data/valid/valid.txt", "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if not line:
                continue

            parts = line.split(maxsplit=2)
            id_ = int(parts[0])
            number = parts[1]

            choices = [s.strip() for s in parts[2].split(',')]
            parsed.append([id_, number, choices])

    ie = IECore()
    net = ie.read_network(
        model="./assets/model/mobilenet_v2.xml",
        weights="./assets/model/mobilenet_v2.bin"
    )
    exec_net = ie.load_network(
        network=net,
        device_name="MYRIAD"
    )

    acc = 0
    for i in range(1, 101):
        image_path = f"./data/valid/valid_{i}.jpeg"
        class_id, time_used = predict(image_path)
        print(f"Image {i}.jpeg is classified as: {labels[class_id]} (Inference time: {time_used:.3f} ms)")
        
        for choice in parsed[i-1][2]:
            if labels[class_id] == choice:
                print("  -> Correct!")
                acc += 1
                break
    print(f"Validation Accuracy: {acc}/100 = {acc/100:.2%}")

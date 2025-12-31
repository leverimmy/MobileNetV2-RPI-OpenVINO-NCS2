import cv2
import numpy as np
import time
from openvino.inference_engine import IECore
import json

# -------------------
# Initialize OpenVINO
# -------------------
ie = IECore()
net = ie.read_network(
    model="./assets/model/mobilenet_v2.xml",
    weights="./assets/model/mobilenet_v2.bin"
)
exec_net = ie.load_network(
    network=net,
    device_name="MYRIAD"
)

input_blob = next(iter(net.input_info))
output_blob = next(iter(net.outputs))

# Load ImageNet labels
with open("./data/label_mapping.json", "r", encoding="utf-8") as f:
    labels = json.load(f)

# -------------------
# Initialize Camera
# -------------------
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

print("=========================")
print("== Press ESC to Escape ==")
print("=========================")

# -------------------
# Optimization Variables
# -------------------
last_time = time.time()
frame_idx = 0
top3_results = []
inf_time = 0.0
DETECT_EVERY_N_FRAMES = 20

while True:
    ret, frame = cap.read()
    if not ret:
        print("Frame read error")
        time.sleep(0.5)
        continue

    frame_idx += 1
    curr_time = time.time()
    fps = 1 / (curr_time - last_time)
    last_time = curr_time

    # -------------------
    # Skip frames for performance
    # -------------------
    if frame_idx % DETECT_EVERY_N_FRAMES == 0:
        # -------------------
        # Preprocess Input
        # -------------------
        img = cv2.resize(frame, (224, 224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))[np.newaxis, :]
        img = np.ascontiguousarray(img)

        # -------------------
        # Asynchronous Inference
        # -------------------
        start_inf = time.time()
        request_id = 0
        exec_net.start_async(request_id=request_id, inputs={input_blob: img})
        exec_net.requests[request_id].wait(-1)
        
        inf_time = (time.time() - start_inf) * 1000 # ms
        
        # Get output
        pred = exec_net.requests[request_id].output_blobs[output_blob].buffer.flatten()
        
        # 1. Softmax calculation (convert logits to probabilities)
        # Subtract max value for numerical stability
        exp_scores = np.exp(pred - np.max(pred))
        probs = exp_scores / np.sum(exp_scores)

        # 2. Get Top-3 indices
        # argsort returns ascending order, [::-1] reverses to descending, take top 3
        top3_indices = np.argsort(probs)[::-1][:3]

        # 3. Format results
        top3_results = []
        for idx in top3_indices:
            prob = probs[idx] * 100
            label = labels[idx]
            top3_results.append((label, prob))

    # -------------------
    # Display Results (Top 3 + FPS)
    # -------------------
    for i, (label, prob) in enumerate(top3_results):
        text = f"{i+1}. {label}: {prob:.1f}%"
        cv2.putText(frame, text, (10, 30 + i*25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    height, width = frame.shape[:2]
    cv2.putText(frame, f"FPS: {fps:.1f} Inf: {inf_time:.1f}ms", (10, height - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    cv2.imshow("Live Classification - Top 3", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()

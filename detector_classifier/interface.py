import cv2
import torch
import torch.nn as nn
from torchvision import transforms, models
from ultralytics import YOLO

# YOLO Detector
# Choose trained model run folder
training_run_path = "models/ship_detector/YOLOv9/runs/detect/train2"

# Load trained model
detector_model = YOLO(f"{training_run_path}/weights/best.pt")

# EfficientNet classifier
save_path = "models/chip_classifier/sar_ship_classifier_reduced.pt"
num_classes = 6

# 1. Recreate the model architecture
classifier_model = models.efficientnet_b0(weights="DEFAULT")
classifier_model.classifier[1] = nn.Linear(classifier_model.classifier[1].in_features, num_classes)

# 2. Load weights
classifier_model.load_state_dict(torch.load(save_path, map_location="cpu"))

# 3. Put model into eval mode
classifier_model.eval()

# Extract chips and classify

# Class names (must have same order as during EfficientNet training)
CLASS_NAMES = ["Cargo", "Tanker", "Fishing", "Dredging", "Tug", "Other"]

# Transforms used during training of EfficientNet
chip_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

def extract_chip(img_rgb, bbox, padding=20):
    """
    Returns RGB chip with padding applied.
    bbox = (x1, y1, x2, y2)
    """
    h, w, _ = img_rgb.shape
    x1, y1, x2, y2 = bbox

    # Add padding while staying inside the image
    x1p = max(0, x1 - padding)
    y1p = max(0, y1 - padding)
    x2p = min(w, x2 + padding)
    y2p = min(h, y2 + padding)

    chip = img_rgb[y1p:y2p, x1p:x2p]
    return chip


def classify_chip(model, chip):
    """
    chip = numpy array HxWxC (RGB)
    returns: (pred_id, pred_name, pred_conf)
    """
    x = chip_transform(chip).unsqueeze(0)  # shape: 1x3x224x224

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]

    pred_id = int(torch.argmax(probs).item())
    pred_name = CLASS_NAMES[pred_id]
    pred_conf = float(probs[pred_id].item())

    return pred_id, pred_name, pred_conf


# YOLO detect ==> extract chips ==> EfficientNet classify ==> visualize
def detect_and_classify(
    image_path,
    detector,
    classifier,
    conf=0.25,
    padding=20
):
    """
    Detect and classify ships in an image.

    Returns
    -------
    detections : list of dict
        Each dictionary contains information about one detected ship.
    """

    img_bgr = cv2.imread(image_path)

    if img_bgr is None:
        raise ValueError(f"Could not read image: {image_path}")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # YOLO inference
    results = detector.predict(
        source=image_path,
        conf=conf,
        save=False,
        show=False
    )

    detections = []

    if not results or results[0].boxes is None:
        return detections

    for det_id, box in enumerate(results[0].boxes):

        # Bounding box
        x1, y1, x2, y2 = [
            int(v)
            for v in box.xyxy[0]
        ]

        # Detector confidence
        det_conf = float(box.conf[0].item())

        # Extract chip
        chip = extract_chip(
            img_rgb,
            (x1, y1, x2, y2),
            padding
        )

        if chip.size == 0:
            continue

        # Classification
        pred_id, pred_name, clf_conf = classify_chip(
            classifier,
            chip
        )

        detection = {
            "id": det_id,

            "bbox": {
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2
            },

            "detector_confidence": det_conf,

            "classification": {
                "class_id": pred_id,
                "class_name": pred_name,
                "confidence": clf_conf
            }
        }

        detections.append(detection)

    return detections

def draw_detections(img_rgb, detections):
    """
    Draw bounding boxes and labels on an RGB image.
    """

    annotated = img_rgb.copy()

    for det in detections:

        bbox = det["bbox"]

        x1 = bbox["x1"]
        y1 = bbox["y1"]
        x2 = bbox["x2"]
        y2 = bbox["y2"]

        ship_type = det["classification"]["class_name"]

        det_id = det["id"]

        cv2.rectangle(
            annotated,
            (x1, y1),
            (x2, y2),
            (255, 0, 0),
            2
        )

        label = f"{det_id} | {ship_type}"

        cv2.putText(
            annotated,
            label,
            (x1, max(y1 - 10, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 0),
            2
        )

    return annotated

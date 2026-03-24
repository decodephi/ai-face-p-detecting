from __future__ import annotations

import cv2

from app.core.config import settings
from app.detectors.dark_spots import detect_dark_spots
from app.detectors.imperfections import detect_imperfections
from app.detectors.oiliness import detect_oiliness
from app.pipeline.face_detection import detect_face
from app.pipeline.preprocessing import preprocess_image
from app.pipeline.skin_segmentation import segment_skin
from app.visualization.annotator import annotate_face


def run_analysis_pipeline(image_rgb):
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    preprocessed = preprocess_image(image_bgr)
    face = detect_face(preprocessed.resized_bgr)
    skin = segment_skin(face.face_bgr)
    face_rgb = cv2.cvtColor(face.face_bgr, cv2.COLOR_BGR2RGB)

    detections = detect_imperfections(face_rgb, skin.refined_mask, settings.max_detections)
    dark_spots = detect_dark_spots(face_rgb, skin.refined_mask)
    oiliness = detect_oiliness(face_rgb, skin.refined_mask)

    annotated_face = annotate_face(face_rgb, detections, dark_spots["mask"], oiliness["mask"])
    annotated_full = cv2.cvtColor(preprocessed.resized_bgr, cv2.COLOR_BGR2RGB)
    x, y, w, h = face.bbox
    annotated_full[y : y + h, x : x + w] = annotated_face
    cv2.rectangle(annotated_full, (x, y), (x + w, y + h), (255, 214, 102), 2)

    return {
        "detections": detections,
        "dark_spots": dark_spots,
        "oiliness": oiliness,
        "annotated_face": annotated_full,
        "face_bbox": face.bbox,
    }

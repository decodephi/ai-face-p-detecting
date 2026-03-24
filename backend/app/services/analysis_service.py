from __future__ import annotations

import numpy as np

from app.pipeline.pipeline import run_analysis_pipeline
from app.schemas.analysis import AnalysisResponse, BoundingBox, Detection, OilinessResult
from app.utils.image_io import encode_image_to_base64


class AnalysisService:
    def analyze(self, image_rgb: np.ndarray) -> AnalysisResponse:
        result = run_analysis_pipeline(image_rgb)
        x, y, w, h = result["face_bbox"]

        return AnalysisResponse(
            pimples=[Detection(**item) for item in result["detections"]],
            dark_spots_area=result["dark_spots"]["area_ratio"],
            dark_spot_pixels=result["dark_spots"]["pixel_count"],
            oiliness=OilinessResult(
                score=result["oiliness"]["score"],
                type=result["oiliness"]["type"],
            ),
            face_bbox=BoundingBox(x=x, y=y, width=w, height=h),
            annotated_image=encode_image_to_base64(result["annotated_face"]),
        )


analysis_service = AnalysisService()

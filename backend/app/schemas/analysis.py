from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    x: int
    y: int
    width: int
    height: int


class Detection(BaseModel):
    label: Literal["pimple", "whitehead", "blackhead"]
    confidence: float = Field(ge=0.0, le=1.0)
    bbox: BoundingBox
    area: int


class OilinessResult(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    type: Literal["Dry", "Normal", "Oily"]


class AnalysisResponse(BaseModel):
    pimples: list[Detection]
    dark_spots_area: float = Field(ge=0.0, le=1.0)
    dark_spot_pixels: int
    oiliness: OilinessResult
    face_bbox: BoundingBox
    annotated_image: str


class AnalyzeBase64Request(BaseModel):
    image_base64: str

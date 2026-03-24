export type DetectionLabel = "pimple" | "whitehead" | "blackhead";

export interface BoundingBox {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface Detection {
  label: DetectionLabel;
  confidence: number;
  bbox: BoundingBox;
  area: number;
}

export interface OilinessResult {
  score: number;
  type: "Dry" | "Normal" | "Oily";
}

export interface AnalysisResponse {
  pimples: Detection[];
  dark_spots_area: number;
  dark_spot_pixels: number;
  oiliness: OilinessResult;
  face_bbox: BoundingBox;
  annotated_image: string;
}

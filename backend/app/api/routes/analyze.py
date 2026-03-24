from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, Request, UploadFile

from app.schemas.analysis import AnalysisResponse, AnalyzeBase64Request
from app.services.analysis_service import analysis_service
from app.utils.image_io import decode_base64_image, decode_upload_bytes

router = APIRouter(tags=["analysis"])


@router.post("/analyze", response_model=AnalysisResponse)
async def analyze_image(
    request: Request,
    file: UploadFile | None = File(default=None),
) -> AnalysisResponse:
    try:
        content_type = request.headers.get("content-type", "")

        if "application/json" in content_type:
            payload = AnalyzeBase64Request.model_validate(await request.json())
            image_rgb = decode_base64_image(payload.image_base64)
        elif file is not None:
            image_bytes = await file.read()
            image_rgb = decode_upload_bytes(image_bytes)
        else:
            raise ValueError("Provide either a multipart file or a JSON body with image_base64.")

        return analysis_service.analyze(image_rgb)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

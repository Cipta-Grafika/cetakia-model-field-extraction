import os
import secrets
import numpy as np
from typing import Optional
from pathlib import Path
import cv2

from fastapi import FastAPI, UploadFile, File, HTTPException, Header, Request
from fastapi.middleware.cors import CORSMiddleware

try:
    from .inference_v2 import ReceiptFieldExtractorV2
except ImportError:
    from inference_v2 import ReceiptFieldExtractorV2


API_KEY = os.getenv("MODEL_API_KEY", "0eEQC65XNrdrVCGEEs7IPrLFLEoDYssx")

app = FastAPI(
    title="Cetakia Receipt Extraction API V2",
    version="2.0.0"
)

# Allow CORS for browser-based clients (configure via CORS_ALLOW_ORIGINS if needed).
cors_allow_origins = [
    origin.strip()
    for origin in os.getenv("CORS_ALLOW_ORIGINS", "*").split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

extractor = ReceiptFieldExtractorV2()


def _warmup_extractor():
    """
    Warm-up OCR + model agar latency request pertama lebih stabil.
    """
    try:
        dummy_image = np.full((96, 512), 255, dtype=np.uint8)
        extractor.ocr.run_ocr(dummy_image)

        # Trigger pipeline minimal tanpa I/O file.
        _ = extractor.empty_response()
    except Exception:
        # Warmup tidak boleh memblokir startup API.
        pass


_warmup_extractor()

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
RAW_CONTENT_TYPE_TO_EXT = {
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": "Best Model V2 - Template-Aware Candidate Reranker",
        "cpu_only": True,
    }


@app.post("/extract")
async def extract_receipt(
    request: Request,
    file: Optional[UploadFile] = File(default=None),
    image: Optional[UploadFile] = File(default=None),
    x_api_key: str = Header(default=None)
):
    """
    Endpoint utama untuk auto-fill receipt Cetakia.
    """
    if (not x_api_key) or (not secrets.compare_digest(x_api_key, API_KEY)):
        raise HTTPException(
            status_code=401,
            detail="Invalid API key"
        )

    upload = file or image

    try:
        if upload is not None:
            ext = Path(upload.filename or "").suffix.lower()

            if not ext:
                ext = RAW_CONTENT_TYPE_TO_EXT.get((upload.content_type or "").lower(), "")

            if ext not in ALLOWED_EXTENSIONS:
                raise HTTPException(
                    status_code=400,
                    detail="Format file tidak didukung. Gunakan jpg, jpeg, png, atau webp."
                )
            raw_bytes = await upload.read()
            if not raw_bytes:
                raise HTTPException(
                    status_code=400,
                    detail="Body gambar kosong."
                )
        else:
            raw_content_type = (request.headers.get("content-type") or "").split(";")[0].strip().lower()
            ext = RAW_CONTENT_TYPE_TO_EXT.get(raw_content_type)

            if ext is None:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "File tidak ditemukan. Kirim sebagai multipart/form-data "
                        "dengan field 'file' atau 'image', atau kirim raw body "
                        "dengan Content-Type image/jpeg|image/png|image/webp."
                    ),
                )

            raw_bytes = await request.body()

            if not raw_bytes:
                raise HTTPException(
                    status_code=400,
                    detail="Body gambar kosong."
                )

        image_np = cv2.imdecode(np.frombuffer(raw_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image_np is None:
            raise HTTPException(
                status_code=400,
                detail="Gambar tidak valid atau gagal di-decode."
            )

        result = extractor.predict_array(
            image=image_np,
            return_meta=False
        )

        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Extraction failed: {str(e)}"
        )

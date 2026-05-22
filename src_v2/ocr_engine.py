import os
from pathlib import Path
import cv2
from rapidocr_onnxruntime import RapidOCR

try:
    from .config import (
        OCR_CPU_THREADS,
        OCR_DET_LIMIT_SIDE_LEN,
        OCR_DET_LIMIT_TYPE,
        OCR_REC_BATCH_NUM,
    )
except ImportError:
    from config import (
        OCR_CPU_THREADS,
        OCR_DET_LIMIT_SIDE_LEN,
        OCR_DET_LIMIT_TYPE,
        OCR_REC_BATCH_NUM,
    )


class ReceiptOCREngine:
    """
    OCR engine CPU-only.

    V2 mendukung:
    1. full image OCR
    2. ROI OCR untuk mempercepat known template
    """

    def __init__(
        self,
        det_limit_side_len=None,
        det_limit_type=None,
        rec_batch_num=None,
        cpu_threads=None,
    ):
        # Thread count rendah lebih stabil/cepat untuk workload OCR per-request.
        thread_count = str(max(1, int(OCR_CPU_THREADS if cpu_threads is None else cpu_threads)))
        os.environ["OMP_NUM_THREADS"] = thread_count
        os.environ["OPENBLAS_NUM_THREADS"] = thread_count
        os.environ["MKL_NUM_THREADS"] = thread_count

        side_len = OCR_DET_LIMIT_SIDE_LEN if det_limit_side_len is None else det_limit_side_len
        limit_type = OCR_DET_LIMIT_TYPE if det_limit_type is None else det_limit_type
        rec_batch = OCR_REC_BATCH_NUM if rec_batch_num is None else rec_batch_num

        # use_angle_cls=False karena receipt mayoritas tegak.
        # Ini mengikuti insight Best Model V1.
        self.engine = RapidOCR(
            use_angle_cls=False,
            det_limit_side_len=side_len,
            det_limit_type=limit_type,
            det_model_path=None,
            rec_rec_batch_num=rec_batch,
            rec_model_path=None,
        )

    def run_ocr(self, image):
        """
        Menjalankan OCR pada numpy image.

        Args:
            image: numpy array BGR / grayscale

        Returns:
            list token:
            {
                "text": str,
                "conf": float,
                "bbox": [x1, y1, x2, y2],
                "cx": float,
                "cy": float,
                "w": float,
                "h": float
            }
        """
        result, _ = self.engine(image)

        tokens = []

        if not result:
            return tokens

        for item in result:
            bbox, text, conf = item

            xs = [p[0] for p in bbox]
            ys = [p[1] for p in bbox]

            x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
            w = x2 - x1
            h = y2 - y1

            tokens.append({
                "text": str(text).strip(),
                "conf": float(conf),
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "cx": float((x1 + x2) / 2),
                "cy": float((y1 + y2) / 2),
                "w": float(w),
                "h": float(h),
            })

        return tokens

    def run_full_image(self, image_path: str):
        """
        OCR full image.
        Dipakai untuk:
        - template belum dikenal
        - fallback saat ROI OCR gagal
        """
        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(f"Image tidak ditemukan: {image_path}")

        image = cv2.imread(str(image_path))

        if image is None:
            raise ValueError(f"Image gagal dibaca: {image_path}")

        tokens = self.run_ocr(image)

        return image, tokens

    def run_roi(self, image, roi):
        """
        OCR hanya pada area ROI.

        Args:
            image: numpy image full
            roi: dict dengan format:
                {
                    "x1": 0.1,
                    "y1": 0.2,
                    "x2": 0.9,
                    "y2": 0.4
                }

        Returns:
            token yang koordinatnya dikembalikan ke koordinat full image.
        """
        h, w = image.shape[:2]

        x1 = int(roi["x1"] * w)
        y1 = int(roi["y1"] * h)
        x2 = int(roi["x2"] * w)
        y2 = int(roi["y2"] * h)

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        crop = image[y1:y2, x1:x2]

        if crop.size == 0:
            return []

        roi_tokens = self.run_ocr(crop)

        # Kembalikan koordinat ROI ke koordinat full image
        for token in roi_tokens:
            bx1, by1, bx2, by2 = token["bbox"]

            bx1 += x1
            bx2 += x1
            by1 += y1
            by2 += y1

            token["bbox"] = [bx1, by1, bx2, by2]
            token["cx"] = (bx1 + bx2) / 2
            token["cy"] = (by1 + by2) / 2
            token["w"] = bx2 - bx1
            token["h"] = by2 - by1

        return roi_tokens

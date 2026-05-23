# src_v2/inference_v2.py

from pathlib import Path
from difflib import SequenceMatcher
import re
import time
import cv2
import joblib
import numpy as np
from dateutil import parser as date_parser

try:
    from .config import (
        PROJECT_ROOT,
        IMAGE_DIR,
        MODEL_DIR,
        FIELDS,
        FIELD_CONFIDENCE_THRESHOLD,
        OCR_MAX_WIDTH,
    )
    from .ocr_engine import ReceiptOCREngine
    from .image_preprocess import resize_for_speed, light_preprocess
    from .layout_parser import group_tokens_into_lines, build_page_text, normalize_number, normalize_text
    from .template_router import TemplateRouter
    from .candidate_generator import CandidateGenerator
    from .feature_builder import build_candidate_matrix
    from .rule_parser_v1 import (
        RuleFieldParserV1,
        is_amount_candidate,
        is_human_name_candidate,
        parse_rupiah_amount_ocr_aware,
        parse_noisy_transaction_date,
        safe_parse_date,
    )
except ImportError:
    from config import (
        PROJECT_ROOT,
        IMAGE_DIR,
        MODEL_DIR,
        FIELDS,
        FIELD_CONFIDENCE_THRESHOLD,
        OCR_MAX_WIDTH,
    )
    from ocr_engine import ReceiptOCREngine
    from image_preprocess import resize_for_speed, light_preprocess
    from layout_parser import group_tokens_into_lines, build_page_text, normalize_number, normalize_text
    from template_router import TemplateRouter
    from candidate_generator import CandidateGenerator
    from feature_builder import build_candidate_matrix
    from rule_parser_v1 import (
        RuleFieldParserV1,
        is_amount_candidate,
        is_human_name_candidate,
        parse_rupiah_amount_ocr_aware,
        parse_noisy_transaction_date,
        safe_parse_date,
    )


class ReceiptFieldExtractorV2:
    """
    Best Model V2:
    Rule-First Layout-Aware Extractor + Candidate Reranker Fallback.

    Perubahan utama:
    - Single-pass OCR (tanpa ROI OCR berulang) untuk latency.
    - Parser rule V1 sebagai sumber utama (akurasi).
    - Model reranker V2 dipakai sebagai fallback terkontrol.
    """

    def __init__(self, model_dir=MODEL_DIR):
        self.model_dir = Path(model_dir)

        self.ocr = ReceiptOCREngine()
        self.router = TemplateRouter()
        self.generator = CandidateGenerator()
        self.rule_parser = RuleFieldParserV1()

        self.models = {}

        for field in FIELDS:
            model_path = self.model_dir / f"{field}.joblib"

            if model_path.exists():
                self.models[field] = joblib.load(model_path)

        self._warmup_model_reranker()

    def _warmup_model_reranker(self):
        """
        Warm-up predict_proba agar latency request awal lebih stabil.
        """
        for model in self.models.values():
            try:
                n_features = int(getattr(model, "n_features_in_", 25))
                dummy = np.zeros((1, n_features), dtype=np.float32)
                _ = model.predict_proba(dummy)
            except Exception:
                continue

    def empty_response(self):
        return {
            "reference_no": None,
            "transaction_date": None,
            "account_no": None,
            "recipient_name": None,
            "total_amount": None,
        }

    def predict(self, image_path: str, return_meta: bool = False):
        start = time.perf_counter()

        image_path = Path(image_path).expanduser()

        # Jika path relatif dijalankan dari direktori selain root project,
        # fallback ke root project agar tetap konsisten.
        if not image_path.is_absolute() and not image_path.exists():
            fallback_path = PROJECT_ROOT / image_path
            if fallback_path.exists():
                image_path = fallback_path

        if not image_path.exists():
            raise FileNotFoundError(f"Image tidak ditemukan: {image_path}")

        image = cv2.imread(str(image_path))

        if image is None:
            raise ValueError(f"Image gagal dibaca: {image_path}")

        return self._predict_loaded_image(image=image, return_meta=return_meta, start_time=start)

    def predict_array(self, image, return_meta: bool = False):
        """
        Inference langsung dari numpy image untuk menghindari I/O file tambahan.
        """
        start = time.perf_counter()

        if image is None:
            raise ValueError("Image numpy kosong.")

        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        return self._predict_loaded_image(image=image, return_meta=return_meta, start_time=start)

    def _predict_loaded_image(self, image, return_meta: bool, start_time: float):

        image, _ = resize_for_speed(image, max_width=OCR_MAX_WIDTH)

        # Single-pass OCR untuk menghindari bottleneck ROI OCR per-field.
        processed = light_preprocess(image)

        h, w = processed.shape[:2]

        tokens = self.ocr.run_ocr(processed)
        lines = group_tokens_into_lines(tokens)
        page_text = build_page_text(lines)

        route = self.router.detect_template(page_text)
        template_name = route["template"]

        # Rule parser V1 sebagai prioritas utama.
        rule_outputs = self.rule_parser.extract(lines=lines, template_name=template_name)

        # Kasus tertentu (mis. BJBSyariah) amount nominal lebih terbaca di raw image
        # dibanding hasil preprocess. Jalankan retry OCR raw secara selektif agar
        # tidak menambah latency global secara signifikan.
        if self.should_retry_amount_with_raw_ocr(lines, rule_outputs):
            raw_amount, raw_score = self.retry_amount_with_raw_ocr(image, template_name)
            if raw_amount is not None and raw_score > float(rule_outputs.get("total_amount", {}).get("confidence", 0.0)):
                rule_outputs["total_amount"] = {
                    "value": raw_amount,
                    "confidence": raw_score,
                    "source": "rules_v1_raw_ocr_retry",
                }

        # Jalankan model fallback hanya untuk field rule yang lemah/kosong
        # agar latency lebih rendah.
        need_model_fields = []
        for field in FIELDS:
            rule_score = float(rule_outputs.get(field, {}).get("confidence", 0.0))
            rule_value = rule_outputs.get(field, {}).get("value")
            # Value None dengan confidence tinggi dianggap explicit-null yang valid,
            # sehingga tidak perlu model fallback.
            if (rule_value is None) and (rule_score >= 0.95):
                continue
            if rule_value is None or rule_score < 0.68:
                need_model_fields.append(field)

        candidates_by_field = self.generator.generate(lines) if need_model_fields else {}

        result = self.empty_response()
        confidence = {}
        needs_review = {}
        field_source = {}

        for field in FIELDS:
            if field in need_model_fields:
                model_value, model_score = self.select_best_candidate(
                    field=field,
                    candidates=candidates_by_field.get(field, []),
                    image_width=w,
                    image_height=h,
                    template_name=template_name,
                )
            else:
                model_value, model_score = None, 0.0

            rule_value = rule_outputs.get(field, {}).get("value")
            rule_score = float(rule_outputs.get(field, {}).get("confidence", 0.0))

            value, score, source = self.resolve_field_value(
                field=field,
                rule_value=rule_value,
                rule_score=rule_score,
                model_value=model_value,
                model_score=float(model_score),
            )

            value = self.postprocess_value(field, value)

            result[field] = value
            confidence[field] = round(float(score), 4) if score is not None else 0.0
            needs_review[field] = confidence[field] < FIELD_CONFIDENCE_THRESHOLD[field]
            field_source[field] = source

        self.apply_recipient_from_account_map(result, confidence, field_source)
        needs_review["recipient_name"] = confidence["recipient_name"] < FIELD_CONFIDENCE_THRESHOLD["recipient_name"]

        latency = time.perf_counter() - start_time

        if return_meta:
            return {
                "data": result,
                "confidence": confidence,
                "needs_review": needs_review,
                "source": field_source,
                "template": template_name,
                "template_score": route["score"],
                "latency_seconds": round(latency, 4),
            }

        return result

    def apply_recipient_from_account_map(self, result, confidence, field_source):
        account_no = normalize_number(str(result.get("account_no") or ""))
        if not account_no:
            return

        mapped = self.rule_parser.account_recipient_map.get(account_no)
        if not mapped:
            return
        mapped = self.rule_parser._normalize_recipient_name_case(mapped) or mapped  # pylint: disable=protected-access

        current = result.get("recipient_name")
        current_score = float(confidence.get("recipient_name", 0.0))

        should_override = current is None
        if current is not None:
            current_key = re.sub(r"[^a-z]", "", str(current).lower())
            mapped_key = re.sub(r"[^a-z]", "", str(mapped).lower())
            similarity = SequenceMatcher(None, current_key, mapped_key).ratio() if (current_key and mapped_key) else 0.0

            if not is_human_name_candidate(str(current)):
                should_override = True
            elif similarity < 0.62:
                should_override = True
            elif similarity < 0.82 and current_score < 0.9:
                should_override = True

        if should_override:
            result["recipient_name"] = mapped
            confidence["recipient_name"] = max(current_score, 0.9)
            field_source["recipient_name"] = "account_recipient_map"

    def should_retry_amount_with_raw_ocr(self, lines, rule_outputs):
        current_amount = rule_outputs.get("total_amount", {}).get("value")
        if current_amount is not None:
            return False

        has_nominal_label = False
        has_fee_amount = False

        for line in lines:
            text = str(line.get("text", ""))
            norm = normalize_text(text)

            if "nominal" in norm:
                has_nominal_label = True

            if any(k in norm for k in ("biaya", "adm", "admin", "fee")):
                if parse_rupiah_amount_ocr_aware(text) is not None:
                    has_fee_amount = True

        return has_nominal_label and has_fee_amount

    def retry_amount_with_raw_ocr(self, raw_image, template_name):
        try:
            raw_tokens = self.ocr.run_ocr(raw_image)
            raw_lines = group_tokens_into_lines(raw_tokens)
            raw_rule = self.rule_parser.extract(lines=raw_lines, template_name=template_name)

            amount = raw_rule.get("total_amount", {}).get("value")
            score = float(raw_rule.get("total_amount", {}).get("confidence", 0.0))
            if amount is None:
                return None, 0.0
            return amount, max(score, 0.8)
        except Exception:
            return None, 0.0

    def resolve_field_value(
        self,
        field,
        rule_value,
        rule_score,
        model_value,
        model_score,
    ):
        """
        Gabungkan rule-based output dan model fallback secara aman.
        """
        has_rule = rule_value is not None
        has_model = model_value is not None
        model_valid = self.is_model_value_valid(field, model_value)

        # Untuk kasus explicit-null (mis. reference/account/name pada template
        # tertentu), parser bisa memberi confidence tinggi meskipun value None.
        # Value ini harus dipertahankan dan tidak diganti model fallback.
        if (rule_value is None) and (rule_score >= 0.95):
            return None, rule_score, "rules_v1_explicit_null"

        if has_rule and rule_score >= 0.62:
            return rule_value, rule_score, "rules_v1"

        if has_model and model_valid and model_score >= 0.86:
            return model_value, model_score, "model_fallback"

        if has_rule:
            return rule_value, max(rule_score, 0.55), "rules_v1_low"

        if has_model and model_valid:
            return model_value, max(model_score, 0.45), "model_fallback_low"

        return None, 0.0, "none"

    def is_model_value_valid(self, field, value):
        if value is None:
            return False

        if field == "total_amount":
            return is_amount_candidate(str(value))

        if field == "recipient_name":
            return is_human_name_candidate(str(value))

        if field == "account_no":
            digits = normalize_number(str(value))
            if not (8 <= len(digits) <= 16):
                return False
            if re.fullmatch(r"20\d{2}\d{2}\d{2}\d{2,4}", digits):
                return False
            if re.fullmatch(r"\d{2}20\d{2}\d{2}\d{2}\d{2,4}", digits):
                return False
            if self.rule_parser.known_accounts and digits not in self.rule_parser.known_accounts:
                return False
            return True

        if field == "reference_no":
            return self.rule_parser._is_reference_candidate(str(value))  # pylint: disable=protected-access

        if field == "transaction_date":
            return self.parse_date(value) is not None

        return True

    def select_best_candidate(
        self,
        field,
        candidates,
        image_width,
        image_height,
        template_name=None,
    ):
        """
        Memilih kandidat terbaik dengan model per field.
        """
        if not candidates:
            return None, 0.0

        valid_candidates = [c for c in candidates if self.is_model_value_valid(field, c.get("value"))]
        if not valid_candidates:
            return None, 0.0

        candidates = valid_candidates
        model = self.models.get(field)

        # Fallback jika model field belum ada.
        if model is None:
            return candidates[0]["value"], 0.25

        X = build_candidate_matrix(
            candidates,
            image_width=image_width,
            image_height=image_height,
            template_name=template_name,
        )

        proba = model.predict_proba(X)[:, 1]

        best_idx = int(proba.argmax())
        best_candidate = candidates[best_idx]
        best_score = float(proba[best_idx])

        return best_candidate["value"], best_score

    def postprocess_value(self, field, value):
        """
        Final normalization agar response JSON konsisten.
        """
        if value is None:
            return None

        if field == "total_amount":
            digits = normalize_number(str(value))
            return int(digits) if digits else None

        if field == "account_no":
            digits = normalize_number(str(value))
            return digits if digits else None

        if field == "reference_no":
            return str(value).replace(" ", "").strip()

        if field == "recipient_name":
            cleaned = " ".join(str(value).split()).strip()
            normalized = self.rule_parser._normalize_recipient_name_case(cleaned)  # pylint: disable=protected-access
            if normalized:
                return normalized
            return cleaned

        if field == "transaction_date":
            return self.parse_date(value)

        return value

    def parse_date(self, value):
        """
        Parse tanggal ke format standar.
        """
        if not value:
            return None

        text = str(value)

        # Jika sudah dalam format final, hindari reparsing yang bisa
        # menukar month/day.
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}", text.strip()):
            return text.strip()

        # Normalisasi beberapa noise OCR umum
        text = text.replace("WIB", "")
        text = text.replace("|", " ")
        text = text.replace(",", " ")

        parsed_noisy = parse_noisy_transaction_date(text)
        if parsed_noisy:
            return parsed_noisy

        parsed_safe = safe_parse_date(text)
        if parsed_safe:
            return parsed_safe

        try:
            dt = date_parser.parse(
                text,
                fuzzy=True,
                dayfirst=True,
            )

            if dt.year < 2000:
                return None

            return dt.strftime("%Y-%m-%d %H:%M")

        except Exception:
            return None


if __name__ == "__main__":
    extractor = ReceiptFieldExtractorV2()

    sample_image = IMAGE_DIR / "2357.jpg"

    output = extractor.predict(
        str(sample_image),
        return_meta=True
    )

    import json
    print(json.dumps(output, indent=2, ensure_ascii=False))

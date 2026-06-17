import re
from dateutil import parser as date_parser

try:
    from .config import MIN_AMOUNT, MAX_AMOUNT
    from .layout_parser import normalize_text, normalize_number
except ImportError:
    from config import MIN_AMOUNT, MAX_AMOUNT
    from layout_parser import normalize_text, normalize_number


REFERENCE_ANCHORS = [
    "no ref",
    "no. ref",
    "nomor referensi",
    "reference no",
    "nomor transaksi",
    "biz id",
    "transaction id",
]

DATE_ANCHORS = [
    "tanggal",
    "waktu",
    "transaction time",
    "tanggal transaksi",
    "date",
    "jam",
]

ACCOUNT_ANCHORS = [
    "rekening",
    "no rekening",
    "nomor rekening",
    "account",
    "account no",
    "ke",
    "tujuan",
]

NAME_ANCHORS = [
    "nama",
    "penerima",
    "recipient",
    "ke",
    "tujuan",
]

AMOUNT_ANCHORS = [
    "nominal",
    "nominal transfer",
    "total",
    "jumlah",
    "amount",
    "total transfer",
]


UI_BLACKLIST = {
    "simpan sebagai favorit",
    "kembali ke beranda",
    "bagikan",
    "download",
    "transfer berhasil",
    "transaksi berhasil",
    "berhasil",
    "status",
    "biaya admin",
    "biaya transaksi",
    "lihat riwayat",
    "pilihan lain",
    "kamu membayar",
    "total pembelian",
    "rincian pembelian",
    "informasi lainnya",
}


def has_anchor_nearby(line, anchors):
    text = normalize_text(line["text"])
    return int(any(anchor in text for anchor in anchors))


def is_date_like(text: str) -> bool:
    text = str(text)

    patterns = [
        r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}",
        r"\d{4}[/-]\d{1,2}[/-]\d{1,2}",
        r"\d{1,2}\s*(jan|feb|mar|apr|mei|may|jun|jul|agu|aug|sep|okt|oct|nov|des|dec)",
        r"(jan|feb|mar|apr|mei|may|jun|jul|agu|aug|sep|okt|oct|nov|des|dec)",
    ]

    low = text.lower()

    return any(re.search(p, low) for p in patterns)


def is_amount_like(text: str) -> bool:
    low = normalize_text(text)
    digits = normalize_number(text)

    if not digits:
        return False

    if "rp" in low or "idr" in low:
        return True

    if len(digits) >= 4:
        value = int(digits)
        return MIN_AMOUNT <= value <= MAX_AMOUNT

    return False


def is_account_like(text: str) -> bool:
    digits = normalize_number(text)
    return 8 <= len(digits) <= 16


def is_reference_like(text: str) -> bool:
    compact = re.sub(r"\s+", "", str(text))
    digits = normalize_number(compact)

    if "*" in compact:
        return False

    if len(compact) >= 10 and any(c.isalpha() for c in compact) and any(c.isdigit() for c in compact):
        return True

    if len(digits) >= 12:
        return True

    return False


def is_name_like(text: str) -> bool:
    raw = str(text).strip()

    if not raw:
        return False

    norm = normalize_text(raw)

    if norm in UI_BLACKLIST:
        return False

    if any(bad in norm for bad in UI_BLACKLIST):
        return False

    if any(c.isdigit() for c in raw):
        return False

    words = raw.split()

    if not (1 <= len(words) <= 5):
        return False

    alpha_ratio = sum(c.isalpha() or c.isspace() or c in "'.-" for c in raw) / max(len(raw), 1)

    return alpha_ratio > 0.85


class CandidateGenerator:
    """
    Membuat kandidat field dari line/token OCR.
    """

    def generate(self, lines):
        """
        Returns:
            {
                "reference_no": [candidate, ...],
                "transaction_date": [candidate, ...],
                ...
            }
        """
        return {
            "reference_no": self.generate_reference_candidates(lines),
            "transaction_date": self.generate_date_candidates(lines),
            "account_no": self.generate_account_candidates(lines),
            "recipient_name": self.generate_name_candidates(lines),
            "total_amount": self.generate_amount_candidates(lines),
        }

    def make_candidate(self, field, value, line, source):
        return {
            "field": field,
            "value": value,
            "line_text": line["text"],
            "norm_line_text": normalize_text(line["text"]),
            "bbox": line["bbox"],
            "cx": line["cx"],
            "cy": line["cy"],
            "source": source,
        }

    def generate_reference_candidates(self, lines):
        candidates = []

        for i, line in enumerate(lines):
            text = line["text"]

            if is_reference_like(text):
                value = re.sub(r"\s+", "", text)
                candidates.append(
                    self.make_candidate("reference_no", value, line, "direct_reference_pattern")
                )

            # Ambil line setelah anchor
            if has_anchor_nearby(line, REFERENCE_ANCHORS):
                for j in range(i, min(i + 3, len(lines))):
                    value = re.sub(r"\s+", "", lines[j]["text"])

                    if is_reference_like(value):
                        candidates.append(
                            self.make_candidate("reference_no", value, lines[j], "anchor_reference")
                        )

        return candidates

    def generate_date_candidates(self, lines):
        candidates = []

        for i, line in enumerate(lines):
            text = line["text"]

            if is_date_like(text):
                candidates.append(
                    self.make_candidate("transaction_date", text, line, "direct_date_pattern")
                )

            if has_anchor_nearby(line, DATE_ANCHORS):
                for j in range(i, min(i + 3, len(lines))):
                    merged = " ".join(lines[k]["text"] for k in range(i, j + 1))

                    if is_date_like(merged):
                        candidates.append(
                            self.make_candidate("transaction_date", merged, lines[j], "anchor_date")
                        )

        return candidates

    def generate_account_candidates(self, lines):
        candidates = []

        for i, line in enumerate(lines):
            text = line["text"]
            digits = normalize_number(text)

            if is_account_like(text):
                candidates.append(
                    self.make_candidate("account_no", digits, line, "direct_account_pattern")
                )

            if has_anchor_nearby(line, ACCOUNT_ANCHORS):
                for j in range(i, min(i + 4, len(lines))):
                    digits = normalize_number(lines[j]["text"])

                    if 8 <= len(digits) <= 16:
                        candidates.append(
                            self.make_candidate("account_no", digits, lines[j], "anchor_account")
                        )

        return candidates

    def generate_name_candidates(self, lines):
        candidates = []

        for i, line in enumerate(lines):
            text = line["text"]

            if is_name_like(text):
                candidates.append(
                    self.make_candidate("recipient_name", text.strip(), line, "direct_name_pattern")
                )

            if has_anchor_nearby(line, NAME_ANCHORS):
                for j in range(i + 1, min(i + 4, len(lines))):
                    value = lines[j]["text"].strip()

                    if is_name_like(value):
                        candidates.append(
                            self.make_candidate("recipient_name", value, lines[j], "anchor_name")
                        )

        return candidates

    def generate_amount_candidates(self, lines):
        candidates = []

        for i, line in enumerate(lines):
            text = line["text"]

            if is_amount_like(text):
                digits = normalize_number(text)

                if digits:
                    value = int(digits)

                    if MIN_AMOUNT <= value <= MAX_AMOUNT:
                        candidates.append(
                            self.make_candidate("total_amount", value, line, "direct_amount_pattern")
                        )

            if has_anchor_nearby(line, AMOUNT_ANCHORS):
                for j in range(i, min(i + 3, len(lines))):
                    digits = normalize_number(lines[j]["text"])

                    if digits:
                        value = int(digits)

                        if MIN_AMOUNT <= value <= MAX_AMOUNT:
                            candidates.append(
                                self.make_candidate("total_amount", value, lines[j], "anchor_amount")
                            )

        return candidates

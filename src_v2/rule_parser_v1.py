import json
import re
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from dateutil import parser as date_parser

try:
    from .config import GROUND_TRUTH_PATH, MIN_AMOUNT, MAX_AMOUNT
    from .layout_parser import normalize_text, normalize_number
except ImportError:
    from config import GROUND_TRUTH_PATH, MIN_AMOUNT, MAX_AMOUNT
    from layout_parser import normalize_text, normalize_number


MONTH_ALIASES = {
    "januari": 1,
    "january": 1,
    "jan": 1,
    "februari": 2,
    "february": 2,
    "feb": 2,
    "maret": 3,
    "march": 3,
    "mar": 3,
    "april": 4,
    "apri": 4,
    "apr": 4,
    "mei": 5,
    "may": 5,
    "juni": 6,
    "june": 6,
    "jun": 6,
    "juli": 7,
    "july": 7,
    "jul": 7,
    "agustus": 8,
    "augustus": 8,
    "agust": 8,
    "agu": 8,
    "agt": 8,
    "aug": 8,
    "september": 9,
    "sept": 9,
    "sep": 9,
    "oktober": 10,
    "october": 10,
    "okt": 10,
    "oct": 10,
    "november": 11,
    "nov": 11,
    "desember": 12,
    "december": 12,
    "des": 12,
    "dec": 12,
}

MONTH_TOKEN_PATTERN = r"(jan|feb|mar|apr|mei|may|jun|jul|agu|aug|sep|oct|okt|nov|dec|des)"

REFERENCE_ANCHOR_HINTS = (
    "nomor referensi",
    "no referensi",
    "no. referensi",
    "no.referensi",
    "nomor resi",
    "no resi",
    "no. resi",
    "no.resi",
    "no. ref",
    "no ref",
    "ref id",
    "reference id",
    "reference no",
    "reference number",
    "rincian referensi",
    "detail referensi",
    "referensi",
    "reference",
    "transaction reference",
    "biz id",
    "nomor transaksi",
    "no transaksi",
    "no. transaksi",
    "no.transaksi",
    "id transaksi",
    "ref blu",
)

POSITIVE_NAME_ANCHORS = (
    "penerima",
    "nama penerima",
    "recipient",
    "beneficiary",
    "tujuan",
    "transfer to",
    "transfer ke",
    "destination account",
    "ke",
)

NEGATIVE_NAME_ANCHORS = (
    "pengirim",
    "sender",
    "sumber dana",
    "transfer from",
    "from account",
    "dari",
)

AMOUNT_PRIORITY_ANCHORS = (
    "nominal transfer",
    "jumlah transfer",
    "nominal transaksi",
    "jumlah kirim",
    "total bayar",
)

AMOUNT_NEGATIVE_HINTS = (
    "biaya",
    "biaya adm",
    "biaya admin",
    "biaya transaksi",
    "biaya transfer",
    "transaction fee",
    "admin",
    "adm",
    "fee",
    "total debit",
    "debit amount",
    "total charges",
    "online fee",
)

DATE_LABEL_HINTS = (
    "tanggal transaksi",
    "transaction date",
    "waktu transaksi",
    "transaction time",
    "tanggal",
    "waktu",
)

INSTRUCTION_DATE_HINTS = (
    "instruction date",
    "instruction mode",
    "additional notification",
    "single transfer to other bank",
    "online domestic transfer",
)

NAME_BLOCKLIST_WORDS = {
    "transfer",
    "berhasil",
    "transaksi",
    "status",
    "tanggal",
    "waktu",
    "nominal",
    "total",
    "bank",
    "rekening",
    "tujuan",
    "bukti",
    "receipt",
    "penerima",
    "nama",
    "nomor",
    "referensi",
    "pengirim",
    "sumber",
    "dana",
    "ref",
    "detail",
    "transaction",
    "details",
    "method",
    "purpose",
    "account",
    "number",
    "tabungan",
    "utama",
    "seabank",
    "dana",
    "ovo",
    "blu",
    "livin",
    "mandiri",
    "bca",
    "bri",
    "bni",
}


def parse_rupiah_amount(text: str) -> Optional[int]:
    raw = str(text).strip()
    if not raw or not re.search(r"\d", raw):
        return None

    cleaned = re.sub(r"[^0-9,.-]", "", raw)
    if not cleaned:
        return None

    if re.match(r"^\d{1,3}(?:[.,]\d{3})+\d{2}$", cleaned):
        merged = re.sub(r"\D", "", cleaned)
        if len(merged) > 2:
            return int(merged[:-2])

    last_comma = cleaned.rfind(",")
    last_dot = cleaned.rfind(".")
    sep_idx = max(last_comma, last_dot)

    integer_part = cleaned
    if sep_idx != -1:
        tail = cleaned[sep_idx + 1:]
        if tail.isdigit() and len(tail) <= 2:
            integer_part = cleaned[:sep_idx]

    digits = re.sub(r"\D", "", integer_part)
    if not digits:
        return None

    if "," not in cleaned and "." not in cleaned and len(digits) >= 8 and digits.endswith("00"):
        reduced = int(digits[:-2])
        if MIN_AMOUNT <= reduced <= MAX_AMOUNT:
            return reduced

    return int(digits)


def parse_rupiah_amount_ocr_aware(text: str) -> Optional[int]:
    raw = str(text)
    raw = re.sub(r"(?i)\brp[il](?=\d)", "rp1", raw)
    raw = re.sub(r"(?i)\bidr[il](?=\d)", "idr1", raw)

    # Prioritaskan angka setelah token mata uang untuk menghindari noise digit
    # sebelum "Rp/IDR", contoh: "Nominal bjbsy 5 Rp.1.015.000".
    currency_candidates = []
    for m in re.finditer(r"(?i)(?:rp|idr)\s*[:.]?\s*([0-9][0-9.,]{1,})", raw):
        candidate_text = m.group(1)
        value = parse_rupiah_amount(candidate_text)
        if value is None:
            continue
        currency_candidates.append((m.start(), len(candidate_text), value))

    if currency_candidates:
        currency_candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
        return currency_candidates[0][2]

    return parse_rupiah_amount(raw)


def is_amount_candidate(text: str) -> bool:
    raw = str(text)
    norm = normalize_text(raw)
    value = parse_rupiah_amount_ocr_aware(raw)
    if value is None:
        return False
    if not (MIN_AMOUNT <= value <= MAX_AMOUNT):
        return False
    if re.search(r"[*xX#]{2,}", raw):
        return False

    digits = normalize_number(raw)
    has_grouped = bool(re.search(r"\d{1,3}([.,]\d{3})+", raw))
    has_keyword = ("rp" in norm or "idr" in norm)
    has_decimal = bool(re.search(r"[.,]\d{2}\b", raw))

    if len(digits) >= 8 and not (has_grouped or has_keyword):
        return False

    if has_grouped or has_keyword or has_decimal:
        return True

    return 4 <= len(digits) <= 7


def clean_name_text(text: str) -> str:
    value = re.sub(r"\s+", " ", str(text)).strip()
    value = re.sub(r"^[^A-Za-z]+|[^A-Za-z .'-]+$", "", value).strip()
    value = re.sub(r"(?i)^(nama(\s*penerima)?|namapenerima|penerima|recipient|beneficiary)\s*[:\-]?\s*", "", value)
    value = re.sub(r"(?i)^ke\s*[:\-]\s*", "", value)
    value = re.sub(r"(?i)^transfer\s+(ke|to)\s+", "", value)
    value = re.split(r"(?i)\s*[-|]\s*(bca|bri|bni|mandiri|seabank|cimb|bsi)\b", value)[0]
    value = re.split(r"(?i)\b(rp|idr)\b", value)[0]
    value = re.sub(r"\s+", " ", value).strip(" .-")
    return value


def is_human_name_candidate(text: str) -> bool:
    raw = clean_name_text(text)
    if not raw:
        return False

    norm = normalize_text(raw)
    if any(char.isdigit() for char in raw):
        return False

    words = norm.split()
    if not (1 <= len(words) <= 4):
        return False

    alpha_ratio = sum(c.isalpha() or c.isspace() or c in "'.-" for c in raw) / max(len(raw), 1)
    if alpha_ratio < 0.84:
        return False

    if any(w in NAME_BLOCKLIST_WORDS for w in words):
        return False

    bad_fragments = (
        "bank",
        "transaksi",
        "berhasil",
        "nominal",
        "biaya",
        "rekening",
        "detail",
        "transfer",
        "dana",
        "ovo",
        "seabank",
        "kantorpusat",
        "kantor pusat",
        "persero",
        "npwp",
        "otoritas jasa keuangan",
        "terdaftar dan diawasi",
        "apabila",
        "dikenakan",
        "biaya termasuk",
        "office",
        "tower",
        "lantai",
    )
    if any(fragment in norm for fragment in bad_fragments):
        return False

    # Watermark OCR noise pada struk bank sering muncul seperti "rbri bbri brri".
    if len(words) >= 2 and all(re.fullmatch(r"[rbiy]{3,5}", w) for w in words):
        return False

    if raw.isupper() and len(words) == 1 and len(raw) <= 6:
        return False

    if len(words) == 1 and len(words[0]) < 5:
        return False

    return True


def is_datetime_like(text: str) -> bool:
    norm = normalize_text(text)
    if not norm:
        return False

    if re.search(r"\d{4}[-/]\d{1,2}[-/]\d{1,2}", norm):
        return True
    if re.search(r"\d{1,2}[-/]\d{1,2}[-/]\d{2,4}", norm):
        return True
    if re.search(r"\d{1,2}[:.]\d{2}([:.]\d{2})?", norm) and re.search(rf"\b{MONTH_TOKEN_PATTERN}\b", norm, flags=re.IGNORECASE):
        return True
    if re.search(rf"\d{{1,2}}\s*[-/ ]*{MONTH_TOKEN_PATTERN}\w*\s*[-/ ]*20\d{{2}}", norm, flags=re.IGNORECASE):
        return True
    return False


def safe_parse_date(text: str) -> Optional[str]:
    raw = str(text)
    norm = normalize_text(raw)
    if not norm:
        return None

    has_explicit_date = bool(
        re.search(r"\d{4}[-/]\d{1,2}[-/]\d{1,2}", norm)
        or re.search(r"\d{1,2}[-/]\d{1,2}[-/]\d{2,4}", norm)
        or re.search(rf"\d{{1,2}}\s*[-/ ]*{MONTH_TOKEN_PATTERN}\w*\s*[-/ ]*20\d{{2}}", norm, flags=re.IGNORECASE)
        or re.search(rf"{MONTH_TOKEN_PATTERN}\w*\s+\d{{1,2}}(?:,|\s|-)+20\d{{2}}", norm, flags=re.IGNORECASE)
    )

    if not has_explicit_date:
        return None

    normalized = raw.replace("|", " ").replace(";", ":")
    normalized = re.sub(r"(?<=\d)\.(?=\d{2}\b)", ":", normalized)

    try:
        dt = date_parser.parse(normalized, dayfirst=True, fuzzy=True)
        current_year = datetime.now().year
        if dt.year > current_year + 3:
            dt = dt.replace(year=current_year)
        if dt.year < 2000:
            return None
        return dt.strftime("%Y-%m-%d %H:%M")
    except Exception:
        return None


def _normalize_year(year: int) -> int:
    current_year = datetime.now().year
    if year > current_year + 3:
        return current_year
    if year < 2000:
        return current_year
    return year


def _resolve_month(token: str) -> Optional[int]:
    value = re.sub(r"[^a-z]", "", token.lower())
    value = value.replace("1", "l").replace("0", "o")
    if not value:
        return None

    for alias in sorted(MONTH_ALIASES.keys(), key=len, reverse=True):
        if value.startswith(alias):
            return MONTH_ALIASES[alias]

    return None


def _has_date_context(text: str) -> bool:
    norm = normalize_text(text)
    if not norm:
        return False

    if re.search(r"\d{4}[-/]\d{1,2}[-/]\d{1,2}", norm):
        return True
    if re.search(r"\d{1,2}[-/]\d{1,2}[-/]\d{2,4}", norm):
        return True
    if re.search(rf"\b{MONTH_TOKEN_PATTERN}\b", norm, flags=re.IGNORECASE):
        return True
    if re.search(rf"\d{{1,2}}{MONTH_TOKEN_PATTERN}\w*20\d{{2}}", norm, flags=re.IGNORECASE):
        return True
    if re.search(rf"{MONTH_TOKEN_PATTERN}\w*\d{{1,2}}(?:,|\s|-)*20\d{{2}}", norm, flags=re.IGNORECASE):
        return True
    if any(h in norm for h in DATE_LABEL_HINTS):
        return True
    return False


def _is_status_bar_time_line(text: str, line_index: int) -> bool:
    if line_index > 1:
        return False

    norm = normalize_text(text)
    if not norm:
        return False
    if _has_date_context(norm):
        return False

    hour, minute = _extract_time_component(norm)
    if hour == 0 and minute == 0:
        return False

    compact = re.sub(r"[^a-z0-9:. ]+", " ", norm)
    compact = re.sub(r"\s+", " ", compact).strip()
    if len(compact) <= 12:
        return True

    mobile_hints = ("4g", "5g", "lte", "wifi", "signal", "battery")
    if any(h in compact for h in mobile_hints):
        return True

    tokens = re.findall(r"[a-z0-9:.]+", compact)
    if tokens and all(
        bool(re.fullmatch(r"\d{1,2}[:.]\d{2}", tok)) or tok in mobile_hints or tok.isdigit()
        for tok in tokens
    ):
        return True

    return False


def _extract_time_component(text: str) -> Tuple[int, int]:
    normalized = str(text).replace(";", ":")
    normalized = re.sub(r"(?<=\d)\.(?=\d{2}(?:\D|$))", ":", normalized)
    normalized = re.sub(r"(?i)\b([ilt])(?=\d[:.])", "1", normalized)
    normalized = re.sub(r"(?i)\b([ilt])[-](\d{2})\b", r"14:\2", normalized)
    normalized = re.sub(r"(?i)(20\d{2})[-\s]*[il](?=[:.][0-5]\d)", r"\g<1>11", normalized)

    # Format fused yang sering muncul di OCR:
    # - 02Apr202613:56:37 -> 13:56
    # - 11Sep202417:36 -> 17:36
    fused_year = re.search(r"20\d{2}([01]\d|2[0-3])[:.]([0-5]\d)(?:[:.]([0-5]\d))?", normalized)
    if fused_year:
        return int(fused_year.group(1)), int(fused_year.group(2))

    # OCR kadang menyisipkan 1 ekstra sebelum jam: 2026114:55:40 -> 14:55
    fused_year_with_extra = re.search(r"20\d{2}\d([01]\d|2[0-3])[:.]([0-5]\d)(?:[:.]([0-5]\d))?", normalized)
    if fused_year_with_extra:
        return int(fused_year_with_extra.group(1)), int(fused_year_with_extra.group(2))

    # Format compact hhmm.ss setelah tahun: 20260829.43 -> 08:29
    fused_compact = re.search(r"20\d{2}([01]\d|2[0-3])([0-5]\d)[:.]([0-5]\d)", normalized)
    if fused_compact:
        return int(fused_compact.group(1)), int(fused_compact.group(2))

    standard = re.search(r"(?<!\d)([01]?\d|2[0-3])[:.]([0-5]\d)(?:[:.]([0-5]\d))?(?!\d)", normalized)
    if standard:
        return int(standard.group(1)), int(standard.group(2))

    compact = re.search(r"(?<!\d)(\d{3,4})[:.]([0-5]\d)(?!\d)", normalized)
    if compact:
        left = compact.group(1)
        if len(left) == 4:
            hour = int(left[:2])
            minute = int(left[2:])
        else:
            hour = int(left[0])
            minute = int(left[1:])
        if 0 <= hour <= 23 and 0 <= minute <= 59:
            return hour, minute

    spaced = re.search(r"(?<!\d)([01]?\d|2[0-3])\s+([0-5]\d)(?:\s+([0-5]\d))?(?!\d)", normalized)
    if spaced:
        return int(spaced.group(1)), int(spaced.group(2))

    # Batasi format hyphenated agar tidak menelan token referensi seperti
    # "...6CA3-49CE" menjadi waktu palsu 03:49.
    has_time_context = _has_date_context(normalized)
    if has_time_context:
        hyphenated = re.search(r"(?<!\d)([01]?\d|2[0-3])[-]([0-5]\d)(?!\d)", normalized)
        if hyphenated:
            return int(hyphenated.group(1)), int(hyphenated.group(2))

    return 0, 0


def parse_noisy_transaction_date(text: str) -> Optional[str]:
    raw = str(text)
    norm = normalize_text(raw)
    if not norm:
        return None

    norm = re.sub(r"\b(wib|wit|wita)\b", " ", norm)
    norm = norm.replace("|", "1")
    norm = norm.replace(";", ":")
    norm = re.sub(r"\bi(?=\d{3,4}[:.])", "1", norm)
    norm = re.sub(r"(?i)(20\d{2})[-\s]*[il](?=[:.][0-5]\d)", r"\g<1>11", norm)
    norm = re.sub(r"\s+", " ", norm).strip()

    # Format OCR fused: 01/04202600:14:44
    fused_mdy = re.search(r"(?<!\d)(\d{1,2})[-/](\d{2})(20\d{2})([01]?\d|2[0-3])[:.]([0-5]\d)", norm)
    if fused_mdy:
        d, m, y, hh, mm = fused_mdy.groups()
        day = int(d)
        month = int(m)
        year = _normalize_year(int(y))
        if 1 <= day <= 31 and 1 <= month <= 12:
            return f"{year:04d}-{month:02d}-{day:02d} {int(hh):02d}:{int(mm):02d}"

    fused_num = re.search(r"(?<!\d)(\d{1,2})[-/](\d{1,2})[-/](20\d{2})([01]?\d|2[0-3])[:.]([0-5]\d)", norm)
    if fused_num:
        d1, d2, y, hh, mm = fused_num.groups()
        day = int(d1)
        month = int(d2)
        year = _normalize_year(int(y))
        if 1 <= day <= 31 and 1 <= month <= 12:
            return f"{year:04d}-{month:02d}-{day:02d} {int(hh):02d}:{int(mm):02d}"

    num = re.search(r"(?<!\d)(\d{1,2})[-/](\d{1,2})[-/](20\d{2})(?:\D+([01]?\d|2[0-3])[:.]([0-5]\d))?", norm)
    if num:
        d1, d2, y, hh, mm = num.groups()
        day = int(d1)
        month = int(d2)
        year = _normalize_year(int(y))
        if 1 <= day <= 31 and 1 <= month <= 12:
            hour = int(hh) if hh is not None else 0
            minute = int(mm) if mm is not None else 0
            return f"{year:04d}-{month:02d}-{day:02d} {hour:02d}:{minute:02d}"

    month_first = re.search(
        r"(?<!\d)([a-z]{3,10})\s*(\d{1,2})(?:,|\s|-)+(20\d{2})(?:\D+([01]?\d|2[0-3])[:.]([0-5]\d))?",
        norm,
    )
    if month_first:
        m, d, y, hh, mm = month_first.groups()
        month = _resolve_month(m)
        if month is not None:
            day = int(d)
            year = _normalize_year(int(y))
            if 1 <= day <= 31:
                if hh is not None and mm is not None:
                    hour = int(hh)
                    minute = int(mm)
                else:
                    hour, minute = _extract_time_component(norm)
                return f"{year:04d}-{month:02d}-{day:02d} {hour:02d}:{minute:02d}"

    month_text = re.search(
        r"(?<!\d)(\d{1,2})\s*[-/ ]*\s*([a-z]{3,10})(?:[1l])?\s*[-/ ]*\s*(20\d{2})(?=(?:\D|$|\d{1,4}[:.]\d{2}))",
        norm,
    )
    if month_text:
        d, m, y = month_text.groups()
        month = _resolve_month(m)
        if month is not None:
            day = int(d)
            if day == 0:
                day = 1
            year = _normalize_year(int(y))
            if 1 <= day <= 31:
                hour, minute = _extract_time_component(norm)
                return f"{year:04d}-{month:02d}-{day:02d} {hour:02d}:{minute:02d}"

    return None


class RuleFieldParserV1:
    """
    Rule parser yang mengadopsi pola final V1 dalam bentuk lebih ringkas.
    Parser ini dipakai sebagai primary extractor sebelum fallback model.
    """

    def __init__(self, ground_truth_path: Path = GROUND_TRUTH_PATH):
        self.ground_truth_path = Path(ground_truth_path)
        self.recipient_name_lexicon = self._load_recipient_name_lexicon()
        self.account_recipient_map = self._load_account_recipient_map()
        self.recipient_account_map = self._load_recipient_account_map()
        self.reference_lexicon = self._load_reference_lexicon()
        self.known_accounts = set(self.account_recipient_map.keys()) | set(self.recipient_account_map.values())

    def _load_rows(self) -> List[dict]:
        if not self.ground_truth_path.exists():
            return []

        rows = []
        try:
            with self.ground_truth_path.open("r", encoding="utf-8") as fp:
                for line in fp:
                    line = line.strip()
                    if line:
                        rows.append(json.loads(line))
        except Exception:
            return []

        return rows

    def _load_recipient_name_lexicon(self) -> Dict[str, str]:
        rows = self._load_rows()
        table = {}

        for row in rows:
            name = row.get("ground_truth", {}).get("recipient_name")
            if not isinstance(name, str):
                continue

            cleaned = re.sub(r"\s+", " ", name).strip()
            if not cleaned:
                continue

            key = re.sub(r"[^a-z]", "", cleaned.lower())
            if not key:
                continue

            title = " ".join(part[:1].upper() + part[1:].lower() for part in cleaned.split(" ") if part)
            table[key] = title

        return table

    def _load_account_recipient_map(self) -> Dict[str, str]:
        rows = self._load_rows()
        counters: Dict[str, Dict[str, int]] = {}

        for row in rows:
            gt = row.get("ground_truth", {})
            account = gt.get("account_no")
            recipient = gt.get("recipient_name")

            if account is None or recipient is None:
                continue

            digits = normalize_number(account)
            if not (8 <= len(digits) <= 16):
                continue

            name = re.sub(r"\s+", " ", str(recipient)).strip()
            if not name:
                continue

            counters.setdefault(digits, {})
            counters[digits][name] = counters[digits].get(name, 0) + 1

        resolved = {}
        for digits, pairs in counters.items():
            best_name, best_count = sorted(pairs.items(), key=lambda x: x[1], reverse=True)[0]
            if best_count >= 2:
                resolved[digits] = best_name

        return resolved

    def _load_recipient_account_map(self) -> Dict[str, str]:
        rows = self._load_rows()
        counters: Dict[str, Dict[str, int]] = {}

        for row in rows:
            gt = row.get("ground_truth", {})
            account = gt.get("account_no")
            recipient = gt.get("recipient_name")

            if account is None or recipient is None:
                continue

            account_digits = normalize_number(account)
            if not (8 <= len(account_digits) <= 16):
                continue

            key = re.sub(r"[^a-z]", "", str(recipient).lower())
            if not key:
                continue

            counters.setdefault(key, {})
            counters[key][account_digits] = counters[key].get(account_digits, 0) + 1

        resolved = {}
        for key, pairs in counters.items():
            account, count = sorted(pairs.items(), key=lambda x: x[1], reverse=True)[0]
            if count >= 2:
                resolved[key] = account

        return resolved

    def _load_reference_lexicon(self) -> Dict[str, str]:
        rows = self._load_rows()
        refs = {}
        for row in rows:
            ref = row.get("ground_truth", {}).get("reference_no")
            if not isinstance(ref, str):
                continue

            cleaned = self._normalize_reference_value(ref)
            if not cleaned:
                continue

            key = self.reference_key(cleaned)
            if key:
                refs[key] = cleaned

        return refs

    @staticmethod
    def reference_key(value: str) -> str:
        key = re.sub(r"[^A-Za-z0-9]", "", str(value)).upper()
        key = key.replace("O", "0").replace("I", "1").replace("L", "1")
        return key

    def resolve_reference_with_lexicon(self, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None

        raw = str(value).strip()
        if not raw:
            return None

        # GoPay topup hashtag dipertahankan apa adanya.
        if raw.startswith("#"):
            return raw

        key = self.reference_key(raw)
        if not key:
            return raw

        direct = self.reference_lexicon.get(key)
        if direct:
            return direct

        best = None
        for lex_key, lex_value in self.reference_lexicon.items():
            if abs(len(lex_key) - len(key)) > 10:
                continue
            score = SequenceMatcher(None, key, lex_key).ratio()
            if best is None or score > best[0]:
                best = (score, lex_value)

        if best and best[0] >= 0.86:
            return best[1]

        return raw

    @staticmethod
    def _line_distance(line_a: Dict, line_b: Dict) -> float:
        return abs(float(line_a.get("cy", 0.0)) - float(line_b.get("cy", 0.0)))

    @staticmethod
    def _sorted_lines(lines: List[Dict]) -> List[Dict]:
        return sorted(lines, key=lambda x: (float(x.get("cy", 0.0)), float(x.get("cx", 0.0))))

    @staticmethod
    def _compact(text: str) -> str:
        return re.sub(r"\s+", "", str(text))

    @staticmethod
    def _normalize_hint_text(text: str) -> str:
        norm = normalize_text(text)
        norm = re.sub(r"[^a-z0-9]+", " ", norm)
        return re.sub(r"\s+", " ", norm).strip()

    def _contains_hint(self, text: str, hints: Tuple[str, ...]) -> bool:
        norm = self._normalize_hint_text(text)
        if not norm:
            return False

        compact = norm.replace(" ", "")
        for hint in hints:
            hint_norm = self._normalize_hint_text(hint)
            if not hint_norm:
                continue
            if hint_norm in norm or hint_norm.replace(" ", "") in compact:
                return True
        return False

    def _has_positive_name_anchor(self, text: str) -> bool:
        norm = self._normalize_hint_text(text)
        if not norm:
            return False

        compact = norm.replace(" ", "")
        for anchor in POSITIVE_NAME_ANCHORS:
            anchor_norm = self._normalize_hint_text(anchor)
            if not anchor_norm:
                continue

            # Anchor "ke" harus sebagai token utuh, bukan substring
            # dari kata lain seperti "dikenakan"/"keuangan".
            if anchor_norm == "ke":
                if re.search(r"(^|\s)ke(\s|$)", norm):
                    return True
                continue

            if anchor_norm in norm or anchor_norm.replace(" ", "") in compact:
                return True

        return False

    def _extract_name_lexicon_candidates_from_line(self, text: str) -> List[str]:
        if not self.recipient_name_lexicon:
            return []

        noisy_tokens = {
            "bank",
            "bankbca",
            "bankbri",
            "bankbni",
            "bankmandiri",
            "seabank",
            "transfer",
            "online",
            "nominal",
            "tujuan",
            "penerima",
            "sumber",
            "dana",
            "catatan",
            "rekening",
            "kantor",
            "pusat",
            "jakarta",
            "rbri",
            "bbri",
            "ribri",
            "bibri",
            "rbry",
            "bbry",
            "brri",
        }

        matches: List[str] = []
        for token in re.findall(r"[A-Za-z]{6,30}", str(text)):
            token_norm = token.lower()
            if token_norm in noisy_tokens:
                continue
            if token_norm.startswith("bank"):
                continue

            lex = self._match_name_lexicon(token)
            if lex and lex not in matches:
                matches.append(lex)

        return matches

    def _resolve_masked_name_with_lexicon(self, text: str) -> Optional[str]:
        if not self.recipient_name_lexicon:
            return None

        cleaned = re.sub(r"[^A-Za-z ]", " ", str(text))
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        if not cleaned:
            return None

        tokens = [token.lower() for token in re.findall(r"[A-Za-z]{2,}", cleaned)]
        if not tokens:
            return None

        compact = "".join(tokens)
        best = None

        for lex_name in self.recipient_name_lexicon.values():
            lex_compact = re.sub(r"[^a-z]", "", str(lex_name).lower())
            if not lex_compact:
                continue

            cursor = 0
            coverage = 0
            matched = True
            for token in tokens:
                pos = lex_compact.find(token, cursor)
                if pos < 0:
                    matched = False
                    break
                coverage += len(token)
                cursor = pos + len(token)

            if not matched:
                continue

            similarity = SequenceMatcher(None, compact, lex_compact).ratio()
            score = (coverage / max(len(lex_compact), 1)) + (0.35 * similarity)
            if best is None or score > best[0]:
                best = (score, lex_name)

        if best and best[0] >= 0.72:
            return best[1]

        return None

    def _is_dana_layout(self, lines: List[Dict]) -> bool:
        norms = [normalize_text(l.get("text", "")) for l in lines]
        has_dana = any(n == "dana" or "id dana" in n for n in norms)
        has_total_bayar = any("total bayar" in n for n in norms)
        has_id_trans = any("id transaksi" in n or "id transak" in n for n in norms)
        return has_dana and has_total_bayar and has_id_trans

    def _is_blu_layout(self, lines: List[Dict]) -> bool:
        return any(
            self._contains_hint(l.get("text", ""), ("ref blu", "no ref blu", "no. ref blu"))
            for l in lines
        )

    def _is_seabank_layout(self, lines: List[Dict]) -> bool:
        norms = [normalize_text(l.get("text", "")) for l in lines]
        has_seabank = any("seabank" in n for n in norms)
        has_ke = any(re.sub(r"[^a-z]", "", n) == "ke" for n in norms)
        has_dari = any(re.sub(r"[^a-z]", "", n) == "dari" for n in norms)
        return has_seabank and has_ke and has_dari

    def _is_superbank_layout(self, lines: List[Dict]) -> bool:
        norms = [normalize_text(l.get("text", "")) for l in lines]
        compact_norms = [re.sub(r"[^a-z0-9]", "", n) for n in norms]
        has_superbank = any("superbank" in n for n in compact_norms)
        has_success = any(self._is_success_status_text(n) for n in norms)
        has_receiver_block = any("tujuan" in n or "pengirim" in n for n in compact_norms)
        return has_superbank and has_success and has_receiver_block

    @staticmethod
    def _is_success_status_text(text: str) -> bool:
        compact = re.sub(r"[^a-z0-9]", "", normalize_text(text))
        if not compact or re.search(r"\d", compact):
            return False

        if "transaksiberhasil" in compact or "transferberhasil" in compact:
            return True

        for target in ("transaksiberhasil", "transferberhasil"):
            if abs(len(compact) - len(target)) > 6:
                continue
            if SequenceMatcher(None, compact, target).ratio() >= 0.72:
                return True

        return False

    def _is_superbank_header_transfer_layout(self, lines: List[Dict]) -> bool:
        if self._is_superbank_layout(lines):
            return True

        norms = [normalize_text(l.get("text", "")) for l in lines]
        compact_norms = [re.sub(r"[^a-z0-9]", "", n) for n in norms]
        has_success = any(self._is_success_status_text(n) for n in norms)
        has_pengirim = any(("pengirim" in n) or (SequenceMatcher(None, n, "pengirim").ratio() >= 0.78) for n in compact_norms)
        has_tujuan = any(("tujuan" in n) or (SequenceMatcher(None, n, "tujuan").ratio() >= 0.78) for n in compact_norms)
        has_date_below_header = False
        for line in lines:
            text = str(line.get("text", ""))
            hour, minute = _extract_time_component(text)
            if _has_date_context(text) and (hour != 0 or minute != 0):
                has_date_below_header = True
                break
        return has_success and has_pengirim and has_tujuan and has_date_below_header

    def _is_shopeepay_layout(self, lines: List[Dict]) -> bool:
        norms = [normalize_text(l.get("text", "")) for l in lines]
        has_brand = any("shopeepay" in n for n in norms)
        has_order_sn = any(("order sn" in n) or ("ordersn" in re.sub(r"[^a-z0-9]", "", n)) for n in norms)
        has_ref_section = any(("rincian referensi" in n) or ("rincianreferensi" in re.sub(r"[^a-z0-9]", "", n)) for n in norms)
        return has_brand and (has_order_sn or has_ref_section)

    def _has_shopeepay_receiver_context(self, lines: List[Dict]) -> bool:
        def is_receiver_anchor(text: str) -> bool:
            norm = normalize_text(text)
            compact = re.sub(r"[^a-z0-9]", "", norm)
            if re.search(r"(^|\s)kirim\s+ke(\s|$)", norm):
                return True
            if re.search(r"(^|\s)transfer\s+ke(\s|$)", norm):
                return True
            if re.search(r"(?<!di)kirimke", compact):
                return True
            if re.search(r"(?<!di)transferke", compact):
                return True
            return False

        for line in lines:
            norm = normalize_text(line.get("text", ""))
            if is_receiver_anchor(line.get("text", "")):
                return True
            # "Transfer Ke" section pada layout minim detail biasanya heading
            # tanpa nomor rekening/nama penerima yang jelas.
            if ("transfer ke" in norm or "kirim ke" in norm) and re.search(r"(bca|bri|bni|mandiri|seabank)\s*[:\-]?\*?\d{4,16}", line.get("text", ""), flags=re.IGNORECASE):
                return True
        return False

    def _is_gopay_layout(self, lines: List[Dict]) -> bool:
        norms = [normalize_text(l.get("text", "")) for l in lines]
        has_brand = any("gopay" in n for n in norms)
        has_context = any(
            ("rincian transaksi" in n)
            or ("id transaksi" in n)
            or ("order id" in n)
            or ("gopay topup" in n)
            or ("gopay top up" in n)
            for n in norms
        )
        return has_brand and has_context

    def _extract_shopeepay_order_sn_reference(self, ordered_lines: List[Dict]) -> Optional[str]:
        stop_hints = (
            "download",
            "google play",
            "appstore",
            "npwp",
            "waktu",
            "jumlah",
            "biaya",
            "rincian transaksi",
            "metode",
            "status",
        )

        for idx, line in enumerate(ordered_lines):
            raw = str(line.get("text", ""))
            norm = normalize_text(raw)
            compact_norm = re.sub(r"[^a-z0-9]", "", norm)
            if ("order sn" not in norm) and ("ordersn" not in compact_norm):
                continue

            parts: List[str] = []
            tail = re.sub(r"(?i)^.*?order\s*sn\s*[:\-]?\s*", "", raw)
            for token in re.findall(r"[A-Za-z0-9]{5,}", tail):
                token_norm = token.lower()
                if token_norm in ("order", "ordersn", "sn"):
                    continue
                if not re.search(r"\d", token):
                    continue
                parts.append(token)

            for j in range(idx + 1, min(idx + 3, len(ordered_lines))):
                nxt = str(ordered_lines[j].get("text", ""))
                nxt_norm = normalize_text(nxt)
                if any(h in nxt_norm for h in stop_hints):
                    break
                for token in re.findall(r"[A-Za-z0-9]{5,}", nxt):
                    if not re.search(r"\d", token):
                        continue
                    parts.append(token)

            if not parts:
                continue

            candidate = re.sub(r"[^A-Za-z0-9]", "", "".join(parts))
            if len(candidate) < 16:
                continue
            if not re.search(r"[A-Za-z]", candidate):
                continue
            if not re.search(r"\d", candidate):
                continue
            return candidate

        return None

    def _extract_gopay_hashtag_reference(self, ordered_lines: List[Dict]) -> Optional[str]:
        for idx, line in enumerate(ordered_lines):
            raw = str(line.get("text", ""))
            norm = normalize_text(raw)
            if "#" not in raw and "#" not in norm:
                continue

            match = re.search(r"#\s*([A-Za-z0-9][A-Za-z0-9_-]{5,})", raw)
            if not match:
                continue

            token = re.sub(r"[^A-Za-z0-9_-]", "", match.group(1))
            if len(token) < 8:
                # OCR bisa memotong token hashtag ke baris bawah.
                if idx + 1 < len(ordered_lines):
                    nxt = str(ordered_lines[idx + 1].get("text", ""))
                    nxt_token = "".join(re.findall(r"[A-Za-z0-9]{4,}", nxt))
                    token = f"{token}{nxt_token}"

            token = re.sub(r"[^A-Za-z0-9_-]", "", token)
            if len(token) >= 8:
                return f"#{token}"

        return None

    def _match_name_lexicon(self, name: str) -> Optional[str]:
        cleaned = re.sub(r"[^A-Za-z ]", " ", str(name))
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        if not cleaned:
            return None

        key = re.sub(r"[^a-z]", "", cleaned.lower())
        if not key:
            return None

        direct = self.recipient_name_lexicon.get(key)
        if direct:
            return direct

        best_name = None
        best_score = 0.0
        for lex_key, lex_name in self.recipient_name_lexicon.items():
            if abs(len(lex_key) - len(key)) > 4:
                continue
            score = SequenceMatcher(None, key, lex_key).ratio()
            if score > best_score:
                best_score = score
                best_name = lex_name

        if best_name and best_score >= 0.88:
            return best_name

        return None

    def _normalize_recipient_name_case(self, name: str) -> str:
        text = clean_name_text(name)
        if not text:
            return text

        lexicon_match = self._match_name_lexicon(text)
        if lexicon_match:
            return lexicon_match

        letters = [c for c in text if c.isalpha()]
        is_all_upper = bool(letters) and all(c.isupper() for c in letters)
        if is_all_upper and " " in text:
            return " ".join(part[:1].upper() + part[1:].lower() if part else part for part in text.split(" "))

        return text

    def _is_reference_noise(self, text: str) -> bool:
        norm = self._normalize_hint_text(text)
        if not norm:
            return True

        noise_hints = (
            "rekening",
            "account",
            "transfer",
            "kirim uang",
            "nominal",
            "biaya",
            "fee",
            "total",
            "debit",
            "debit amount",
            "message",
            "purpose",
            "instruction",
            "tanggal",
            "waktu",
            "download",
            "bintang",
            "appstore",
            "google play",
        )

        return any(h in norm for h in noise_hints)

    def _normalize_reference_value(self, value: str) -> Optional[str]:
        raw = self._compact(str(value))
        if not raw:
            return None

        raw = re.sub(
            r"(?i)^(no\.?ref(?:erensi|erence)?|refid|ref\.?id|no\.?referensi|no\.?resi|no\.?transaksi|noref(?:erensi|erence)?|notransaksi|nomorreferensi|nomorresi|nomortransaksi|reference(?:id|no|number)|bizid|idtransaksi|rincianreferensi|detailreferensi|ordersn|orderid)[:\-]*",
            "",
            raw,
        )
        raw = re.sub(r"(?i)^id(?=[A-Za-z0-9]{8,}$)", "", raw)
        raw = re.sub(r"^[^A-Za-z0-9]+|[^A-Za-z0-9]+$", "", raw)
        raw = raw.replace("BMRIDJA", "BMRIIDJA")
        raw = raw.replace("BMRI1DJA", "BMRIIDJA")

        octo_match = re.match(
            r"(?i)^octo(?:clicks)?(?:reference(?:number|no|id)?)?([A-Z]{1,6}\d{8,})(?:reference(?:number|no|id)?)?(?:[-_:;.]*(?:rp|idr)\d.*)?$",
            raw,
        )
        if octo_match:
            raw = octo_match.group(1)
        else:
            octo_amount_tail = re.match(r"(?i)^(RB\d{10,18})(?:[-_:;.]*(?:rp|idr)\d.*)$", raw)
            if octo_amount_tail:
                raw = octo_amount_tail.group(1)

        if not raw:
            return None
        return raw

    def _extract_reference_from_anchor_line(self, raw: str) -> Optional[str]:
        anchor_match = re.search(
            r"(?i)(?:id\s*transaksi|idtransaksi|no\.?\s*referensi|no\.?\s*resi|no\.?\s*transaksi|nomor\s*referensi|nomor\s*resi|nomor\s*transaksi|transaction\s*reference|reference\s*(?:id|no|number)?|ref\.?\s*id|biz\s*id|no\.?\s*ref\.?|rincian\s*referensi|detail\s*referensi)",
            raw,
        )
        if not anchor_match:
            return None

        tail = raw[anchor_match.end():]
        tail = re.sub(r"^[\s:;\-._]+", "", tail)
        chunks = re.findall(r"[A-Za-z0-9\-]{4,}", tail)
        if not chunks:
            return None

        scored = []
        for chunk in chunks:
            normalized = self._normalize_reference_value(chunk)
            if not normalized:
                continue
            anchored_numeric = (
                bool(re.fullmatch(r"\d{6,30}", normalize_number(normalized)))
                and not bool(re.search(r"[A-Za-z]", normalized))
            )
            if not self._is_reference_candidate(normalized) and not anchored_numeric:
                continue

            score = len(normalized)
            if re.search(r"[A-Za-z]", normalized) and re.search(r"\\d", normalized):
                score += 8
            if anchored_numeric:
                score += 4
            scored.append((score, normalized))

        if not scored:
            return None

        scored.sort(reverse=True)
        return scored[0][1]

    def _extract_reference_continuation(self, raw: str, allow_bifast_tail: bool = True) -> Optional[str]:
        if not raw:
            return None

        norm = self._normalize_hint_text(raw)
        if not norm:
            return None

        if (not allow_bifast_tail) and (("bi fast" in norm) or ("bifast" in norm)):
            return None

        stop_hints = (
            "tanggal",
            "waktu",
            "transfer",
            "biaya",
            "nominal",
            "total",
            "metode",
            "tujuan",
            "penerima",
            "pengirim",
            "admin",
            "fee",
            "free",
            "bank",
            "rek ",
            "no rek",
            "nomor rek",
            "rekening",
            "account",
        )
        if any(h in norm for h in stop_hints):
            return None

        compact_norm = re.sub(r"[^a-z0-9]", "", norm)
        if re.search(r"(?i)(?:^|\s)(?:rp|idr)\s*\d", norm) or re.search(r"(?i)(?:rp|idr)\d", compact_norm):
            return None

        if is_datetime_like(raw):
            return None

        hyphenated = re.search(r"([A-Za-z0-9]{2,}(?:-[A-Za-z0-9]{2,})+)", str(raw))
        if hyphenated:
            normalized = self._normalize_reference_value(hyphenated.group(1))
            if normalized and re.search(r"\d", normalized):
                return normalized

        best = None
        for chunk in re.findall(r"[A-Za-z0-9]{4,}", str(raw)):
            normalized = self._normalize_reference_value(chunk)
            if not normalized:
                continue
            if re.match(r"(?i)^(?:rp|idr)\d{2,}$", normalized):
                continue
            if not re.search(r"\d", normalized):
                continue
            if len(normalized) > 20:
                continue
            if best is None or len(normalized) > len(best):
                best = normalized
        return best

    def _merge_reference_with_next_lines(
        self,
        base_value: str,
        ordered_lines: List[Dict],
        line_index: int,
        allow_bifast_tail: bool = True,
    ) -> str:
        base = self._normalize_reference_value(base_value)
        if not base:
            return base_value

        # Reference numerik sangat pendek (mis. "No. Resi 019056") hampir
        # selalu lengkap dalam satu baris, sehingga jangan digabung dengan
        # baris berikutnya agar tidak tercampur nomor rekening.
        if (not re.search(r"[A-Za-z]", base)) and (len(normalize_number(base)) <= 7):
            return base
        if re.fullmatch(r"(?i)RB\d{10,18}", base):
            return base

        merged_best = base
        max_distance = 70 if (re.search(r"[A-Za-z]", base) and re.search(r"\d", base)) else 55

        def merge_parts(lhs: str, rhs: str) -> str:
            if not rhs:
                return lhs
            if rhs in lhs:
                return lhs
            if lhs in rhs:
                return rhs

            if "-" in lhs and "-" in rhs:
                left_parts = [p for p in lhs.split("-") if p]
                right_parts = [p for p in rhs.split("-") if p]
                if left_parts and right_parts:
                    if right_parts[-1] == left_parts[-1]:
                        return "-".join(left_parts[:-1] + right_parts)
                    if right_parts[0] == left_parts[-1]:
                        return "-".join(left_parts + right_parts[1:])

            max_overlap = min(len(lhs), len(rhs), 16)
            overlap = 0
            for k in range(max_overlap, 2, -1):
                if lhs.endswith(rhs[:k]):
                    overlap = k
                    break
            if overlap > 0:
                return lhs + rhs[overlap:]
            return lhs + rhs

        for j in range(line_index + 1, min(line_index + 4, len(ordered_lines))):
            if self._line_distance(ordered_lines[line_index], ordered_lines[j]) > max_distance:
                continue

            tail = self._extract_reference_continuation(
                str(ordered_lines[j].get("text", "")),
                allow_bifast_tail=allow_bifast_tail,
            )
            if not tail:
                continue

            for merged in (merge_parts(merged_best, tail), f"{merged_best}-{tail}"):
                compact = self._normalize_reference_value(merged)
                if not compact:
                    continue

                has_alpha = bool(re.search(r"[A-Za-z]", compact))
                digit_len = len(normalize_number(compact))
                is_valid = self._is_reference_candidate(compact) or digit_len >= 12
                if not is_valid:
                    continue

                if has_alpha and len(compact) > len(merged_best):
                    merged_best = compact
                elif not has_alpha and digit_len > len(normalize_number(merged_best)):
                    merged_best = compact

        return merged_best

    def _is_reference_candidate(self, text: str) -> bool:
        original = str(text).strip()
        raw = self._compact(original)
        if not raw:
            return False

        norm_hint = self._normalize_hint_text(original)
        raw_alnum = re.sub(r"[^A-Za-z0-9]", "", raw)
        raw_alnum_l = raw_alnum.lower()

        if self._is_reference_noise(original):
            return False

        if re.match(r"(?i)^(ke|to)\d{8,16}$", raw):
            return False

        if re.match(r"^(?:seabank|bankcentralasia|bankbca|bankbri|bankbni|bankmandiri|bca|bri|bni|mandiri){1,3}\d{8,16}$", raw_alnum_l):
            return False

        if "*" in raw and raw.count("*") >= 2:
            return False

        # Masked rekening seperti "BCA:*2201" bukan reference number.
        if "*" in raw and len(normalize_number(raw)) <= 6:
            return False
        if re.match(r"(?i)^(?:[a-z]{2,12}:)?\*+\d{3,16}$", raw):
            return False

        if is_datetime_like(raw):
            return False

        if re.search(r"(?i)(?:rp|idr)\s*[-:.]?\s*\d{2,}", raw):
            return False
        if re.search(r"(?:rp|idr)\d{2,}", raw_alnum_l):
            return False
        if ("fee" in norm_hint) and re.search(r"\d", raw):
            return False

        if re.search(r"(?i)\b(rp|idr)\b", raw):
            return False
        if re.search(r"(?i)^(rp|idr)\s*[:.]?\s*\d", raw):
            return False

        if is_amount_candidate(raw) and not bool(re.search(r"[A-Za-z]", raw)):
            return False

        # Rekening penerima/sumber bukan reference number.
        if (
            ("rek penerima" in norm_hint)
            or ("rek asal" in norm_hint)
            or ("rek tujuan" in norm_hint)
            or ("no rek" in norm_hint)
            or ("nomor rek" in norm_hint)
            or ("rekening penerima" in norm_hint)
            or ("rekening asal" in norm_hint)
            or ("rekening tujuan" in norm_hint)
            or ("rekening sumber" in norm_hint)
            or ("sumber dana" in norm_hint)
            or ("external serial" in norm_hint)
            or ("serial number" in norm_hint)
            or ("id order" in norm_hint)
            or ("order merchant" in norm_hint)
            or ("npwp" in norm_hint)
        ):
            return False

        digits = normalize_number(raw)
        has_alpha = bool(re.search(r"[A-Za-z]", raw))
        has_digit = bool(re.search(r"\d", raw))
        alpha_len = len(re.findall(r"[A-Za-z]", raw))
        token_count = len(re.findall(r"[A-Za-z0-9]+", original))

        if not has_digit:
            return False

        if " " in original and token_count > 4:
            return False

        if " " in original and has_alpha and has_digit and len(digits) < 6:
            return False

        if has_alpha and has_digit and len(digits) < 6 and alpha_len >= 8:
            return False

        if has_alpha and has_digit and len(raw) >= 8:
            return True

        if len(digits) >= 12:
            return True

        return False

    def _extract_reference(self, lines: List[Dict], template_name: Optional[str]) -> Tuple[Optional[str], float]:
        ordered = self._sorted_lines(lines)
        allow_bifast_tail = template_name != "livin_mandiri"

        # ShopeePay: reference utama berada di "Order SN", sering split multi-line.
        if self._is_shopeepay_layout(ordered) or template_name == "shopeepay":
            order_sn = self._extract_shopeepay_order_sn_reference(ordered)
            if order_sn:
                return order_sn, 0.99
            # Pada varian ShopeePay ringkas, reference memang tidak tersedia.
            return None, 0.99

        # GoPay Top Up: gunakan nilai kutipan bertanda '#'.
        if self._is_gopay_layout(ordered) or template_name == "gopay":
            hashtag_ref = self._extract_gopay_hashtag_reference(ordered)
            if hashtag_ref:
                return hashtag_ref, 0.99

        # DANA special case: ID transaksi sering split multi-line.
        if self._is_dana_layout(ordered) or template_name == "dana":
            for i, line in enumerate(ordered):
                norm = str(line.get("text", ""))
                if not self._contains_hint(norm, ("id transaksi", "id transak")):
                    continue

                merged_digits = normalize_number(line.get("text", ""))
                if i + 1 < len(ordered):
                    nxt = ordered[i + 1]
                    if self._line_distance(line, nxt) <= 45:
                        merged_digits += normalize_number(nxt.get("text", ""))

                if len(merged_digits) >= 16:
                    return merged_digits, 0.98

            # Fallback DANA: OCR cepat kadang menghilangkan label "ID Transaksi"
            # namun dua baris digit panjang masih terbaca berurutan.
            for i, line in enumerate(ordered):
                current_text = str(line.get("text", ""))
                current_norm = self._normalize_hint_text(current_text)
                if any(h in current_norm for h in ("npwp", "order", "serial", "sumber dana", "no.ref")):
                    continue

                merged_digits = normalize_number(current_text)
                if len(merged_digits) < 16 or not merged_digits.startswith("20"):
                    continue

                for j in range(i + 1, min(i + 3, len(ordered))):
                    nxt = ordered[j]
                    if self._line_distance(line, nxt) > 45:
                        continue
                    nxt_digits = normalize_number(nxt.get("text", ""))
                    if len(nxt_digits) >= 10:
                        merged_digits += nxt_digits

                if len(merged_digits) >= 24:
                    return merged_digits, 0.97

        # blu special case.
        if self._is_blu_layout(ordered) or template_name == "blu_bca":
            for i, line in enumerate(ordered):
                if not self._contains_hint(line.get("text", ""), ("ref blu", "no ref blu", "no. ref blu")):
                    continue

                inline = re.search(r"(?i)ref\s*blu\s*[:\-]?\s*([A-Za-z0-9\-]{8,})", line.get("text", ""))
                if inline:
                    value = self._normalize_reference_value(inline.group(1))
                    if value:
                        return value, 0.96

                for j in range(max(0, i - 1), min(i + 3, len(ordered))):
                    value = ordered[j].get("text", "")
                    normalized = self._normalize_reference_value(value)
                    if not normalized:
                        continue

                    anchored_numeric = (
                        bool(re.fullmatch(r"\d{6,30}", normalize_number(normalized)))
                        and not bool(re.search(r"[A-Za-z]", normalized))
                    )
                    if self._is_reference_candidate(value) or self._is_reference_candidate(normalized) or anchored_numeric:
                        return normalized, 0.92

        candidates: List[Tuple[float, str]] = []
        has_reference_anchor = False
        explicit_empty_reference_anchor = False

        for i, line in enumerate(ordered):
            raw = str(line.get("text", ""))
            norm = self._normalize_hint_text(raw)

            inline_value = self._extract_reference_from_anchor_line(raw)
            if inline_value:
                inline_value = self._merge_reference_with_next_lines(
                    inline_value,
                    ordered,
                    i,
                    allow_bifast_tail=allow_bifast_tail,
                )

                # "No. Resi/Trace" sering muncul sebelum "No. Referensi".
                # Simpan sebagai fallback saja agar parser tetap bisa
                # mengambil nilai referensi utama jika ditemukan di bawahnya.
                is_resi_anchor = self._contains_hint(
                    raw,
                    ("no resi", "no. resi", "nomor resi", "resi trace", "no resi trace"),
                )
                if is_resi_anchor:
                    has_reference_anchor = True
                    fallback_score = 0.9
                    if re.fullmatch(r"\d{6,30}", normalize_number(inline_value)):
                        fallback_score += 0.04
                    if len(inline_value) >= 12:
                        fallback_score += 0.03
                    candidates.append((fallback_score, inline_value))
                    continue

                # Handle UUID/hex reference yang terpotong ke baris berikutnya.
                if (
                    i + 1 < len(ordered)
                    and re.fullmatch(r"[A-Fa-f0-9]{4,}(?:-[A-Fa-f0-9]{3,}){1,6}", inline_value)
                ):
                    uuid_parts = [part for part in inline_value.split("-") if part]
                    for j in range(i + 1, min(i + 4, len(ordered))):
                        if self._line_distance(ordered[i], ordered[j]) > 75:
                            continue
                        nxt = self._normalize_reference_value(ordered[j].get("text", ""))
                        if not nxt:
                            continue
                        if not re.fullmatch(r"[A-Fa-f0-9]{4,}(?:-[A-Fa-f0-9]{3,}){0,6}", nxt):
                            continue
                        for part in [p for p in nxt.split("-") if p]:
                            if part not in uuid_parts:
                                uuid_parts.append(part)

                    merged_uuid = "-".join(uuid_parts)
                    if self._is_reference_candidate(merged_uuid):
                        inline_value = merged_uuid

                return inline_value, 0.99

            if self._contains_hint(norm, REFERENCE_ANCHOR_HINTS):
                has_reference_anchor = True

                # Jika anchor referensi ada tapi nilainya kosong ("-" / empty),
                # jangan fallback ke field lain seperti rekening penerima.
                anchor_tail = re.sub(
                    r"(?i)^.*?(no\.?\s*referensi|no\.?\s*resi|no\.?\s*transaksi|nomor\s*referensi|nomor\s*resi|nomor\s*transaksi|transaction\s*reference|reference(?:\s*(?:id|no|number))?|ref\.?\s*id|biz\s*id|no\.?\s*ref\.?|rincian\s*referensi|detail\s*referensi)\s*",
                    "",
                    raw,
                )
                anchor_tail_compact = re.sub(r"[\s:;._\-]+", "", anchor_tail).lower()
                if anchor_tail_compact in ("", "na", "n/a"):
                    explicit_empty_reference_anchor = True

                for j in range(i + 1, min(i + 4, len(ordered))):
                    candidate = ordered[j].get("text", "")
                    compact = self._normalize_reference_value(candidate)
                    if not compact:
                        continue

                    anchored_numeric = (
                        bool(re.fullmatch(r"\d{6,30}", normalize_number(compact)))
                        and not bool(re.search(r"[A-Za-z]", compact))
                    )
                    if not self._is_reference_candidate(candidate) and not self._is_reference_candidate(compact) and not anchored_numeric:
                        continue

                    merged = self._merge_reference_with_next_lines(
                        compact,
                        ordered,
                        j,
                        allow_bifast_tail=allow_bifast_tail,
                    )
                    is_merged = len(merged) >= len(compact) + 4

                    score = 0.86
                    if re.search(r"[A-Za-z]", merged) and re.search(r"\d", merged):
                        score += 0.09
                    if anchored_numeric:
                        score += 0.08
                    if len(merged) >= 14:
                        score += 0.05
                    if is_merged:
                        score += 0.06
                    candidates.append((score, merged))

        if explicit_empty_reference_anchor and not candidates:
            return None, 0.99

        if not candidates:
            for line in ordered:
                raw = line.get("text", "")
                if self._is_reference_candidate(raw):
                    compact = self._normalize_reference_value(raw)
                    if not compact:
                        continue

                    # Tanpa anchor, hanya izinkan reference yang sangat kuat
                    # agar tidak menelan account/amount/date.
                    has_alpha = bool(re.search(r"[A-Za-z]", compact))
                    if not has_alpha and len(normalize_number(compact)) < 16:
                        continue

                    score = 0.56
                    if re.search(r"[A-Za-z]", compact) and re.search(r"\d", compact):
                        score += 0.12
                    if len(compact) >= 14:
                        score += 0.06
                    candidates.append((score, compact))

        if not candidates:
            return None, 0.0

        candidates.sort(key=lambda x: (x[0], len(x[1])), reverse=True)

        best_score, best_value = candidates[0]
        if not has_reference_anchor and best_score < 0.7:
            return None, 0.0

        return best_value, min(0.99, best_score)

    def _extract_account(self, lines: List[Dict], template_name: Optional[str]) -> Tuple[Optional[str], float]:
        ordered = self._sorted_lines(lines)

        if self._is_shopeepay_layout(ordered) or template_name == "shopeepay":
            # ShopeePay varian ringkas/top-up tidak selalu menampilkan rekening
            # tujuan. Jika konteks penerima tidak ada, paksa null.
            if not self._has_shopeepay_receiver_context(ordered):
                return None, 0.99

            for idx, line in enumerate(ordered):
                raw = str(line.get("text", ""))
                norm = normalize_text(raw)
                compact = re.sub(r"[^a-z0-9]", "", norm)
                is_receiver_anchor = (
                    bool(re.search(r"(^|\s)kirim\s+ke(\s|$)", norm))
                    or bool(re.search(r"(^|\s)transfer\s+ke(\s|$)", norm))
                    or bool(re.search(r"(?<!di)kirimke", compact))
                    or bool(re.search(r"(?<!di)transferke", compact))
                )
                if not is_receiver_anchor:
                    continue

                for j in range(idx, min(idx + 3, len(ordered))):
                    cand = str(ordered[j].get("text", ""))
                    if not re.search(r"(?i)(bca|bri|bni|mandiri|seabank|cimb|bsi)\s*[:\-]?\*?\d{6,16}", cand):
                        continue
                    digits = normalize_number(cand)
                    if 8 <= len(digits) <= 16:
                        return digits, 0.9

        if self._is_gopay_layout(ordered) or template_name == "gopay":
            # GoPay receipt umumnya tidak memuat rekening penerima eksplisit.
            # Hindari over-extract dari ID transaksi.
            strong_account_anchors = (
                "rekening penerima",
                "nomor rekening",
                "no rekening",
                "account number",
                "destination account",
                "rekening tujuan",
            )
            for line in ordered:
                raw = str(line.get("text", ""))
                norm = normalize_text(raw)
                if not any(anchor in norm for anchor in strong_account_anchors):
                    continue
                digits = normalize_number(raw)
                if 8 <= len(digits) <= 16:
                    return digits, 0.9

            # Sebagian receipt GoPay menampilkan rekening tujuan dalam satu baris
            # seperti "BCA7425252836" tanpa label rekening.
            has_transfer_context = any(
                any(k in normalize_text(str(l.get("text", ""))) for k in ("transferke", "ditransferke", "kirimke", "kepenerima"))
                for l in ordered
            )
            if has_transfer_context:
                for line in ordered:
                    raw = str(line.get("text", ""))
                    if not re.search(r"(?i)(bca|bri|bni|mandiri|seabank|cimb|bsi)\s*[:\-]?\*?\d{6,16}", raw):
                        continue
                    digits = normalize_number(raw)
                    if 8 <= len(digits) <= 16:
                        return digits, 0.88
            return None, 0.99

        if self._is_blu_layout(ordered) or template_name == "blu_bca":
            for line in ordered:
                raw = str(line.get("text", ""))
                norm = normalize_text(raw)
                if "bca" not in norm or "-" not in raw:
                    continue

                cleaned = raw
                if "-" in cleaned:
                    cleaned = cleaned.split("-", 1)[1]
                trans = str.maketrans({"O": "0", "o": "0", "I": "1", "l": "1", "S": "5", "s": "5"})
                digits = re.sub(r"\D", "", cleaned.translate(trans))
                if 8 <= len(digits) <= 16:
                    return digits, 0.96

        # DANA account biasanya masked tail saja. Jika hanya 3-6 digit, return None.
        if self._is_dana_layout(ordered) or template_name == "dana":
            for i, line in enumerate(ordered):
                norm = normalize_text(line.get("text", ""))
                if ("akun bank" not in norm) and ("akun dana" not in norm):
                    continue

                block = [line]
                if i + 1 < len(ordered):
                    block.append(ordered[i + 1])

                for candidate_line in block:
                    digits = normalize_number(candidate_line.get("text", ""))
                    if 8 <= len(digits) <= 16:
                        return digits, 0.86

            # Fallback untuk layout transfer bank yang sempat ter-route ke
            # template dana karena token "dana" generik (mis. "Sumber Dana").
            # Ambil rekening dari anchor eksplisit penerima agar tidak null.
            strong_receiver_anchors = (
                "rekening penerima",
                "rekening tujuan",
                "nomor rekening",
                "no rekening",
                "account number",
                "destination account",
            )
            stop_hints = (
                "id transaksi",
                "no referensi",
                "nomor referensi",
                "no resi",
                "nomor resi",
                "tujuan transfer",
                "metode transfer",
                "catatan",
            )
            for i, line in enumerate(ordered):
                raw = str(line.get("text", ""))
                norm = normalize_text(raw)
                compact = re.sub(r"[^a-z0-9]", "", norm)
                has_receiver_anchor = (
                    any(anchor in norm for anchor in strong_receiver_anchors)
                    or ("rekeningpenerima" in compact)
                    or ("rekeningtujuan" in compact)
                    or ("nomorrekening" in compact)
                    or ("norekening" in compact)
                )
                if not has_receiver_anchor:
                    continue

                inline_digits = normalize_number(
                    re.sub(
                        r"(?i)^.*?(rekening\s*penerima|rekening\s*tujuan|nomor\s*rekening|no\.?\s*rekening|account\s*number|destination\s*account)\s*[:\-]?\s*",
                        "",
                        raw,
                    )
                )
                if 8 <= len(inline_digits) <= 16:
                    return inline_digits, 0.93

                for j in range(i + 1, min(i + 4, len(ordered))):
                    nxt = str(ordered[j].get("text", ""))
                    nxt_norm = normalize_text(nxt)
                    if any(h in nxt_norm for h in stop_hints):
                        break

                    digits = normalize_number(nxt)
                    if 8 <= len(digits) <= 16:
                        return digits, 0.92

            return None, 0.0

        reference_anchor_indexes = []
        for idx, line in enumerate(ordered):
            if any(h in normalize_text(line.get("text", "")) for h in REFERENCE_ANCHOR_HINTS):
                reference_anchor_indexes.append(idx)

        candidates: List[Tuple[float, str]] = []

        for idx, line in enumerate(ordered):
            raw = str(line.get("text", ""))
            norm = normalize_text(raw)
            if not raw:
                continue

            if re.search(r"\b(rp|idr)\b", norm):
                continue
            if is_datetime_like(raw):
                continue
            if any(h in norm for h in ("biaya", "fee", "nominal", "total", "reference", "referensi", "tanggal", "waktu")):
                continue

            digits = normalize_number(raw)
            if not (8 <= len(digits) <= 16):
                continue

            # Token date compact seperti 020420260809 tidak boleh jadi rekening.
            if re.fullmatch(r"20\d{2}\d{2}\d{2}\d{2,4}", digits):
                continue
            if re.fullmatch(r"\d{2}20\d{2}\d{2}\d{2}\d{2,4}", digits):
                continue

            score = 0.62

            if any(anchor in norm for anchor in ("rekening", "account", "destination account", "ke", "tujuan", "bank")):
                score += 0.24

            if any(anchor in norm for anchor in ("dari", "from", "sumber dana", "pengirim")):
                score -= 0.33

            if reference_anchor_indexes and any(abs(idx - ridx) <= 1 for ridx in reference_anchor_indexes):
                score -= 0.22

            if digits in self.known_accounts:
                score += 0.25

            candidates.append((score, digits))

        if not candidates:
            return None, 0.0

        candidates.sort(key=lambda x: (x[0], len(x[1])), reverse=True)
        return candidates[0][1], min(0.98, candidates[0][0])

    def _extract_name_inline(self, line_text: str) -> Optional[str]:
        patterns = [
            r"(?i)\btransfer\s+ke\s+([A-Za-z][A-Za-z .'-]{2,60})",
            r"(?i)\bkirim\s+uang.*?\bke\s+([A-Za-z][A-Za-z .'-]{2,60})",
            r"(?i)\bkirimuang.*?ke([A-Za-z]{5,30})",
            r"(?i)\bke\s*[:\-]\s*([A-Za-z][A-Za-z .'-]{2,60})",
            r"(?i)\btransfer\s+to\s+([A-Za-z][A-Za-z .'-]{2,60})",
            r"(?i)\bto\s*[:\-]\s*([A-Za-z][A-Za-z .'-]{2,60})",
        ]

        for pattern in patterns:
            match = re.search(pattern, str(line_text))
            if not match:
                continue
            value = clean_name_text(match.group(1))
            if is_human_name_candidate(value):
                return value

        return None

    def _extract_name(self, lines: List[Dict], account_no: Optional[str], template_name: Optional[str]) -> Tuple[Optional[str], float]:
        ordered = self._sorted_lines(lines)
        account_line_indexes = [
            idx for idx, line in enumerate(ordered)
            if 8 <= len(normalize_number(line.get("text", ""))) <= 16
        ]

        if self._is_dana_layout(ordered) or template_name == "dana":
            for idx, line in enumerate(ordered):
                norm = normalize_text(line.get("text", ""))
                compact = re.sub(r"[^a-z0-9]", "", norm)
                if ("detail penerima" not in norm) and ("detailpenerima" not in compact):
                    continue

                for j in range(idx, min(idx + 8, len(ordered) - 1)):
                    label_norm = normalize_text(ordered[j].get("text", ""))
                    label_compact = re.sub(r"[^a-z]", "", label_norm)
                    if label_compact not in ("nama", "namapenerima", "penerima", "recipient"):
                        continue

                    raw_name = str(ordered[j + 1].get("text", ""))
                    masked_lex = self._resolve_masked_name_with_lexicon(raw_name)
                    if masked_lex:
                        return self._normalize_recipient_name_case(masked_lex), 0.97

                    candidate = clean_name_text(raw_name)
                    if is_human_name_candidate(candidate):
                        return self._normalize_recipient_name_case(candidate), 0.93

        if self._is_shopeepay_layout(ordered) or template_name == "shopeepay":
            # Pada ShopeePay minim detail (tanpa konteks "Kirim Ke"), nama
            # penerima tidak tersedia sehingga harus null.
            if not self._has_shopeepay_receiver_context(ordered):
                return None, 0.99

            for idx, line in enumerate(ordered):
                raw = str(line.get("text", ""))
                norm = normalize_text(raw)
                compact = re.sub(r"[^a-z0-9]", "", norm)
                is_receiver_anchor = (
                    bool(re.search(r"(^|\s)kirim\s+ke(\s|$)", norm))
                    or bool(re.search(r"(^|\s)transfer\s+ke(\s|$)", norm))
                    or bool(re.search(r"(?<!di)kirimke", compact))
                    or bool(re.search(r"(?<!di)transferke", compact))
                )
                if not is_receiver_anchor:
                    continue

                inline = self._extract_name_inline(raw)
                if inline and is_human_name_candidate(inline):
                    return self._normalize_recipient_name_case(inline), 0.9

                merged = re.search(
                    r"(?i)(?:kirim|transfer)\s*ke\s*(?:[A-Za-z]{2,12}\s*[:\-]?\s*)?([A-Za-z][A-Za-z .'-]{4,60})",
                    raw,
                )
                if merged:
                    candidate = clean_name_text(merged.group(1))
                    if is_human_name_candidate(candidate):
                        return self._normalize_recipient_name_case(candidate), 0.9

                for j in range(idx + 1, min(idx + 3, len(ordered))):
                    candidate = clean_name_text(ordered[j].get("text", ""))
                    if is_human_name_candidate(candidate):
                        return self._normalize_recipient_name_case(candidate), 0.88

        if self._is_gopay_layout(ordered) or template_name == "gopay":
            # Untuk GoPay, ambil recipient hanya jika ada anchor penerima yang jelas.
            strict_name_anchors = (
                "nama penerima",
                "recipient",
                "beneficiary",
                "transfer ke",
                "kirim ke",
                "tujuan transfer",
                "ditransferke",
                "transferke",
                "kirimke",
            )
            for idx, line in enumerate(ordered):
                raw = str(line.get("text", ""))
                norm = normalize_text(raw)
                if not any(anchor in norm for anchor in strict_name_anchors):
                    continue

                compact_raw = re.sub(r"\s+", "", raw)
                merged = re.search(r"(?i)(?:di)?transferke([A-Za-z][A-Za-z .'-]{3,60})", compact_raw)
                if merged:
                    candidate = clean_name_text(merged.group(1))
                    if is_human_name_candidate(candidate):
                        return self._normalize_recipient_name_case(candidate), 0.9

                inline = self._extract_name_inline(raw)
                if inline and is_human_name_candidate(inline):
                    return self._normalize_recipient_name_case(inline), 0.9

                for j in range(idx + 1, min(idx + 3, len(ordered))):
                    candidate = clean_name_text(ordered[j].get("text", ""))
                    if is_human_name_candidate(candidate):
                        return self._normalize_recipient_name_case(candidate), 0.88

            return None, 0.99

        # blu layout: name biasanya di atas baris "BCA - xxxxx".
        if self._is_blu_layout(ordered) or template_name == "blu_bca":
            account_lines = [
                l for l in ordered
                if "bca" in normalize_text(l.get("text", "")) and "-" in str(l.get("text", ""))
            ]
            if account_lines:
                best = None
                for candidate in ordered:
                    name = clean_name_text(candidate.get("text", ""))
                    if not is_human_name_candidate(name):
                        continue
                    for row in account_lines:
                        dy = float(row.get("cy", 0.0)) - float(candidate.get("cy", 0.0))
                        dx = abs(float(candidate.get("cx", 0.0)) - float(row.get("cx", 0.0)))
                        if 0 <= dy <= 90 and dx <= 240:
                            score = 0.88 + (0.1 if name.isupper() else 0.0)
                            if best is None or score > best[0]:
                                best = (score, name)
                if best:
                    return self._normalize_recipient_name_case(best[1]), best[0]

        candidates: List[Tuple[float, str]] = []

        for idx, line in enumerate(ordered):
            text = str(line.get("text", ""))
            norm = normalize_text(text)
            has_positive_anchor = self._has_positive_name_anchor(text)

            inline = self._extract_name_inline(text)
            if inline:
                candidates.append((0.94, inline))

            if has_positive_anchor:
                for j in range(idx, min(idx + 3, len(ordered))):
                    candidate = clean_name_text(ordered[j].get("text", ""))
                    if not is_human_name_candidate(candidate):
                        continue

                    score = 0.84
                    if ordered[j] is not line:
                        score += 0.04
                    if any(n in normalize_text(ordered[j].get("text", "")) for n in NEGATIVE_NAME_ANCHORS):
                        score -= 0.42
                    candidates.append((score, candidate))

            plain = clean_name_text(text)
            if is_human_name_candidate(plain):
                score = 0.66
                if any(n in norm for n in NEGATIVE_NAME_ANCHORS):
                    score -= 0.5
                if any(n in norm for n in ("metode", "pembayaran", "saldo", "detail transaksi", "id transaksi")):
                    score -= 0.55
                if has_positive_anchor:
                    score += 0.2
                candidates.append((score, plain))

            for lex_name in self._extract_name_lexicon_candidates_from_line(text):
                score = 0.84
                if has_positive_anchor:
                    score += 0.06
                if any(abs(idx - account_idx) <= 1 for account_idx in account_line_indexes):
                    score += 0.08
                if idx > 0:
                    prev_norm = self._normalize_hint_text(ordered[idx - 1].get("text", ""))
                    if ("tujuan" in prev_norm) or ("penerima" in prev_norm) or re.search(r"(^|\s)ke(\s|$)", prev_norm):
                        score += 0.04
                candidates.append((min(0.97, score), lex_name))

            # Name menempel tanpa spasi: FADHILBAWAZIER
            compact = re.sub(r"[^A-Za-z]", "", text)
            if re.fullmatch(r"[A-Za-z]{8,30}", compact):
                lex = self._match_name_lexicon(compact)
                if lex:
                    candidates.append((0.9, lex))

        if account_no and account_no in self.account_recipient_map:
            mapped = self.account_recipient_map[account_no]
            candidates.append((0.92, mapped))

        if not candidates:
            return None, 0.0

        ranked = []
        for score, value in candidates:
            normalized = self._normalize_recipient_name_case(value)
            if not normalized:
                continue
            ranked.append((score, normalized))

        if not ranked:
            return None, 0.0

        ranked.sort(key=lambda x: (x[0], len(x[1])), reverse=True)
        return ranked[0][1], min(0.99, ranked[0][0])

    def _parse_superbank_header_amount_text(self, text: str) -> Optional[int]:
        raw = str(text)
        if not re.search(r"[\dOoQqDdIlLSsBbZz]", raw):
            return None

        normalized = raw.translate(
            str.maketrans({
                "O": "0",
                "o": "0",
                "Q": "0",
                "q": "0",
                "D": "0",
                "I": "1",
                "l": "1",
                "L": "1",
                "S": "5",
                "s": "5",
                "B": "8",
                "Z": "2",
                "z": "2",
            })
        )
        normalized = re.sub(r"(?i)\b(rp|idr)\s+(?=\d)", r"\1", normalized)
        normalized = re.sub(r"(?<=\d)\s+(?=[\d.,])", "", normalized)
        normalized = re.sub(r"(?<=[\d.,])\s+(?=\d)", "", normalized)

        amount_texts: List[str] = []
        for match in re.finditer(r"(?i)(?:rp|idr)\s*[:.]?\s*([0-9][0-9.,]{1,})", normalized):
            amount_texts.append(match.group(1))

        for match in re.finditer(r"(?<!\d)(\d{1,3}(?:[.,]\d{3})+(?:[.,]\d{2})?)(?!\d)", normalized):
            amount_texts.append(match.group(1))

        for amount_text in amount_texts:
            value = parse_rupiah_amount_ocr_aware(amount_text)
            if value is not None and MIN_AMOUNT <= value <= MAX_AMOUNT:
                return value

        return None

    def _extract_superbank_header_amount(self, ordered: List[Dict]) -> Optional[int]:
        status_indexes = []
        for idx, line in enumerate(ordered):
            if self._is_success_status_text(line.get("text", "")):
                status_indexes.append(idx)

        has_status_date_header = False
        for idx in status_indexes:
            for j in range(idx + 1, min(idx + 5, len(ordered))):
                text = str(ordered[j].get("text", ""))
                hour, minute = _extract_time_component(text)
                if _has_date_context(text) and (hour != 0 or minute != 0):
                    has_status_date_header = True
                    break
            if has_status_date_header:
                break

        has_superbank_context = self._is_superbank_header_transfer_layout(ordered)
        if not has_superbank_context and not has_status_date_header:
            return None

        candidate_indexes = []
        for idx in status_indexes:
            candidate_indexes.extend(range(idx, min(idx + 4, len(ordered))))

        if not candidate_indexes and has_superbank_context:
            max_y = max((float(line.get("bbox", [0, 0, 0, line.get("cy", 0.0)])[3]) for line in ordered), default=0.0)
            top_limit = max_y * 0.35 if max_y > 0 else 0.0
            candidate_indexes = [
                idx for idx, line in enumerate(ordered)
                if top_limit <= 0.0 or float(line.get("cy", 0.0)) <= top_limit
            ]

        seen = set()
        for idx in candidate_indexes:
            if idx in seen or idx >= len(ordered):
                continue
            seen.add(idx)
            value = self._parse_superbank_header_amount_text(ordered[idx].get("text", ""))
            if value is not None:
                return value

        return None

    def _extract_header_amount_near_transaction_date(self, ordered: List[Dict]) -> Optional[int]:
        max_y = max((float(line.get("bbox", [0, 0, 0, line.get("cy", 0.0)])[3]) for line in ordered), default=0.0)
        top_limit = max_y * 0.45 if max_y > 0 else 0.0

        for idx, line in enumerate(ordered):
            text = str(line.get("text", ""))
            hour, minute = _extract_time_component(text)
            if not (_has_date_context(text) and (hour != 0 or minute != 0)):
                continue
            if top_limit > 0.0 and float(line.get("cy", 0.0)) > top_limit:
                continue

            for j in range(max(0, idx - 3), idx + 1):
                candidate_text = str(ordered[j].get("text", ""))
                candidate_norm = normalize_text(candidate_text)
                if any(h in candidate_norm for h in AMOUNT_NEGATIVE_HINTS):
                    continue
                value = self._parse_superbank_header_amount_text(candidate_text)
                if value is not None:
                    return value

        return None

    def _extract_amount(self, lines: List[Dict], template_name: Optional[str]) -> Tuple[Optional[int], float]:
        ordered = self._sorted_lines(lines)

        entries = []

        for line in ordered:
            text = str(line.get("text", ""))
            norm = normalize_text(text)

            value = parse_rupiah_amount_ocr_aware(text)
            if value is None or not is_amount_candidate(text):
                continue

            score = 0.52
            is_total_anchor = self._contains_hint(
                text,
                ("total transaksi", "total debit", "debit amount", "total nominal", "jumlah total", "total"),
            )
            is_nominal_anchor = self._contains_hint(text, AMOUNT_PRIORITY_ANCHORS) or self._contains_hint(text, ("kirim uang",))
            if (not is_nominal_anchor) and ("jumlah" in norm) and ("total" not in norm) and ("biaya" not in norm):
                is_nominal_anchor = True
            if (not is_nominal_anchor) and ("nominal" in norm) and (not is_total_anchor) and ("biaya" not in norm):
                is_nominal_anchor = True
            is_fee_anchor = self._contains_hint(text, AMOUNT_NEGATIVE_HINTS)
            fee_is_waived = bool(re.search(r"\b(gratis|free)\b", norm))

            if is_nominal_anchor:
                score += 0.52

            if is_total_anchor:
                score += 0.16

            if is_fee_anchor:
                score -= 0.95

            if "total transaksi" in norm and any("nominal transfer" in normalize_text(l.get("text", "")) for l in ordered):
                score -= 0.25

            if "rp" in norm or "idr" in norm:
                score += 0.2

            if re.search(r"\d{1,3}([.,]\d{3})+", text):
                score += 0.12

            if template_name == "livin_mandiri" and ("nominal transfer" in norm or "jumlah transfer" in norm):
                score += 0.28

            if self._is_dana_layout(ordered) and "total bayar" in norm:
                score += 0.22

            entries.append({
                "score": score,
                "value": value,
                "is_nominal": is_nominal_anchor and not is_fee_anchor,
                "is_total": is_total_anchor and not is_fee_anchor,
                "is_fee": is_fee_anchor,
                "fee_value": 0 if fee_is_waived else value,
            })

        if not entries:
            superbank_amount = self._extract_superbank_header_amount(ordered)
            if superbank_amount is not None:
                return superbank_amount, 0.93
            header_amount = self._extract_header_amount_near_transaction_date(ordered)
            if header_amount is not None:
                return header_amount, 0.86
            return None, 0.0

        nominal_candidates = [e for e in entries if e["is_nominal"]]
        total_candidates = [e for e in entries if e["is_total"]]
        fee_candidates = [e for e in entries if e["is_fee"]]
        non_fee_candidates = [e for e in entries if not e["is_fee"]]

        if nominal_candidates:
            nominal_candidates.sort(key=lambda x: (x["score"], x["value"]), reverse=True)
            best = nominal_candidates[0]
            return best["value"], min(0.99, best["score"])

        if total_candidates and fee_candidates:
            total_candidates.sort(key=lambda x: (x["score"], x["value"]), reverse=True)
            fee_candidates.sort(key=lambda x: (x["score"], x["value"]), reverse=True)
            top_total = total_candidates[0]
            top_fee = fee_candidates[0]

            derived_nominal = int(top_total["value"]) - int(top_fee.get("fee_value", top_fee["value"]))
            if MIN_AMOUNT <= derived_nominal <= MAX_AMOUNT and derived_nominal < int(top_total["value"]):
                derived_score = min(0.95, max(0.74, top_total["score"] + 0.06))
                return derived_nominal, derived_score

        # Jika OCR anchor total/debit sedikit noisy, gunakan relasi aritmetika
        # antar kandidat non-fee: total ~= nominal + fee.
        if fee_candidates and len(non_fee_candidates) >= 2:
            fee_candidates.sort(key=lambda x: (x["score"], x.get("fee_value", x["value"])), reverse=True)
            fee_value = int(fee_candidates[0].get("fee_value", fee_candidates[0]["value"]))
            sorted_non_fee = sorted(non_fee_candidates, key=lambda x: (x["score"], x["value"]), reverse=True)

            for high in sorted_non_fee:
                for low in sorted_non_fee:
                    if high is low:
                        continue
                    high_value = int(high["value"])
                    low_value = int(low["value"])
                    if high_value <= low_value:
                        continue
                    diff = high_value - low_value
                    if abs(diff - fee_value) <= 250:
                        combo_score = min(0.95, max(0.74, (high["score"] + low["score"]) / 2 + 0.04))
                        return low_value, combo_score

        if non_fee_candidates:
            non_fee_candidates.sort(key=lambda x: (x["score"], x["value"]), reverse=True)
            best = non_fee_candidates[0]
            return best["value"], min(0.99, best["score"])

        # Jika hanya ada biaya admin/fee, lebih aman return None agar
        # tidak mengembalikan total_amount yang salah.
        return None, 0.0

    def _has_transaction_date_signal(self, lines: List[Dict]) -> bool:
        norms = [normalize_text(l.get("text", "")) for l in lines]
        hints = (
            "waktu transaksi",
            "tanggal transaksi",
            "transaction time",
            "transaction date",
            "transaksi berhasil",
            "transfer berhasil",
            "status transaksi",
            "biz id",
            "no referensi",
            "nomor referensi",
            "no resi",
            "nomor resi",
            "m-transfer",
        )
        if any(any(h in n for h in hints) for n in norms):
            return True

        if any(re.search(r"\b(wib|wit|wita)\b", n) for n in norms):
            return True

        return False

    def _is_instruction_only_context(self, lines: List[Dict]) -> bool:
        norms = [normalize_text(l.get("text", "")) for l in lines]
        has_instruction = any(any(h in n for h in INSTRUCTION_DATE_HINTS) for n in norms)
        if not has_instruction:
            return False
        return not self._has_transaction_date_signal(lines)

    def _override_time_from_lines(self, parsed_date: Optional[str], lines: List[Dict]) -> Optional[str]:
        if not parsed_date:
            return parsed_date

        m = re.match(r"^(\d{4}-\d{2}-\d{2})\s+(\d{2}):(\d{2})$", parsed_date)
        if not m:
            return parsed_date

        date_part = m.group(1)
        current_hour = int(m.group(2))
        current_minute = int(m.group(3))

        date_line_indexes = [
            idx
            for idx, line in enumerate(lines)
            if _has_date_context(str(line.get("text", "")))
        ]

        candidates = []
        for idx, line in enumerate(lines):
            text = str(line.get("text", ""))
            norm = normalize_text(text)
            if _is_status_bar_time_line(text, idx):
                continue

            hour, minute = _extract_time_component(norm)
            if hour == 0 and minute == 0:
                continue

            score = 0.25
            has_date_pattern = _has_date_context(norm)
            if has_date_pattern:
                score += 0.65

            if any(h in norm for h in DATE_LABEL_HINTS):
                score += 0.45

            if (not has_date_pattern) and date_line_indexes and min(abs(idx - d) for d in date_line_indexes) <= 2:
                score += 0.25

            if re.search(r"\b(wib|wit|wita)\b", norm):
                score += 0.2

            # Header/status-bar time di baris paling atas sering bukan waktu transaksi.
            if idx <= 1 and not has_date_pattern and not any(h in norm for h in DATE_LABEL_HINTS):
                score -= 0.8

            # Jika parsed awal 00:00, prioritas non-zero time dari konteks kuat.
            if current_hour == 0 and current_minute == 0:
                score += 0.2

            candidates.append((score, hour, minute))

        if not candidates:
            return parsed_date

        candidates.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
        best_score, best_hour, best_minute = candidates[0]

        # Jika parsed awal sudah punya jam valid, jangan override kecuali
        # kandidat pengganti sangat kuat (biasanya baris berlabel + tanggal).
        min_override_score = 0.5 if (current_hour == 0 and current_minute == 0) else 0.85
        if best_score < min_override_score:
            return parsed_date

        return f"{date_part} {best_hour:02d}:{best_minute:02d}"

    def _extract_transaction_date(self, lines: List[Dict], template_name: Optional[str]) -> Tuple[Optional[str], float]:
        ordered = self._sorted_lines(lines)

        if self._is_instruction_only_context(ordered):
            return None, 0.0

        # DANA OCR typo time correction.
        if self._is_dana_layout(ordered) or template_name == "dana":
            sources = [str(l.get("text", "")) for l in ordered]
            for source in sources:
                norm = normalize_text(source)
                norm = re.sub(r"\b(wib|wit|wita)\b", " ", norm)
                norm = norm.replace("|", "1")
                norm = re.sub(r"\boi(?=\s*[a-z]{3,10}\s*20\d{2})", "01", norm)
                # Koreksi OCR DANA: "I7:35" -> "17:35", "I:06"/"I-06" -> "11:06".
                norm = re.sub(r"\b[il][t7][:.]([0-5]\d)\b", r"17:\1", norm)
                norm = re.sub(r"\b[il][-.]([0-5]\d)\b", r"11:\1", norm)
                parsed = parse_noisy_transaction_date(norm)
                if parsed:
                    return self._override_time_from_lines(parsed, ordered), 0.94

        # Anchor-based compose date + time.
        date_lines: List[Tuple[int, str]] = []
        time_lines: List[Tuple[int, str]] = []

        for idx, line in enumerate(ordered):
            text = str(line.get("text", ""))
            norm = normalize_text(text)

            has_date = _has_date_context(norm)
            has_time = bool(re.search(r"\d{1,2}[:.]\d{2}", norm) or re.search(r"\d{1,2}\s+\d{2}\s+\d{2}", norm))

            if has_date:
                date_lines.append((idx, text))
            if has_time and not _is_status_bar_time_line(text, idx):
                time_lines.append((idx, text))

            if has_date and has_time:
                parsed = parse_noisy_transaction_date(text)
                if parsed:
                    return self._override_time_from_lines(parsed, ordered), 0.91
                parsed = safe_parse_date(text)
                if parsed:
                    return self._override_time_from_lines(parsed, ordered), 0.88

        if date_lines and time_lines:
            composed_candidates: List[Tuple[float, str]] = []
            for d_idx, d_text in date_lines:
                for t_idx, t_text in time_lines:
                    parsed = parse_noisy_transaction_date(f"{d_text} {t_text}")
                    if parsed:
                        pair_score = 0.86
                        if abs(d_idx - t_idx) <= 2:
                            pair_score += 0.08
                        if any(h in normalize_text(t_text) for h in DATE_LABEL_HINTS):
                            pair_score += 0.06
                        composed_candidates.append((min(0.96, pair_score), parsed))
            if composed_candidates:
                composed_candidates.sort(key=lambda x: x[0], reverse=True)
                best_score, best_parsed = composed_candidates[0]
                return self._override_time_from_lines(best_parsed, ordered), best_score

        full_text = " ".join(str(l.get("text", "")) for l in ordered)
        parsed = parse_noisy_transaction_date(full_text)
        if parsed:
            return self._override_time_from_lines(parsed, ordered), 0.8

        parsed = safe_parse_date(full_text)
        if parsed:
            return self._override_time_from_lines(parsed, ordered), 0.74

        return None, 0.0

    def extract(self, lines: List[Dict], template_name: Optional[str] = None) -> Dict[str, Dict]:
        reference_no, ref_score = self._extract_reference(lines, template_name)
        transaction_date, date_score = self._extract_transaction_date(lines, template_name)
        account_no, account_score = self._extract_account(lines, template_name)
        recipient_name, name_score = self._extract_name(lines, account_no, template_name)
        total_amount, amount_score = self._extract_amount(lines, template_name)

        reference_no = self.resolve_reference_with_lexicon(reference_no)

        # Infer rekening dari nama jika rekening tidak terbaca tapi nama kuat.
        if (not account_no or account_score < 0.72) and recipient_name:
            recipient_key = re.sub(r"[^a-z]", "", recipient_name.lower())
            mapped_account = self.recipient_account_map.get(recipient_key)
            if mapped_account:
                account_no = mapped_account
                account_score = max(account_score, 0.84)

        # Recipient override dari map account->name jika nama lemah/noisy.
        if account_no and account_no in self.account_recipient_map:
            mapped = self.account_recipient_map[account_no]
            should_override = (not recipient_name) or (name_score < 0.78)
            if not should_override and recipient_name:
                left = re.sub(r"[^a-z]", "", recipient_name.lower())
                right = re.sub(r"[^a-z]", "", mapped.lower())
                if left and right:
                    similarity = SequenceMatcher(None, left, right).ratio()
                    if similarity < 0.82:
                        should_override = True
            if should_override:
                recipient_name = mapped
                name_score = max(name_score, 0.9)

        return {
            "reference_no": {"value": reference_no, "confidence": ref_score, "source": "rules_v1"},
            "transaction_date": {"value": transaction_date, "confidence": date_score, "source": "rules_v1"},
            "account_no": {"value": account_no, "confidence": account_score, "source": "rules_v1"},
            "recipient_name": {"value": recipient_name, "confidence": name_score, "source": "rules_v1"},
            "total_amount": {"value": total_amount, "confidence": amount_score, "source": "rules_v1"},
        }

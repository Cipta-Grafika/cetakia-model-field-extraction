try:
    from .layout_parser import normalize_text
except ImportError:
    # Mode script langsung: `python src_v2/template_router.py`.
    from layout_parser import normalize_text


TEMPLATE_RULES = {
    # Rule dipisah antara brand anchor (wajib) dan context anchor (pendukung).
    "bca": {
        "brand": ["bca", "m-transfer"],
        "context": ["bukti transfer", "no. ref", "ke", "berhasil"],
    },
    "bni": {
        "brand": ["bni"],
        "context": ["biz id", "bukti transaksi", "reference"],
    },
    "livin_mandiri": {
        "brand": ["livin", "mandiri"],
        "context": ["nominal transfer", "biaya transaksi", "detail transaksi"],
    },
    "byond_bsi": {
        "brand": ["byond", "bsi"],
        "context": ["nomor transaksi", "terminal", "detail transaksi"],
    },
    "seabank": {
        "brand": ["seabank"],
        "context": ["ke", "dari", "waktu transaksi", "waktu pemrosesan"],
    },
    "blu_bca": {
        "brand": ["blu"],
        "context": ["ref blu", "no ref blu", "transfer berhasil", "bca"],
    },
    "btn": {
        "brand": ["btn"],
        "context": ["id transaksi", "no. reff", "pembayaran ke", "jumlahtransaksi", "jenis transaksi"],
    },
    "dana": {
        "brand": ["dana"],
        "context": ["id transaksi", "total bayar", "akun bank"],
    },
    "shopeepay": {
        "brand": ["shopeepay"],
        "context": ["order sn", "rincian referensi", "waktu selesai", "kirim ke"],
    },
    "gopay": {
        "brand": ["gopay"],
        "context": ["rincian transaksi", "id transaksi", "order id", "gopay topup", "gopay top up"],
    },
}


DEFAULT_TEMPLATE_ROIS = {
    "bca": {
        "reference_no": {"x1": 0.05, "y1": 0.55, "x2": 0.95, "y2": 0.80},
        "transaction_date": {"x1": 0.05, "y1": 0.35, "x2": 0.95, "y2": 0.55},
        "account_no": {"x1": 0.05, "y1": 0.25, "x2": 0.95, "y2": 0.55},
        "recipient_name": {"x1": 0.05, "y1": 0.20, "x2": 0.95, "y2": 0.50},
        "total_amount": {"x1": 0.05, "y1": 0.10, "x2": 0.95, "y2": 0.40},
    },
    "livin_mandiri": {
        "reference_no": {"x1": 0.05, "y1": 0.65, "x2": 0.95, "y2": 0.95},
        "transaction_date": {"x1": 0.05, "y1": 0.10, "x2": 0.95, "y2": 0.35},
        "account_no": {"x1": 0.05, "y1": 0.30, "x2": 0.95, "y2": 0.65},
        "recipient_name": {"x1": 0.05, "y1": 0.25, "x2": 0.95, "y2": 0.55},
        "total_amount": {"x1": 0.05, "y1": 0.45, "x2": 0.95, "y2": 0.75},
    },
}


class TemplateRouter:
    """
    Router template berbasis anchor text.
    Ringan, explainable, dan mudah di-debug.
    """

    def __init__(self):
        self.template_rules = TEMPLATE_RULES
        self.template_rois = DEFAULT_TEMPLATE_ROIS

    def detect_template(self, page_text: str):
        """
        Deteksi template berdasarkan jumlah anchor yang match.

        Returns:
            {
                "template": str|None,
                "score": float,
                "matched_anchors": list
            }
        """
        text = normalize_text(page_text)

        best_template = None
        best_score = 0.0
        best_matches = []

        for template_name, rule in self.template_rules.items():
            brand_anchors = [normalize_text(a) for a in rule.get("brand", [])]
            ctx_anchors = [normalize_text(a) for a in rule.get("context", [])]

            brand_matches = [a for a in brand_anchors if a in text]
            ctx_matches = [a for a in ctx_anchors if a in text]

            # Brand anchor harus muncul agar template tidak salah route karena
            # kata generik seperti "ke/dari/berhasil".
            if not brand_matches:
                continue

            # Brand diberi bobot lebih besar daripada context.
            brand_score = len(brand_matches) / max(len(brand_anchors), 1)
            ctx_score = len(ctx_matches) / max(len(ctx_anchors), 1)
            score = (0.7 * brand_score) + (0.3 * ctx_score)

            # "dana" sebagai token tunggal rawan false-positive pada receipt lain
            # (contoh "Sumber Dana" di ShopeePay), sehingga perlu context pendukung.
            if template_name == "dana" and not ctx_matches:
                score *= 0.35

            if score > best_score:
                best_template = template_name
                best_score = score
                best_matches = brand_matches + ctx_matches

        if best_score < 0.35:
            best_template = None

        return {
            "template": best_template,
            "score": float(best_score),
            "matched_anchors": best_matches,
        }

    def get_rois(self, template_name: str):
        """
        Mengambil ROI field untuk template tertentu.
        """
        if not template_name:
            return None

        return self.template_rois.get(template_name)

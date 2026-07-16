"""Deterministic regex+validator detection layer.

Every pattern here is paired with a validator so plain-looking digit runs
don't get flagged just because they're the right length — SSN/EIN/ITIN area
rules, Luhn for cards, the ABA 3-7-1 checksum for routing numbers, and
"needs a nearby context word" for the inherently ambiguous bare 9-17 digit
forms (routing/account/bare-SSN/bare-EIN all look alike as raw digits).

All spans from this module carry confidence=1.0 and source="regex" per the
Detector contract in scrub.types. Overlap handling within this detector:
longest match wins, ties broken by rule registration order, so the same
character range is never reported twice.
"""

from __future__ import annotations

import ipaddress
import math
import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass

from ..types import EntityType, Span

_CONTEXT_WINDOW = 40  # chars scanned each side of a bare digit match


# --------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------


def _context_has(text: str, start: int, end: int, keywords: tuple[str, ...]) -> bool:
    lo = max(0, start - _CONTEXT_WINDOW)
    hi = min(len(text), end + _CONTEXT_WINDOW)
    window = text[lo:hi].lower()
    return any(kw in window for kw in keywords)


def _digits_only(s: str) -> str:
    return "".join(c for c in s if c.isdigit())


def _shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    counts = Counter(s)
    length = len(s)
    return -sum((n / length) * math.log2(n / length) for n in counts.values())


def _luhn_valid(digits: str) -> bool:
    total = 0
    parity = len(digits) % 2
    for i, ch in enumerate(digits):
        d = int(ch)
        if i % 2 == parity:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


_CARD_IIN_PREFIXES: tuple[str, ...] = (
    "4",  # Visa
    "51", "52", "53", "54", "55",  # Mastercard (legacy range)
    "34", "37",  # Amex
    "6011", "65",  # Discover
    "300", "301", "302", "303", "304", "305", "36", "38",  # Diners
    "35",  # JCB
)


def _card_iin_valid(digits: str) -> bool:
    return any(digits.startswith(p) for p in _CARD_IIN_PREFIXES)


def _aba_checksum_valid(digits: str) -> bool:
    if len(digits) != 9 or digits == "000000000":
        return False
    d = [int(c) for c in digits]
    checksum = 3 * (d[0] + d[3] + d[6]) + 7 * (d[1] + d[4] + d[7]) + 1 * (d[2] + d[5] + d[8])
    return checksum % 10 == 0


def _ssn_digits_valid(digits: str) -> bool:
    if len(digits) != 9:
        return False
    area, group, serial = digits[0:3], digits[3:5], digits[5:9]
    if area in ("000", "666"):
        return False
    if area[0] == "9":
        return False
    if group == "00":
        return False
    if serial == "0000":
        return False
    return True


# EIN "campus" prefixes historically issued by the IRS. Best-effort list
# assembled from commonly published EIN-validator references — the IRS does
# not publish one single canonical current list, so treat this as a
# reasonable approximation, not gospel. Flagged for CTO review.
_EIN_VALID_PREFIXES: frozenset[str] = frozenset(
    """
    01 02 03 04 05 06 10 11 12 13 14 15 16
    20 21 22 23 24 25 26 27 30 31 32 33 34 35 36 37 38 39
    40 41 42 43 44 45 46 47 48
    50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68
    71 72 73 74 75 76 77
    80 81 82 83 84 85 86 87 88
    90 91 92 93 94 95 98 99
    """.split()
)

_SSN_CONTEXT_WORDS = ("ssn", "social security")
_EIN_CONTEXT_WORDS = ("ein", "employer", "fein", "tax id", "taxpayer id")
_ROUTING_CONTEXT_WORDS = ("routing", "aba", "r/t", "rtn")
_ACCOUNT_CONTEXT_WORDS = ("account", "acct", "a/c")


# --------------------------------------------------------------------------
# Individual validators (Match, full_text) -> bool
# --------------------------------------------------------------------------


def _validate_ssn_formatted(m: re.Match, text: str) -> bool:
    return _ssn_digits_valid(_digits_only(m.group(0)))


def _validate_ssn_bare(m: re.Match, text: str) -> bool:
    digits = m.group(0)
    if not _ssn_digits_valid(digits):
        return False
    return _context_has(text, m.start(), m.end(), _SSN_CONTEXT_WORDS)


def _validate_itin(m: re.Match, text: str) -> bool:
    area, group, serial = m.group(1), m.group(2), m.group(3)
    if area[0] != "9":
        return False
    if not (70 <= int(group) <= 89):
        return False
    return True


def _validate_ein_formatted(m: re.Match, text: str) -> bool:
    return m.group(1) in _EIN_VALID_PREFIXES


def _validate_ein_bare(m: re.Match, text: str) -> bool:
    digits = m.group(0)
    if digits[:2] not in _EIN_VALID_PREFIXES:
        return False
    return _context_has(text, m.start(), m.end(), _EIN_CONTEXT_WORDS)


def _validate_credit_card(m: re.Match, text: str) -> bool:
    digits = _digits_only(m.group(0))
    if not (13 <= len(digits) <= 19):
        return False
    if not _card_iin_valid(digits):
        return False
    return _luhn_valid(digits)


def _validate_routing(m: re.Match, text: str) -> bool:
    digits = m.group(0)
    if not _aba_checksum_valid(digits):
        return False
    return _context_has(text, m.start(), m.end(), _ROUTING_CONTEXT_WORDS)


def _validate_bank_account(m: re.Match, text: str) -> bool:
    digits = m.group(0)
    if not (6 <= len(digits) <= 17):
        return False
    return _context_has(text, m.start(), m.end(), _ACCOUNT_CONTEXT_WORDS)


_NANP_RE = re.compile(r"^\+?1?\D*([2-9]\d{2})\D*([2-9]\d{2})\D*(\d{4})$")


def _validate_phone_nanp(m: re.Match, text: str) -> bool:
    candidate = re.sub(r"(?i)\s*(?:ext\.?|x)\s*\d{1,10}$", "", m.group(0))
    return bool(_NANP_RE.match(candidate))


def _validate_phone_intl(m: re.Match, text: str) -> bool:
    digits = _digits_only(m.group(0))
    return 8 <= len(digits) <= 15


_TLD_RE = re.compile(r"[A-Za-z]{2,24}$")


def _validate_email(m: re.Match, text: str) -> bool:
    domain = m.group(0).split("@", 1)[1]
    if ".." in domain:
        return False
    last_label = domain.rsplit(".", 1)[-1]
    return bool(_TLD_RE.match(last_label))


def _validate_url(m: re.Match, text: str) -> bool:
    return True


def _validate_ipv4(m: re.Match, text: str) -> bool:
    try:
        ipaddress.IPv4Address(m.group(0))
    except ValueError:
        return False
    return True


def _validate_ipv6(m: re.Match, text: str) -> bool:
    candidate = m.group(0)
    if candidate.count(":") < 2:
        return False
    try:
        ipaddress.IPv6Address(candidate)
    except ValueError:
        return False
    return True


def _validate_mac(m: re.Match, text: str) -> bool:
    return True


_AWS_KEY_RE_STR = r"\bAKIA[0-9A-Z]{16}\b"
_GITHUB_TOKEN_RE_STR = r"\bgh[pos]_[A-Za-z0-9]{36,}\b"
_STRIPE_KEY_RE_STR = r"\bsk_live_[A-Za-z0-9]{16,}\b"


def _validate_labeled_credential(m: re.Match, text: str) -> bool:
    value = m.group(1)
    return len(value) >= 6 and any(c.isalnum() for c in value)


def _validate_labeled_password(m: re.Match, text: str) -> bool:
    value = m.group(1)
    return len(value) >= 6 and any(not c.isspace() for c in value)


def _validate_labeled_id(m: re.Match, text: str) -> bool:
    value = m.group(1)
    return 4 <= len(value) <= 25 and any(c.isdigit() for c in value)


def _validate_labeled_secret(m: re.Match, text: str) -> bool:
    value = m.group(1)
    return len(value) >= 16 and _shannon_entropy(value) >= 3.0


def _validate_generic_secret(m: re.Match, text: str) -> bool:
    value = m.group(1)
    if len(value) < 20:
        return False
    return _shannon_entropy(value) >= 3.0


def _validate_private_key(m: re.Match, text: str) -> bool:
    return True


def _b64url_decode_len_ok(segment: str) -> bool:
    # JWT header/payload are base64url, typically unpadded. Just check the
    # alphabet + that padding math works out; don't need the decoded bytes.
    return len(segment) >= 4


def _validate_jwt(m: re.Match, text: str) -> bool:
    header, payload, signature = m.group(1), m.group(2), m.group(3)
    return (
        header.startswith("eyJ")
        and _b64url_decode_len_ok(header)
        and _b64url_decode_len_ok(payload)
        and len(signature) >= 4
    )


# --------------------------------------------------------------------------
# Rule table
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Rule:
    entity_type: EntityType
    pattern: re.Pattern[str]
    validate: Callable[[re.Match, str], bool]
    group: int = 0  # which regex group's span to use (0 = whole match)


# PyMuPDF joins every word on a page with spaces, so delimiter-free field
# values need to stop at the next recognizable field label rather than at a
# newline. Keep this list restricted to strong document labels. In particular,
# delimiter-free free-form rules below require one of these boundaries; they
# do not treat end-of-string as sufficient, which avoids matching ordinary
# prose such as "The clinic remains open".
_PDF_FIELD_LABEL_RE_STR = (
    r"(?:full\s+legal\s+name|date\s+of\s+birth|ssn|passport\s+no\.?|"
    r"driver(?:'s)?\s+licen[cs]e|issuing\s+state|mother's\s+maiden\s+name|"
    r"(?:employee|payroll)\s+id|home\s+address|(?:personal\s+|work\s+)?email|mobile\s+phone|"
    r"emergency\s+contact|ip\s+address|device\s+id|bank|account\s+holder|"
    r"routing\s+number|account\s+number|credit\s+card|expiration\s*/\s*cvv|"
    r"annual\s+income|tax\s+filing\s+status|employer|manager|"
    r"work\s+phone|office\s+address|username|temporary\s+password|password|"
    r"api\s+(?:token|key)|security\s+answer|health\s+plan|member\s+id|"
    r"group\s+number|primary\s+physician|medical\s+record\s+(?:number|no\.?)|"
    r"blood\s+type|condition|medication|recent\s+visit|clinic|"
    r"verification\s+pin|\d+\.\s+[A-Za-z])"
)
_PDF_FIELD_END_RE_STR = rf"(?=\s+(?i:{_PDF_FIELD_LABEL_RE_STR})|[\r\n]+)"
_EXPLICIT_FIELD_END_RE_STR = rf"(?=\s+(?i:{_PDF_FIELD_LABEL_RE_STR})|[\r\n]+|$)"
_FREEFORM_VALUE_RE_STR = r"(\S(?:.{0,98}?\S)?)"


def _build_rules() -> list[_Rule]:
    return [
        # --- SSN ---------------------------------------------------------
        _Rule(EntityType.SSN, re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), _validate_ssn_formatted),
        _Rule(EntityType.SSN, re.compile(r"\b\d{9}\b"), _validate_ssn_bare),
        # --- ITIN ----------------------------------------------------------
        _Rule(
            EntityType.ITIN,
            re.compile(r"\b(9\d{2})-(\d{2})-(\d{4})\b"),
            _validate_itin,
        ),
        # --- EIN -------------------------------------------------------
        _Rule(
            EntityType.EIN, re.compile(r"\b(\d{2})-(\d{7})\b"), _validate_ein_formatted
        ),
        _Rule(EntityType.EIN, re.compile(r"\b\d{9}\b"), _validate_ein_bare),
        # --- CREDIT_CARD -------------------------------------------------
        _Rule(
            EntityType.CREDIT_CARD,
            re.compile(r"\b\d(?:[ -]?\d){12,18}\b"),
            _validate_credit_card,
        ),
        # --- ROUTING_NUMBER ------------------------------------------------
        _Rule(EntityType.ROUTING_NUMBER, re.compile(r"\b\d{9}\b"), _validate_routing),
        # --- BANK_ACCOUNT --------------------------------------------------
        _Rule(
            EntityType.BANK_ACCOUNT, re.compile(r"\b\d{6,17}\b"), _validate_bank_account
        ),
        # --- PHONE -----------------------------------------------------
        _Rule(
            EntityType.PHONE,
            re.compile(
                r"(?<!\d)(?:\+?1[\s.-]?)?\(?[2-9]\d{2}\)?[\s.-]?[2-9]\d{2}[\s.-]?\d{4}"
                r"(?:\s*(?:ext\.?|x)\s*\d{1,10})?(?!\d)",
                re.IGNORECASE,
            ),
            _validate_phone_nanp,
        ),
        _Rule(
            EntityType.PHONE,
            re.compile(r"(?<!\d)\+\d{1,3}[\s.-]?\d{1,4}(?:[\s.-]?\d{2,4}){1,4}(?!\d)"),
            _validate_phone_intl,
        ),
        # --- EMAIL -----------------------------------------------------
        _Rule(
            EntityType.EMAIL,
            re.compile(
                r"\b[A-Za-z0-9][A-Za-z0-9._%+-]*@[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?"
                r"(?:\.[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?)+\b"
            ),
            _validate_email,
        ),
        # --- URL -------------------------------------------------------
        _Rule(
            EntityType.URL,
            re.compile(r"\bhttps?://[^\s<>\"')\]]+"),
            _validate_url,
        ),
        # --- IP_ADDRESS --------------------------------------------------
        _Rule(
            EntityType.IP_ADDRESS,
            re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
            _validate_ipv4,
        ),
        _Rule(
            EntityType.IP_ADDRESS,
            # Not \b on either side: ':' is a non-word char, so a leading/
            # trailing "::" (a very common IPv6 compression, e.g. "::1")
            # would never satisfy a word-boundary anchor there.
            re.compile(
                r"(?<![0-9A-Fa-f:])(?:[0-9A-Fa-f]{0,4}:){2,7}[0-9A-Fa-f]{0,4}(?![0-9A-Fa-f:])"
            ),
            _validate_ipv6,
        ),
        # --- MAC_ADDRESS -------------------------------------------------
        _Rule(
            EntityType.MAC_ADDRESS,
            re.compile(r"\b(?:[0-9A-Fa-f]{2}[:-]){5}[0-9A-Fa-f]{2}\b"),
            _validate_mac,
        ),
        # --- Secrets: API_KEY ----------------------------------------------
        _Rule(EntityType.API_KEY, re.compile(_AWS_KEY_RE_STR), lambda m, t: True),
        _Rule(EntityType.API_KEY, re.compile(_GITHUB_TOKEN_RE_STR), lambda m, t: True),
        _Rule(EntityType.API_KEY, re.compile(_STRIPE_KEY_RE_STR), lambda m, t: True),
        _Rule(
            EntityType.API_KEY,
            re.compile(r"\bsk_(?:live|test)_[A-Za-z0-9]{16,}\b"),
            lambda m, t: True,
        ),
        _Rule(
            EntityType.API_KEY,
            re.compile(
                r"(?i)\b(?:api[_-]?key|secret[_-]?key|access[_-]?token|auth[_-]?token|"
                r"secret|token|password|passwd|pwd)\b\s*[:=]\s*[\"']?"
                r"([A-Za-z0-9_\-/+]{20,100})[\"']?"
            ),
            _validate_generic_secret,
            group=1,
        ),
        # Labels in PDFs are commonly extracted as plain whitespace-separated
        # text rather than `key: value`. These rules deliberately require a
        # strong field label so values are not guessed from shape alone.
        _Rule(
            EntityType.DATE_OF_BIRTH,
            re.compile(
                r"(?i)\b(?:date\s+of\s+birth|dob)\b\s*[:=]?\s*"
                r"((?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
                r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|"
                r"Dec(?:ember)?)\s+\d{1,2},?\s+\d{4}|\d{1,2}[/-]\d{1,2}[/-]\d{2,4})"
            ),
            lambda m, t: True,
            group=1,
        ),
        _Rule(
            EntityType.DRIVERS_LICENSE,
            re.compile(
                r"(?i)\b(?:driver(?:'s)?\s+licen[cs]e|dl\s*(?:number|no\.?))\b"
                r"\s*[:=]?\s*([A-Z0-9][A-Z0-9-]{4,24})"
            ),
            lambda m, t: True,
            group=1,
        ),
        _Rule(
            EntityType.DEVICE_ID,
            re.compile(
                r"(?i)\bdevice\s+id\b\s*[:=]?\s*"
                r"([0-9A-F]{8}-[0-9A-F]{4}-[0-9A-F]{4}-[0-9A-F]{4}-"
                r"[0-9A-F](?:\s?[0-9A-F]){11})"
            ),
            lambda m, t: True,
            group=1,
        ),
        _Rule(
            EntityType.GOVERNMENT_ID,
            re.compile(
                r"(?i)\b(?:employee|payroll)\s+id\b\s*[:=]?\s*"
                r"([A-Z0-9][A-Z0-9-]{3,24})"
            ),
            _validate_labeled_id,
            group=1,
        ),
        _Rule(
            EntityType.GOVERNMENT_ID,
            re.compile(r"(?i)\bverification\s+pin\b\s*[:=]?\s*(\d{4,12})"),
            lambda m, t: True,
            group=1,
        ),
        _Rule(
            EntityType.MEDICAL_ID,
            re.compile(
                r"(?i)\b(?:member\s+id|group\s+number|medical\s+record\s+(?:number|no\.?))"
                r"\s*[:=]?\s*([A-Z0-9][A-Z0-9-]{3,24})"
            ),
            _validate_labeled_id,
            group=1,
        ),
        _Rule(
            EntityType.CREDENTIAL,
            re.compile(
                r"(?i)\busername\b\s*[:=]\s*([^\s,;]{6,100})"
            ),
            _validate_labeled_credential,
            group=1,
        ),
        _Rule(
            EntityType.CREDENTIAL,
            re.compile(
                rf"(?i)\busername\b\s+([^\s,;]{{6,100}}){_PDF_FIELD_END_RE_STR}"
            ),
            _validate_labeled_credential,
            group=1,
        ),
        _Rule(
            EntityType.CREDENTIAL,
            re.compile(
                r"(?i)\b(?:temporary\s+password|password)\b\s*[:=]\s*[\"']"
                r"([^\"'\r\n]{6,100})[\"']"
            ),
            lambda m, t: True,
            group=1,
        ),
        _Rule(
            EntityType.CREDENTIAL,
            re.compile(
                rf"(?i)\b(?:temporary\s+password|password)\b\s*[:=]\s*"
                rf"(?![\"']){_FREEFORM_VALUE_RE_STR}{_EXPLICIT_FIELD_END_RE_STR}"
            ),
            _validate_labeled_password,
            group=1,
        ),
        _Rule(
            EntityType.CREDENTIAL,
            re.compile(
                rf"(?i)\btemporary\s+password\b\s+"
                rf"{_FREEFORM_VALUE_RE_STR}{_EXPLICIT_FIELD_END_RE_STR}"
            ),
            _validate_labeled_password,
            group=1,
        ),
        _Rule(
            EntityType.CREDENTIAL,
            re.compile(
                rf"\bPassword\b\s+{_FREEFORM_VALUE_RE_STR}{_PDF_FIELD_END_RE_STR}"
            ),
            _validate_labeled_password,
            group=1,
        ),
        _Rule(
            EntityType.CREDENTIAL,
            re.compile(
                rf"(?i)\bsecurity\s+answer\b\s*[:=]\s*"
                rf"{_FREEFORM_VALUE_RE_STR}{_EXPLICIT_FIELD_END_RE_STR}"
            ),
            _validate_labeled_credential,
            group=1,
        ),
        _Rule(
            EntityType.CREDENTIAL,
            re.compile(
                rf"(?i)\bsecurity\s+answer\b\s+"
                rf"{_FREEFORM_VALUE_RE_STR}{_PDF_FIELD_END_RE_STR}"
            ),
            _validate_labeled_credential,
            group=1,
        ),
        _Rule(
            EntityType.API_KEY,
            re.compile(r"(?i)\b(?:api\s+token|api\s+key)\b\s*[:=]?\s*([^\s,;]{16,100})"),
            _validate_labeled_secret,
            group=1,
        ),
        _Rule(
            EntityType.FINANCIAL_INFORMATION,
            re.compile(r"(?i)\b(?:expiration\s*/\s*cvv|cvv)\b\s*[:=]?\s*(\d{1,2}/\d{2,4}\s*/\s*\d{3,4}|\d{3,4})"),
            lambda m, t: True,
            group=1,
        ),
        _Rule(
            EntityType.FINANCIAL_INFORMATION,
            re.compile(r"(?i)\bannual\s+income\b\s*[:=]?\s*(\$[\d,]+(?:\.\d{2})?)"),
            lambda m, t: True,
            group=1,
        ),
        _Rule(
            EntityType.FINANCIAL_INFORMATION,
            re.compile(
                r"(?i)\btax\s+filing\s+status\b\s*[:=]?\s*"
                r"(single|married\s+filing\s+(?:jointly|separately)|head\s+of\s+household|widow(?:er)?)"
            ),
            lambda m, t: True,
            group=1,
        ),
        _Rule(
            EntityType.STREET_NAME,
            re.compile(
                r"(?i)\b(\d{1,6}\s+(?:(?!(?:address|phone|ext|number|id)\b)[A-Z0-9.'-]+\s+){0,5}"
                r"(?:Street|St\.?|Avenue|Ave\.?|Road|Rd\.?|Boulevard|Blvd\.?|Lane|Ln\.?|"
                r"Drive|Court|Ct\.?|Terrace|Way|Place|Pl\.?)(?![A-Za-z])"
                r"(?:,?\s+(?:Apt\.?|Suite|Ste\.?|Unit|#)\s*[A-Z0-9-]+)?)"
            ),
            lambda m, t: True,
            group=1,
        ),
        _Rule(
            EntityType.HEALTH_INFORMATION,
            re.compile(r"(?i)\bblood\s+type\b\s*[:=]?\s*((?:A|B|AB|O)\s+(?:positive|negative))"),
            lambda m, t: True,
            group=1,
        ),
        _Rule(
            EntityType.HEALTH_INFORMATION,
            re.compile(
                rf"(?i)\bhealth\s+plan\b\s*[:=]\s*"
                rf"{_FREEFORM_VALUE_RE_STR}{_EXPLICIT_FIELD_END_RE_STR}"
            ),
            lambda m, t: True,
            group=1,
        ),
        _Rule(
            EntityType.HEALTH_INFORMATION,
            re.compile(
                rf"(?i)\bhealth\s+plan\b\s+{_FREEFORM_VALUE_RE_STR}{_PDF_FIELD_END_RE_STR}"
            ),
            lambda m, t: True,
            group=1,
        ),
        _Rule(
            EntityType.HEALTH_INFORMATION,
            re.compile(
                rf"(?i)\bprimary\s+physician\b\s*[:=]\s*"
                rf"{_FREEFORM_VALUE_RE_STR}{_EXPLICIT_FIELD_END_RE_STR}"
            ),
            lambda m, t: True,
            group=1,
        ),
        _Rule(
            EntityType.HEALTH_INFORMATION,
            re.compile(
                rf"(?i)\bprimary\s+physician\b\s+{_FREEFORM_VALUE_RE_STR}{_PDF_FIELD_END_RE_STR}"
            ),
            lambda m, t: True,
            group=1,
        ),
        _Rule(
            EntityType.HEALTH_INFORMATION,
            re.compile(
                r"(?i)\brecent\s+visit\b\s*[:=]?\s*"
                r"((?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
                r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|"
                r"Dec(?:ember)?)\s+\d{1,2},?\s+\d{4}|\d{1,2}[/-]\d{1,2}[/-]\d{2,4})"
            ),
            lambda m, t: True,
            group=1,
        ),
        _Rule(
            EntityType.HEALTH_INFORMATION,
            re.compile(
                rf"(?i)\b(?:clinic|condition|medication)\b\s*[:=]\s*"
                rf"{_FREEFORM_VALUE_RE_STR}{_EXPLICIT_FIELD_END_RE_STR}"
            ),
            lambda m, t: True,
            group=1,
        ),
        _Rule(
            EntityType.HEALTH_INFORMATION,
            re.compile(
                rf"\b(?:Clinic|Condition|Medication)\b\s+"
                rf"{_FREEFORM_VALUE_RE_STR}{_PDF_FIELD_END_RE_STR}"
            ),
            lambda m, t: True,
            group=1,
        ),
        # --- PRIVATE_KEY -------------------------------------------------
        _Rule(
            EntityType.PRIVATE_KEY,
            re.compile(
                r"-----BEGIN (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----"
                r"[\s\S]+?"
                r"-----END (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----"
            ),
            _validate_private_key,
        ),
        # --- JWT ---------------------------------------------------------
        _Rule(
            EntityType.JWT,
            re.compile(r"\b(eyJ[A-Za-z0-9_-]+)\.([A-Za-z0-9_-]+)\.([A-Za-z0-9_-]+)\b"),
            _validate_jwt,
        ),
    ]


class RegexDetector:
    """Detector protocol implementation: deterministic regex + validator layer."""

    name = "regex"

    def __init__(self) -> None:
        self._rules = _build_rules()

    def detect(self, text: str) -> list[Span]:
        candidates: list[tuple[int, Span]] = []
        for order, rule in enumerate(self._rules):
            for m in rule.pattern.finditer(text):
                if not rule.validate(m, text):
                    continue
                start, end = m.span(rule.group)
                candidates.append(
                    (
                        order,
                        Span(
                            start=start,
                            end=end,
                            entity_type=rule.entity_type,
                            text=text[start:end],
                            confidence=1.0,
                            source="regex",
                        ),
                    )
                )
        return _resolve_longest_wins(candidates)


def _resolve_longest_wins(candidates: list[tuple[int, Span]]) -> list[Span]:
    """Longest match wins; ties broken by rule registration order. No range
    is ever reported twice (overlapping spans are mutually exclusive)."""
    ordered = sorted(candidates, key=lambda pair: (-(len(pair[1])), pair[0]))
    chosen: list[Span] = []
    for _, span in ordered:
        if any(span.overlaps(c) for c in chosen):
            continue
        chosen.append(span)
    chosen.sort(key=lambda s: s.start)
    return chosen

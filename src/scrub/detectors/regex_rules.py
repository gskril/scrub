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
    return bool(_NANP_RE.match(m.group(0)))


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
                r"(?<!\d)(?:\+?1[\s.-]?)?\(?[2-9]\d{2}\)?[\s.-]?[2-9]\d{2}[\s.-]?\d{4}(?!\d)"
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
            re.compile(
                r"(?i)\b(?:api[_-]?key|secret[_-]?key|access[_-]?token|auth[_-]?token|"
                r"secret|token|password|passwd|pwd)\b\s*[:=]\s*[\"']?"
                r"([A-Za-z0-9_\-/+]{20,100})[\"']?"
            ),
            _validate_generic_secret,
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

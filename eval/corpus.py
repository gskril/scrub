"""Golden corpus for the release-blocking eval harness (Phase 6).

Builds ~15-20 synthetic documents with KNOWN ground-truth PII annotations:
plain-text letters/emails/memos, source files with embedded secrets, a couple
of CSVs, three PDFs (reusing `tests/fixtures/make_pdfs.py`), and clean
false-positive-control documents (including realistic source code) that carry
NO ground truth at all.

All values are fabricated. Structured values (SSN/EIN/ITIN/routing/credit
card) are drawn from small pools that satisfy the regex+validator rules in
`scrub.detectors.regex_rules` (Luhn-valid cards, ABA-checksum-valid routing
numbers, IRS-prefix-valid EINs, etc.) so recall failures reflect real misses,
not fixtures the deterministic layer was never going to accept.

Names/addresses/phones/domains are drawn from seeded, shuffled pools so runs
are deterministic but every document gets different values. Each "genre" of
document (bank letter, HR offer email, IT memo, ...) has exactly ONE template,
used once — combined with per-document values, no two documents in this
corpus share a verbatim sentence or paragraph. That matters because Rampart
(the ML detector, see ARCHITECTURE.md/PLAN.md) has been observed to suppress
detections on exactly-repeated identical text; this corpus is built to never
trigger that quirk.

Runnable directly:

    python3 eval/corpus.py [output_dir]

Or imported:

    from eval.corpus import build_corpus
    docs = build_corpus(some_dir)
"""

from __future__ import annotations

import random
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests" / "fixtures"))
from make_pdfs import make_all as _make_pdf_fixtures  # noqa: E402

from scrub.types import EntityType

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "generated"

_SEED = 20260716

# Entity types the deterministic regex+validator layer owns (PLAN.md Sec 9):
# recall on these is release-blocking. Everything else (names, addresses,
# ITIN, ...) is reported but not gated -- Rampart is alpha.
RELEASE_BLOCKING_TYPES: frozenset[EntityType] = frozenset(
    {
        EntityType.SSN,
        EntityType.EIN,
        EntityType.CREDIT_CARD,
        EntityType.ROUTING_NUMBER,
        EntityType.BANK_ACCOUNT,
        EntityType.API_KEY,
        EntityType.PRIVATE_KEY,
        EntityType.JWT,
    }
)


@dataclass(slots=True)
class GroundTruth:
    """One known-PII value that must not survive redaction, plus the entity
    type it's expected to be caught as (for per-type recall reporting)."""

    value: str
    entity_type: EntityType


@dataclass(slots=True)
class CorpusDoc:
    """One golden-corpus document."""

    name: str
    path: Path
    kind: str  # "text" | "pdf"
    category: str  # "letter" | "code" | "csv" | "pdf" | "control"
    ground_truth: list[GroundTruth] = field(default_factory=list)
    is_control: bool = False  # True => false-positive control, ground_truth is always []


def _gt(value: str, etype: EntityType) -> GroundTruth:
    return GroundTruth(value=value, entity_type=etype)


# --------------------------------------------------------------------------
# Seeded value pools
# --------------------------------------------------------------------------

_NAME_POOL: list[tuple[str, str]] = [
    ("Maria", "Garcia"), ("Ravi", "Patel"), ("Fernanda", "Villalobos"), ("Wei", "Chen"),
    ("Amara", "Okafor"), ("Diego", "Torres"), ("Priya", "Sharma"), ("Owen", "Kelly"),
    ("Nadia", "Petrov"), ("Jamal", "Bennett"), ("Grace", "Doyle"), ("Kwame", "Mensah"),
    ("Elena", "Rossi"), ("Lucas", "Rivera"), ("Sofia", "Novak"), ("Yuki", "Tanaka"),
]

_STREET_POOL: list[str] = [
    "431 Elmwood Avenue", "2200 Baywood Terrace", "78 Sequoia Court",
    "1450 Harborview Drive", "615 Prairie Lane", "93 Foxglove Road",
    "3300 Cedar Ridge Boulevard", "512 Willowbrook Circle", "88 Lighthouse Way",
    "1010 Maplecrest Street", "240 Ironwood Path", "775 Riverside Parkway",
    "150 Birchwood Lane", "602 Meridian Court", "29 Thistledown Way",
]

_CSZ_POOL: list[tuple[str, str, str]] = [
    ("Springfield", "IL", "62704"), ("Portland", "OR", "97201"), ("Asheville", "NC", "28801"),
    ("Tucson", "AZ", "85701"), ("Madison", "WI", "53703"), ("Boulder", "CO", "80301"),
    ("Savannah", "GA", "31401"), ("Spokane", "WA", "99201"), ("Burlington", "VT", "05401"),
    ("Fargo", "ND", "58102"), ("Tempe", "AZ", "85281"), ("Frederick", "MD", "21701"),
    ("Eugene", "OR", "97401"), ("Boise", "ID", "83701"), ("Reno", "NV", "89501"),
]

_PHONE_POOL: list[str] = [
    "(312) 555-0148", "(503) 555-0173", "(720) 555-0119", "(206) 555-0142",
    "(414) 555-0186", "(602) 555-0157", "(919) 555-0134", "(802) 555-0121",
    "(701) 555-0198", "(480) 555-0163", "(828) 555-0177", "(509) 555-0111",
    "(541) 555-0129", "(208) 555-0166", "(775) 555-0144",
]

_DOMAIN_POOL: list[str] = ["example.com", "example.org", "example.net", "mail.example.com"]

# All satisfy scrub.detectors.regex_rules._ssn_digits_valid (area not
# 000/666/9xx, group != 00, serial != 0000).
_SSN_POOL: list[str] = [
    "458-02-6841", "523-11-4477", "612-45-9823", "734-22-1156", "845-67-3390",
    "219-08-5567", "367-94-2214", "581-33-7749", "702-56-1183", "158-77-4926",
]

# Formatted EIN, prefix must be in the validator's IRS-campus-prefix allowlist.
_EIN_POOL: list[str] = [
    "20-1234567", "33-9876543", "47-5566778", "52-1122334",
    "63-8877665", "74-4433221", "85-9911223", "91-6655443",
]

# area starts with 9, group in [70, 89].
_ITIN_POOL: list[str] = ["912-75-3348", "934-81-2207"]

# ABA-checksum-valid (generated + verified against the same checksum formula
# scrub.detectors.regex_rules._aba_checksum_valid uses).
_ROUTING_POOL: list[str] = [
    "026542351", "475255341", "395376724", "734714345",
    "581223623", "166587603", "466889373", "716572628",
]

_ACCOUNT_POOL: list[str] = [
    "000198456712", "000784512309", "000556781234",
    "000923847561", "000467128935", "000312985674",
]

# Well-known Luhn-valid + correct-IIN-prefix test card numbers (industry
# standard "test mode" numbers published by every major payment processor;
# not real cards).
_CARD_POOL: list[str] = [
    "4242 4242 4242 4242", "4000 0566 5566 5556", "5555 5555 5555 4444",
    "5105 1051 0510 5100", "378282246310005", "371449635398431",
    "6011 1111 1111 1117", "6011 0009 9013 9424",
]


class _Cycle:
    """Deterministic seeded-shuffle draw cursor: `next()` walks a shuffled
    copy of `items` without repeating until the pool is exhausted, then wraps.
    Keeps every document's values different from its neighbors' without
    forcing callers to hand-track indices."""

    def __init__(self, rng: random.Random, items: list) -> None:
        self._items = list(items)
        rng.shuffle(self._items)
        self._i = 0

    def next(self):
        v = self._items[self._i % len(self._items)]
        self._i += 1
        return v


@dataclass(slots=True)
class _Pools:
    names: _Cycle
    streets: _Cycle
    cszs: _Cycle
    phones: _Cycle
    domains: _Cycle
    ssns: _Cycle
    eins: _Cycle
    itins: _Cycle
    routings: _Cycle
    accounts: _Cycle
    cards: _Cycle


def _make_pools(seed: int = _SEED) -> _Pools:
    rng = random.Random(seed)
    return _Pools(
        names=_Cycle(rng, _NAME_POOL),
        streets=_Cycle(rng, _STREET_POOL),
        cszs=_Cycle(rng, _CSZ_POOL),
        phones=_Cycle(rng, _PHONE_POOL),
        domains=_Cycle(rng, _DOMAIN_POOL),
        ssns=_Cycle(rng, _SSN_POOL),
        eins=_Cycle(rng, _EIN_POOL),
        itins=_Cycle(rng, _ITIN_POOL),
        routings=_Cycle(rng, _ROUTING_POOL),
        accounts=_Cycle(rng, _ACCOUNT_POOL),
        cards=_Cycle(rng, _CARD_POOL),
    )


def _email(given: str, surname: str, domain: str) -> str:
    return f"{given.lower()}.{surname.lower()}@{domain}"


# --------------------------------------------------------------------------
# Letters / emails / memos -- one template each, drawn values plugged in
# --------------------------------------------------------------------------


def _doc_letter_bank(p: _Pools) -> tuple[str, list[GroundTruth]]:
    given, surname = p.names.next()
    street = p.streets.next()
    city, state, zip_ = p.cszs.next()
    phone = p.phones.next()
    email = _email(given, surname, p.domains.next())
    routing = p.routings.next()
    account = p.accounts.next()

    text = (
        "Riverbend Mutual Bank\n"
        f"200 Commerce Street, {city}, {state} {zip_}\n\n"
        "March 3, 2026\n\n"
        f"{given} {surname}\n"
        f"{street}\n"
        f"{city}, {state} {zip_}\n\n"
        "Re: Statement Cycle Confirmation\n\n"
        f"Dear {given} {surname},\n\n"
        "This letter confirms that your checking account statement for February "
        "2026 has been mailed to the address above. If you have questions, reach "
        f"us at {phone} or by email at {email}.\n\n"
        "For direct deposit setup, our processing routing number is "
        f"{routing} and your account number on file is {account}.\n\n"
        "Sincerely,\n"
        "Riverbend Mutual Bank Customer Service\n"
    )
    gt = [
        _gt(given, EntityType.GIVEN_NAME),
        _gt(surname, EntityType.SURNAME),
        _gt(street, EntityType.STREET_NAME),
        _gt(phone, EntityType.PHONE),
        _gt(email, EntityType.EMAIL),
        _gt(routing, EntityType.ROUTING_NUMBER),
        _gt(account, EntityType.BANK_ACCOUNT),
    ]
    return text, gt


def _doc_email_hr_offer(p: _Pools) -> tuple[str, list[GroundTruth]]:
    given, surname = p.names.next()
    street = p.streets.next()
    city, state, zip_ = p.cszs.next()
    phone = p.phones.next()
    email = _email(given, surname, p.domains.next())
    ssn = p.ssns.next()

    text = (
        "From: recruiting@northgate-tech.example\n"
        f"To: {email}\n"
        "Subject: Offer of Employment - Software Engineer II\n\n"
        f"Hi {given} {surname},\n\n"
        "We are delighted to offer you the Software Engineer II position at "
        "Northgate Technologies, starting Monday, September 14, 2026. Please "
        "review the attached offer letter and reply to confirm.\n\n"
        "For payroll setup we'll need to verify your Social Security number on "
        f"file: {ssn}. Our background-check vendor may also contact you at "
        f"{phone} to schedule a routine verification call.\n\n"
        "Please have your mailing address ready for the paperwork: "
        f"{street}, {city}, {state} {zip_}.\n\n"
        "Congratulations again,\n"
        "Northgate Technologies Recruiting Team\n"
    )
    gt = [
        _gt(given, EntityType.GIVEN_NAME),
        _gt(surname, EntityType.SURNAME),
        _gt(email, EntityType.EMAIL),
        _gt(ssn, EntityType.SSN),
        _gt(phone, EntityType.PHONE),
        _gt(street, EntityType.STREET_NAME),
    ]
    return text, gt


def _doc_memo_it_incident(p: _Pools) -> tuple[str, list[GroundTruth]]:
    given, surname = p.names.next()
    street = p.streets.next()
    city, state, zip_ = p.cszs.next()
    phone = p.phones.next()
    email = _email(given, surname, p.domains.next())

    text = (
        "INTERNAL MEMO -- IT Security\n\n"
        "To: All Engineering Staff\n"
        "From: IT Security Operations\n"
        "Date: April 18, 2026\n\n"
        "We detected unusual login activity on the account belonging to "
        f"{given} {surname} (workstation registered at {street}, {city}, "
        f"{state} {zip_}). The account has been temporarily locked pending "
        "review.\n\n"
        f"If this was you, please contact IT Security directly at {phone} or "
        f"{email} to verify your identity and restore access.\n\n"
        "As a reminder, never share your password or one-time codes with "
        "anyone claiming to be from IT, even by phone.\n\n"
        "IT Security Operations\n"
    )
    gt = [
        _gt(given, EntityType.GIVEN_NAME),
        _gt(surname, EntityType.SURNAME),
        _gt(street, EntityType.STREET_NAME),
        _gt(phone, EntityType.PHONE),
        _gt(email, EntityType.EMAIL),
    ]
    return text, gt


def _doc_letter_landlord(p: _Pools) -> tuple[str, list[GroundTruth]]:
    given, surname = p.names.next()
    street = p.streets.next()
    city, state, zip_ = p.cszs.next()
    phone = p.phones.next()
    email = _email(given, surname, p.domains.next())
    routing = p.routings.next()
    account = p.accounts.next()

    text = (
        "Hilltop Property Management\n"
        f"240 Ironwood Path, Suite 2, {city}, {state} {zip_}\n\n"
        "June 2, 2026\n\n"
        f"{given} {surname}\n"
        f"{street}\n"
        f"{city}, {state} {zip_}\n\n"
        "Re: Security Deposit Refund -- Unit 12B\n\n"
        f"Dear {given} {surname},\n\n"
        "Following your move-out inspection, we are processing your security "
        "deposit refund of $1,450.00. Refunds are issued by direct deposit; "
        "please confirm the banking details we have on file:\n\n"
        f"    Routing number: {routing}\n"
        f"    Account number: {account}\n\n"
        f"If any details are incorrect, call us at {phone} or email {email} "
        "within five business days.\n\n"
        "Hilltop Property Management\n"
    )
    gt = [
        _gt(given, EntityType.GIVEN_NAME),
        _gt(surname, EntityType.SURNAME),
        _gt(street, EntityType.STREET_NAME),
        _gt(phone, EntityType.PHONE),
        _gt(email, EntityType.EMAIL),
        _gt(routing, EntityType.ROUTING_NUMBER),
        _gt(account, EntityType.BANK_ACCOUNT),
    ]
    return text, gt


def _doc_email_support(p: _Pools) -> tuple[str, list[GroundTruth]]:
    given, surname = p.names.next()
    street = p.streets.next()
    city, state, zip_ = p.cszs.next()
    phone = p.phones.next()
    email = _email(given, surname, p.domains.next())
    card = p.cards.next()

    text = (
        "From: support@brightpath-retail.example\n"
        f"To: {email}\n"
        "Subject: Re: Order #48213 -- Delivery Update\n\n"
        f"Hi {given} {surname},\n\n"
        "Thanks for reaching out about order #48213. Your package was out for "
        f"delivery this morning and should arrive at {street}, {city}, {state} "
        f"{zip_} by end of day.\n\n"
        f"If it doesn't arrive, call our support line at {phone} and reference "
        "this ticket. You can also reply directly to this email.\n\n"
        "For verification, can you confirm the last four digits of the card "
        f"used? The full card number on file is {card}, expiring 09/28.\n\n"
        "Thanks for your patience,\n"
        "BrightPath Retail Support\n"
    )
    gt = [
        _gt(given, EntityType.GIVEN_NAME),
        _gt(surname, EntityType.SURNAME),
        _gt(email, EntityType.EMAIL),
        _gt(phone, EntityType.PHONE),
        _gt(street, EntityType.STREET_NAME),
        _gt(card, EntityType.CREDIT_CARD),
    ]
    return text, gt


def _doc_memo_finance(p: _Pools) -> tuple[str, list[GroundTruth]]:
    given, surname = p.names.next()
    given2, surname2 = p.names.next()
    street = p.streets.next()
    city, state, zip_ = p.cszs.next()
    phone = p.phones.next()
    email = _email(given, surname, p.domains.next())
    ein = p.eins.next()
    itin = p.itins.next()
    routing = p.routings.next()
    account = p.accounts.next()

    text = (
        "FINANCE MEMO -- Contractor Payment Setup\n\n"
        "To: Accounts Payable\n"
        "From: Finance Operations\n"
        "Date: July 2, 2026\n\n"
        "Please set up recurring payments for our new independent contractor, "
        f"{given} {surname} (EIN {ein}), who will be invoicing under "
        f"{surname} Consulting LLC. {given}'s mailing address for 1099 "
        f"purposes is {street}, {city}, {state} {zip_}.\n\n"
        "Direct deposit details:\n\n"
        f"    Routing number: {routing}\n"
        f"    Account number: {account}\n\n"
        f"{given} can be reached at {phone} or {email} with any questions "
        "about invoicing cadence.\n\n"
        f"For our overseas contractor {given2} {surname2}, payroll has "
        f"recorded ITIN {itin} pending final W-8BEN review.\n\n"
        "Finance Operations\n"
    )
    gt = [
        _gt(given, EntityType.GIVEN_NAME),
        _gt(surname, EntityType.SURNAME),
        _gt(ein, EntityType.EIN),
        _gt(street, EntityType.STREET_NAME),
        _gt(routing, EntityType.ROUTING_NUMBER),
        _gt(account, EntityType.BANK_ACCOUNT),
        _gt(phone, EntityType.PHONE),
        _gt(email, EntityType.EMAIL),
        _gt(given2, EntityType.GIVEN_NAME),
        _gt(surname2, EntityType.SURNAME),
        _gt(itin, EntityType.ITIN),
    ]
    return text, gt


# --------------------------------------------------------------------------
# Code files with embedded secrets
# --------------------------------------------------------------------------


def _doc_aws_config() -> tuple[str, list[GroundTruth]]:
    aws_key = "AKIAQR7ZXJH3MN2K9PLE"
    api_key = "8f3a91c0e4b7d2f6a9c1e5b8d3f7a2c6e9b4d1f8a7c2"
    secret_key = "wJalrXUtnFEMI7K8vN2pQeYs3Lm9BhZxRtCkAplo"
    stripe_key = "sk_live_51Hh9nzQr8mLpX3vT6bKdEfGh2AwYcZ"

    text = (
        '"""Deployment configuration for the staging environment.\n\n'
        "Loaded at deploy time by the release pipeline. Do not commit real\n"
        'credentials to version control (this file is for local testing only).\n"""\n\n'
        f'AWS_ACCESS_KEY_ID = "{aws_key}"\n'
        'AWS_REGION = "us-west-2"\n\n'
        f'api_key = "{api_key}"\n'
        f'secret_key = "{secret_key}"\n\n'
        f'STRIPE_LIVE_KEY = "{stripe_key}"\n\n\n'
        "def get_deploy_config() -> dict:\n"
        "    return {\n"
        '        "access_key_id": AWS_ACCESS_KEY_ID,\n'
        '        "region": AWS_REGION,\n'
        "    }\n"
    )
    gt = [
        _gt(aws_key, EntityType.API_KEY),
        _gt(api_key, EntityType.API_KEY),
        _gt(secret_key, EntityType.API_KEY),
        _gt(stripe_key, EntityType.API_KEY),
    ]
    return text, gt


def _doc_dotenv() -> tuple[str, list[GroundTruth]]:
    secret_key = "4f8b2a9d6e1c7f3a8b5d2e9c6f1a4b7d8e3c9f2a"
    stripe_key = "sk_live_9pQrTz2VbNmXeWy4LkJhGf7DsA"
    github_token = "ghp_9f3kD8mQpL2xZtRv6NcYbHaWj4Se7UoI1BgP"
    auth_token = "nR8vKpQ2xLmZ7bYcFj4WsD9tHaE6oU3iG5rN"
    password = "Sn0wLeopardTrek42Zephyr99"

    text = (
        "# Production environment secrets -- DO NOT COMMIT\n"
        "DATABASE_URL=postgres://svc_prod:CorrectHorseBatteryStaple99@db.internal:5432/appdb\n"
        f"SECRET_KEY={secret_key}\n"
        f"STRIPE_SECRET_KEY={stripe_key}\n"
        f"GITHUB_TOKEN={github_token}\n"
        f"AUTH_TOKEN={auth_token}\n"
        f"password={password}\n"
    )
    gt = [
        _gt(secret_key, EntityType.API_KEY),
        _gt(stripe_key, EntityType.API_KEY),
        _gt(github_token, EntityType.API_KEY),
        _gt(auth_token, EntityType.API_KEY),
        _gt(password, EntityType.API_KEY),
    ]
    return text, gt


def _doc_private_key() -> tuple[str, list[GroundTruth]]:
    body_line = "Xw2Yv1mQzR8tL5nJ6cW3aE9pD0oV7xU4iS1kH2gT6rB3yN8fZ1qC5jM0lX9wQe"
    text = (
        "-----BEGIN RSA PRIVATE KEY-----\n"
        "MIIEpQIBAAKCAQEAwc9GVh6vC8pQdVYVn6+KRj4gk3RxG1M8sT2P9jZ4bQb7Kf\n"
        f"{body_line}\n"
        "YuI7tR4bV2nK8sD1oP6cM9xJ3gH5vN0aQ7fL2rW6yT4mE8pB1sX9uV3jK5oR7\n"
        "zC0nL4bI6dQ2fH8mS5tY3rV9wA1eG7pK4uN0oX6cJ8sB2vD5lT9qR3fW1yI7hM\n"
        "QIDAQABAoIBAQCU4x0k+ntQ8m4vRq2fA9lYcG3sJ7wZ1oX5bK6tD8nH0eV2rM\n"
        "zL9pC4qA1sT6uY3vN8kJ0hR5wG2eB7fD9lX4mQ1oI6cP3rN8sV0tY7uK5jH2wZ\n"
        "-----END RSA PRIVATE KEY-----\n"
    )
    gt = [_gt(body_line, EntityType.PRIVATE_KEY)]
    return text, gt


def _doc_auth_service_ts() -> tuple[str, list[GroundTruth]]:
    jwt1 = (
        "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
        ".eyJzdWIiOiJzdmMtYXV0aC1ib3QiLCJyb2xlIjoiYWRtaW4ifQ"
        ".k3JmX9pQ7vT2rY8wL4bN6sD1oH5cA0eG3fK9uV7mZ2x"
    )
    jwt2 = (
        "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
        ".eyJzdWIiOiJzdmMtcmVmcmVzaCIsImV4cCI6MTc4OTAwMDAwMH0"
        ".qL8vN2mK5wR9tY3bH7fD0sA4eG1oC6uX8pJ2rZ9kW3n"
    )
    text = (
        "// Auth service bootstrap -- talks to the identity provider.\n\n"
        "export const SERVICE_JWT =\n"
        f'  "{jwt1}";\n\n'
        "export const REFRESH_JWT =\n"
        f'  "{jwt2}";\n\n'
        "export function currentServiceToken(): string {\n"
        "  return SERVICE_JWT;\n"
        "}\n"
    )
    gt = [_gt(jwt1, EntityType.JWT), _gt(jwt2, EntityType.JWT)]
    return text, gt


# --------------------------------------------------------------------------
# CSVs
# --------------------------------------------------------------------------


def _doc_customers_csv(p: _Pools) -> tuple[str, list[GroundTruth]]:
    rows = []
    gt: list[GroundTruth] = []
    for _ in range(4):
        given, surname = p.names.next()
        email = _email(given, surname, p.domains.next())
        ssn = p.ssns.next()
        phone = p.phones.next()
        rows.append(f"{given} {surname},{email},{ssn},{phone}")
        gt += [
            _gt(given, EntityType.GIVEN_NAME),
            _gt(surname, EntityType.SURNAME),
            _gt(email, EntityType.EMAIL),
            _gt(ssn, EntityType.SSN),
            _gt(phone, EntityType.PHONE),
        ]
    text = "name,email,ssn,phone\n" + "\n".join(rows) + "\n"
    return text, gt


def _doc_contractors_csv(p: _Pools) -> tuple[str, list[GroundTruth]]:
    rows = []
    gt: list[GroundTruth] = []
    for _ in range(4):
        given, surname = p.names.next()
        email = _email(given, surname, p.domains.next())
        ein = p.eins.next()
        routing = p.routings.next()
        account = p.accounts.next()
        rows.append(
            f"{given} {surname},{email},{ein},routing {routing},account {account}"
        )
        gt += [
            _gt(given, EntityType.GIVEN_NAME),
            _gt(surname, EntityType.SURNAME),
            _gt(email, EntityType.EMAIL),
            _gt(ein, EntityType.EIN),
            _gt(routing, EntityType.ROUTING_NUMBER),
            _gt(account, EntityType.BANK_ACCOUNT),
        ]
    text = "name,email,ein,bank_routing,bank_account\n" + "\n".join(rows) + "\n"
    return text, gt


# --------------------------------------------------------------------------
# Clean false-positive controls -- ground truth is always []
# --------------------------------------------------------------------------


def _doc_control_data_pipeline() -> str:
    return (
        '"""Batch ETL step: normalize and aggregate daily event counts."""\n\n'
        "from __future__ import annotations\n\n"
        "from collections import defaultdict\n"
        "from dataclasses import dataclass\n\n\n"
        "@dataclass(slots=True)\n"
        "class EventBucket:\n"
        "    key: str\n"
        "    count: int = 0\n"
        "    total_value: float = 0.0\n\n"
        "    def add(self, value: float) -> None:\n"
        "        self.count += 1\n"
        "        self.total_value += value\n\n\n"
        "def aggregate(events: list[dict]) -> dict[str, EventBucket]:\n"
        "    buckets: dict[str, EventBucket] = defaultdict(lambda: EventBucket(key=''))\n"
        "    for event in events:\n"
        "        key = event.get('category', 'uncategorized')\n"
        "        bucket = buckets.setdefault(key, EventBucket(key=key))\n"
        "        bucket.add(float(event.get('value', 0.0)))\n"
        "    return dict(buckets)\n\n\n"
        "def top_n(buckets: dict[str, EventBucket], n: int = 5) -> list[EventBucket]:\n"
        "    return sorted(buckets.values(), key=lambda b: b.total_value, reverse=True)[:n]\n"
    )


def _doc_control_dashboard_tsx() -> str:
    return (
        "import { useEffect, useState } from 'react';\n\n"
        "interface MetricCardProps {\n"
        "  label: string;\n"
        "  value: number;\n"
        "  trend: 'up' | 'down' | 'flat';\n"
        "}\n\n"
        "export function MetricCard({ label, value, trend }: MetricCardProps) {\n"
        "  return (\n"
        "    <div className=\"metric-card\">\n"
        "      <span className=\"metric-label\">{label}</span>\n"
        "      <span className=\"metric-value\">{value.toLocaleString()}</span>\n"
        "      <span className={`metric-trend metric-trend--${trend}`}>{trend}</span>\n"
        "    </div>\n"
        "  );\n"
        "}\n\n"
        "export function Dashboard() {\n"
        "  const [loading, setLoading] = useState(true);\n"
        "  useEffect(() => {\n"
        "    const timer = setTimeout(() => setLoading(false), 300);\n"
        "    return () => clearTimeout(timer);\n"
        "  }, []);\n"
        "  if (loading) return <p>Loading metrics...</p>;\n"
        "  return <MetricCard label=\"Active sessions\" value={1240} trend=\"up\" />;\n"
        "}\n"
    )


def _doc_control_readme() -> str:
    return (
        "# widget-toolkit\n\n"
        "A small collection of reusable layout primitives for internal dashboards.\n\n"
        "## Install\n\n"
        "```\n"
        "npm install widget-toolkit\n"
        "```\n\n"
        "## Usage\n\n"
        "Import the primitives you need and compose them like any other "
        "component. See the docs site for a full component gallery and API "
        "reference: https://docs.example.com/widget-toolkit\n\n"
        "## Contributing\n\n"
        "Pull requests are welcome. Please open an issue first to discuss any "
        "significant change. Run the test suite before submitting.\n\n"
        "## License\n\n"
        "MIT\n"
    )


def _doc_control_numeric_edge_cases() -> str:
    return (
        '"""Fixtures for the order-parsing edge-case tests.\n\n'
        "None of the values below are valid PII-shaped identifiers -- they're\n"
        "deliberately malformed (bad checksum / bad prefix / out-of-range\n"
        'octets) so the parser\'s rejection path gets exercised.\n"""\n\n'
        "INVALID_SSN_LIKE = \"000-00-0000\"        # area 000 is never issued\n"
        "INVALID_CARD_LIKE = \"1234 5678 9012 3456\"  # fails Luhn and IIN checks\n"
        "INVALID_PHONE_LIKE = \"111-111-1111\"      # NANP area codes can't start with 1\n"
        "INVALID_IP_LIKE = \"999.999.999.999\"      # octets out of range\n"
        "BUILD_VERSION = \"v10.20.30.40021\"\n"
        "ORDER_REF = \"ORD-2026-000481\"\n\n\n"
        "def looks_like_order_ref(value: str) -> bool:\n"
        "    return value.startswith(\"ORD-\")\n"
    )


def _doc_control_newsletter() -> str:
    return (
        "Product Update -- Q3 Release Notes\n\n"
        "This quarter's release focuses on performance and reliability. "
        "Highlights include a faster cold-start path, a redesigned settings "
        "panel, and a batch of accessibility fixes across the dashboard.\n\n"
        "We also shipped a long-requested export feature: any table view can "
        "now be exported to CSV directly from the toolbar.\n\n"
        "As always, thank you for the detailed bug reports last quarter -- "
        "they shaped a good chunk of this release. The full changelog is "
        "linked from the release notes page.\n\n"
        "See you next quarter.\n"
    )


# --------------------------------------------------------------------------
# PDFs -- reuse tests/fixtures/make_pdfs.py; ground truth mirrors the values
# hardcoded there (see that module for the source of truth).
# --------------------------------------------------------------------------


def _pdf_ground_truth() -> dict[str, list[GroundTruth]]:
    return {
        "fake_w2.pdf": [
            _gt("458-02-6841", EntityType.SSN),
            _gt("12-3456789", EntityType.EIN),
            _gt("Maria", EntityType.GIVEN_NAME),
            _gt("Garcia", EntityType.SURNAME),
            _gt("431 Elmwood Avenue", EntityType.STREET_NAME),
        ],
        "contract.pdf": [
            _gt("Maria", EntityType.GIVEN_NAME),
            _gt("Garcia", EntityType.SURNAME),
            _gt("Jordan", EntityType.GIVEN_NAME),
            _gt("Blake", EntityType.SURNAME),
            _gt("maria.garcia@example.com", EntityType.EMAIL),
            _gt("(312) 555-0148", EntityType.PHONE),
            _gt("021000021", EntityType.ROUTING_NUMBER),
            _gt("000123456789", EntityType.BANK_ACCOUNT),
        ],
        "letter.pdf": [
            _gt("Maria", EntityType.GIVEN_NAME),
            _gt("Garcia", EntityType.SURNAME),
            _gt("431 Elmwood Avenue", EntityType.STREET_NAME),
        ],
    }


# --------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------


def build_corpus(output_dir: Path | str = DEFAULT_OUTPUT_DIR) -> list[CorpusDoc]:
    """Write every corpus document into `output_dir` and return their
    metadata (path + ground truth). Deterministic across runs."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    pools = _make_pools()
    docs: list[CorpusDoc] = []

    def _write_text(name: str, category: str, text: str, gt: list[GroundTruth]) -> None:
        path = out / name
        path.write_text(text, encoding="utf-8")
        docs.append(CorpusDoc(name=name, path=path, kind="text", category=category, ground_truth=gt))

    # -- letters / emails / memos ------------------------------------------
    for name, builder in (
        ("letter_bank.txt", _doc_letter_bank),
        ("email_hr_offer.txt", _doc_email_hr_offer),
        ("memo_it_incident.txt", _doc_memo_it_incident),
        ("letter_landlord.txt", _doc_letter_landlord),
        ("email_customer_support.txt", _doc_email_support),
        ("memo_finance_reimbursement.txt", _doc_memo_finance),
    ):
        text, gt = builder(pools)
        _write_text(name, "letter", text, gt)

    # -- code files with embedded secrets ------------------------------------
    for name, builder in (
        ("aws_config.py", _doc_aws_config),
        (".env.production", _doc_dotenv),
        ("deploy_key", _doc_private_key),
        ("auth_service.ts", _doc_auth_service_ts),
    ):
        text, gt = builder()
        _write_text(name, "code", text, gt)

    # -- CSVs -----------------------------------------------------------
    for name, builder in (
        ("customers.csv", _doc_customers_csv),
        ("contractors.csv", _doc_contractors_csv),
    ):
        text, gt = builder(pools)
        _write_text(name, "csv", text, gt)

    # -- clean false-positive controls (no ground truth) ---------------------
    for name, builder in (
        ("control_data_pipeline.py", _doc_control_data_pipeline),
        ("control_dashboard.tsx", _doc_control_dashboard_tsx),
        ("control_readme.md", _doc_control_readme),
        ("control_numeric_edge_cases.py", _doc_control_numeric_edge_cases),
        ("control_newsletter.txt", _doc_control_newsletter),
    ):
        text = builder()
        path = out / name
        path.write_text(text, encoding="utf-8")
        docs.append(
            CorpusDoc(name=name, path=path, kind="text", category="control", is_control=True)
        )

    # -- PDFs (reuse tests/fixtures/make_pdfs.py) ----------------------------
    pdf_dir = out / "pdf"
    pdf_paths = {p.name: p for p in _make_pdf_fixtures(pdf_dir)}
    pdf_gt = _pdf_ground_truth()
    for name, path in pdf_paths.items():
        docs.append(
            CorpusDoc(
                name=name, path=path, kind="pdf", category="pdf",
                ground_truth=pdf_gt.get(name, []),
            )
        )

    return docs


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    output_dir = Path(argv[0]) if argv else DEFAULT_OUTPUT_DIR
    docs = build_corpus(output_dir)
    n_gt = sum(len(d.ground_truth) for d in docs)
    n_control = sum(1 for d in docs if d.is_control)
    print(f"Wrote {len(docs)} documents to {output_dir} "
          f"({len(docs) - n_control} with ground truth, {n_control} clean controls, "
          f"{n_gt} total ground-truth values).")
    for d in docs:
        print(f"  [{d.category:7}] {d.name:36} gt={len(d.ground_truth)}")


if __name__ == "__main__":
    main()

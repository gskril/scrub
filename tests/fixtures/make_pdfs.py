"""Generate synthetic sample PDFs for testing the scrub PDF extractor/redactor.

No external assets — everything is drawn with PyMuPDF at runtime. All PII in
these fixtures is fabricated for testing purposes only.

Runnable directly:

    python3 tests/fixtures/make_pdfs.py [output_dir]

Or imported:

    from tests.fixtures.make_pdfs import make_all
    paths = make_all(some_dir)
"""

from __future__ import annotations

import sys
from pathlib import Path

import pymupdf

DEFAULT_OUTPUT_DIR = Path(__file__).parent / "generated"

PAGE_RECT = pymupdf.paper_rect("letter")  # 612 x 792 pt

BLACK = (0, 0, 0)
GRAY = (0.4, 0.4, 0.4)


def _label_value(page: pymupdf.Page, x: float, y: float, label: str, value: str,
                  label_size: float = 7, value_size: float = 10, gap: float = 11) -> None:
    """Draw a small caps label above a value line, W-2-box style."""
    page.insert_text((x, y), label, fontsize=label_size, color=GRAY, fontname="helv")
    page.insert_text((x, y + gap), value, fontsize=value_size, color=BLACK, fontname="helv")


def _textbox(page: pymupdf.Page, rect: pymupdf.Rect, text: str, **kwargs) -> float:
    """insert_textbox that fails loudly if the rect is too small.

    PyMuPDF's insert_textbox silently draws NOTHING (not even a partial fit)
    when the text doesn't fit the rect — it returns a negative "unused space"
    value instead of raising. That's an easy way to accidentally ship a
    fixture that's missing the very PII the tests are supposed to redact, so
    every textbox in this module goes through here instead of calling
    insert_textbox directly.
    """
    unused = page.insert_textbox(rect, text, **kwargs)
    if unused < 0:
        raise RuntimeError(
            f"textbox too small by {-unused:.1f}pt for rect {tuple(rect)}: {text[:60]!r}..."
        )
    return unused


def _make_fake_w2(path: Path) -> Path:
    """A mock W-2-style tax form, one page, laid out with boxes/lines."""
    doc = pymupdf.open()
    page = doc.new_page(width=PAGE_RECT.width, height=PAGE_RECT.height)

    _textbox(
        page,
        pymupdf.Rect(36, 20, 576, 62),
        "Form W-2  Wage and Tax Statement (SPECIMEN - SYNTHETIC TEST DATA, NOT A REAL RECORD)",
        fontsize=13,
        fontname="helv",
        color=BLACK,
        lineheight=1.15,
    )

    # Outer form border
    page.draw_rect(pymupdf.Rect(36, 60, 576, 500), color=BLACK, width=1)
    page.draw_line((36, 90), (576, 90), color=BLACK, width=0.75)

    # Box a: Employee SSN
    box_a = pymupdf.Rect(44, 66, 220, 88)
    page.draw_rect(box_a, color=BLACK, width=0.5)
    _label_value(page, 48, 76, "a  Employee's social security number", "458-02-6841")

    # Box b: Employer EIN
    box_b = pymupdf.Rect(228, 66, 420, 88)
    page.draw_rect(box_b, color=BLACK, width=0.5)
    _label_value(page, 232, 76, "b  Employer identification number (EIN)", "12-3456789")

    # Box c: Employer name/address
    box_c = pymupdf.Rect(44, 96, 420, 160)
    page.draw_rect(box_c, color=BLACK, width=0.5)
    page.insert_text((48, 106), "c  Employer's name, address, and ZIP code",
                      fontsize=7, color=GRAY, fontname="helv")
    _textbox(
        page,
        pymupdf.Rect(48, 112, 416, 158),
        "Acme Corp\n900 Innovation Drive, Suite 400\nSpringfield, IL 62701",
        fontsize=10,
        fontname="helv",
        color=BLACK,
        lineheight=1.15,
    )

    # Box e: Employee name; Box f: Employee address
    box_e = pymupdf.Rect(44, 168, 420, 192)
    page.draw_rect(box_e, color=BLACK, width=0.5)
    _label_value(page, 48, 178, "e  Employee's first name and last name", "Maria Garcia")

    box_f = pymupdf.Rect(44, 200, 420, 236)
    page.draw_rect(box_f, color=BLACK, width=0.5)
    page.insert_text((48, 210), "f  Employee's address and ZIP code",
                      fontsize=7, color=GRAY, fontname="helv")
    _textbox(
        page,
        pymupdf.Rect(48, 216, 416, 234),
        "431 Elmwood Avenue, Springfield, IL 62704",
        fontsize=10,
        fontname="helv",
        color=BLACK,
    )

    # Wage boxes (1-6), simple grid
    wage_fields = [
        ("1  Wages, tips, other compensation", "68,540.00"),
        ("2  Federal income tax withheld", "9,127.40"),
        ("3  Social security wages", "68,540.00"),
        ("4  Social security tax withheld", "4,249.48"),
        ("5  Medicare wages and tips", "68,540.00"),
        ("6  Medicare tax withheld", "993.83"),
    ]
    col_w = 176
    row_h = 40
    top = 244
    for i, (label, value) in enumerate(wage_fields):
        col = i % 3
        row = i // 3
        x = 44 + col * col_w
        y = top + row * row_h
        rect = pymupdf.Rect(x, y, x + col_w, y + row_h)
        page.draw_rect(rect, color=BLACK, width=0.5)
        page.insert_text((x + 4, y + 12), label, fontsize=6.5, color=GRAY, fontname="helv")
        page.insert_text((x + 4, y + 28), value, fontsize=10, color=BLACK, fontname="helv")

    _textbox(
        page,
        pymupdf.Rect(44, 480, 500, 505),
        "Tax year 2025  |  Reference: SPECIMEN-0001  |  All values fabricated for software testing.",
        fontsize=8,
        fontname="helv",
        color=GRAY,
    )

    doc.save(path)
    doc.close()
    return path


def _make_contract(path: Path) -> Path:
    """A 2-3 page synthetic consulting agreement."""
    doc = pymupdf.open()

    # --- Page 1: preamble ---
    p1 = doc.new_page(width=PAGE_RECT.width, height=PAGE_RECT.height)
    _textbox(
        p1,
        pymupdf.Rect(54, 44, 558, 100),
        "CONSULTING SERVICES AGREEMENT (SYNTHETIC TEST DOCUMENT)",
        fontsize=15,
        fontname="helv",
        color=BLACK,
    )
    _textbox(
        p1,
        pymupdf.Rect(54, 110, 558, 410),
        (
            "This Consulting Services Agreement (\"Agreement\") is entered into as of "
            "January 5, 2026, by and between Acme Corp, a Delaware corporation "
            "(\"Client\"), and Maria Garcia, an independent contractor "
            "(\"Consultant\").\n\n"
            "1. Contact Information. The Consultant's contact details for all "
            "notices under this Agreement are as follows: email "
            "maria.garcia@example.com, phone (312) 555-0148.\n\n"
            "2. Services. Consultant Maria Garcia shall provide software "
            "engineering consulting services to Acme Corp as described in "
            "Exhibit A, attached hereto and incorporated by reference.\n\n"
            "3. Term. This Agreement shall commence on the date first written "
            "above and continue until terminated by either party with thirty "
            "(30) days' written notice to the other party."
        ),
        fontsize=10.5,
        fontname="helv",
        color=BLACK,
        lineheight=1.4,
    )

    # --- Page 2: payment clause with banking details ---
    p2 = doc.new_page(width=PAGE_RECT.width, height=PAGE_RECT.height)
    _textbox(
        p2,
        pymupdf.Rect(54, 54, 558, 80),
        "Page 2 - Payment Terms",
        fontsize=12,
        fontname="helv",
        color=GRAY,
    )
    _textbox(
        p2,
        pymupdf.Rect(54, 90, 558, 420),
        (
            "4. Compensation. In consideration for the services rendered by "
            "Maria Garcia, Acme Corp shall pay Consultant a monthly retainer "
            "of $8,500.00, due on the first business day of each month.\n\n"
            "5. Payment Method. Payments shall be made by direct deposit to "
            "the bank account designated by Consultant Maria Garcia:\n\n"
            "     Bank routing number: 021000021\n"
            "     Bank account number: 000123456789\n"
            "     Account holder: Maria Garcia\n\n"
            "6. Invoicing. Consultant shall submit invoices to Acme Corp on a "
            "monthly basis. Questions regarding payment may be directed to "
            "maria.garcia@example.com or by phone at (312) 555-0148."
        ),
        fontsize=10.5,
        fontname="helv",
        color=BLACK,
        lineheight=1.4,
    )

    # --- Page 3: signatures ---
    p3 = doc.new_page(width=PAGE_RECT.width, height=PAGE_RECT.height)
    _textbox(
        p3,
        pymupdf.Rect(54, 54, 558, 80),
        "Page 3 - Signatures",
        fontsize=12,
        fontname="helv",
        color=GRAY,
    )
    _textbox(
        p3,
        pymupdf.Rect(54, 90, 558, 300),
        (
            "IN WITNESS WHEREOF, the parties have executed this Agreement as "
            "of the date first written above.\n\n"
            "CLIENT: Acme Corp\n\n"
            "By: _______________________\n"
            "Name: Jordan Blake, VP Engineering\n\n"
            "CONSULTANT:\n\n"
            "By: _______________________\n"
            "Name: Maria Garcia\n"
            "Email: maria.garcia@example.com\n"
            "Phone: (312) 555-0148"
        ),
        fontsize=10.5,
        fontname="helv",
        color=BLACK,
        lineheight=1.4,
    )

    doc.save(path)
    doc.close()
    return path


def _make_letter(path: Path) -> Path:
    """A 1-page letter with letterhead, address block, and reference number.

    Sets document metadata (author/title) referencing the recipient so the
    redaction test can prove metadata is scrubbed too.
    """
    doc = pymupdf.open()
    page = doc.new_page(width=PAGE_RECT.width, height=PAGE_RECT.height)

    _textbox(
        page,
        pymupdf.Rect(54, 36, 558, 72),
        "Springfield Community Bank",
        fontsize=16,
        fontname="helv",
        color=BLACK,
    )
    _textbox(
        page,
        pymupdf.Rect(54, 68, 558, 86),
        "100 Financial Plaza, Springfield, IL 62701",
        fontsize=9,
        fontname="helv",
        color=GRAY,
    )
    page.draw_line((54, 90), (558, 90), color=GRAY, width=0.75)

    _textbox(
        page,
        pymupdf.Rect(54, 108, 300, 132),
        "July 16, 2026",
        fontsize=10,
        fontname="helv",
        color=BLACK,
    )

    _textbox(
        page,
        pymupdf.Rect(54, 136, 300, 192),
        "Maria Garcia\n431 Elmwood Avenue\nSpringfield, IL 62704",
        fontsize=10,
        fontname="helv",
        color=BLACK,
        lineheight=1.15,
    )

    _textbox(
        page,
        pymupdf.Rect(54, 198, 460, 222),
        "Re: Account Reference SCB-0042-7719",
        fontsize=10,
        fontname="helv",
        color=BLACK,
    )

    _textbox(
        page,
        pymupdf.Rect(54, 226, 558, 460),
        (
            "Dear Maria Garcia,\n\n"
            "Thank you for being a valued customer of Springfield Community "
            "Bank. This letter confirms that your account referenced above "
            "(SCB-0042-7719) is in good standing.\n\n"
            "If you have any questions, please contact our support team.\n\n"
            "Sincerely,\n\n"
            "Springfield Community Bank"
        ),
        fontsize=10.5,
        fontname="helv",
        color=BLACK,
        lineheight=1.4,
    )

    doc.set_metadata(
        {
            "author": "Maria Garcia",
            "title": "Account Confirmation Letter for Maria Garcia",
            "subject": "Account SCB-0042-7719",
            "keywords": "Maria Garcia, Springfield Community Bank",
        }
    )

    doc.save(path)
    doc.close()
    return path


def make_all(output_dir: Path | str = DEFAULT_OUTPUT_DIR) -> list[Path]:
    """Generate all synthetic fixture PDFs into output_dir; return their paths."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    return [
        _make_fake_w2(out / "fake_w2.pdf"),
        _make_contract(out / "contract.pdf"),
        _make_letter(out / "letter.pdf"),
    ]


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    output_dir = Path(argv[0]) if argv else DEFAULT_OUTPUT_DIR
    paths = make_all(output_dir)
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()

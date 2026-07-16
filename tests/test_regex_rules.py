"""Tests for scrub.detectors.regex_rules.RegexDetector.

Every entity type gets at least one positive and one negative case. All PII
here is synthetic (per ARCHITECTURE.md testing rules); several values are
the plan's designated known-fake fixtures: SSN 458-02-6841, card
4111 1111 1111 1111, routing 021000021.
"""

from __future__ import annotations

from scrub.detectors.regex_rules import RegexDetector
from scrub.types import EntityType

d = RegexDetector()


def types_found(text: str) -> set[str]:
    return {s.entity_type.value for s in d.detect(text)}


def texts_found(text: str, entity_type: EntityType) -> list[str]:
    return [s.text for s in d.detect(text) if s.entity_type == entity_type]


# --------------------------------------------------------------------------
# SSN
# --------------------------------------------------------------------------


def test_ssn_formatted_positive():
    spans = d.detect("Employee SSN: 458-02-6841 on file.")
    assert any(s.entity_type == EntityType.SSN and s.text == "458-02-6841" for s in spans)
    assert all(s.confidence == 1.0 and s.source == "regex" for s in spans)


def test_ssn_bare_with_context_positive():
    spans = texts_found("my social security number is 458026841 here", EntityType.SSN)
    assert "458026841" in spans


def test_ssn_bare_without_context_not_detected():
    assert texts_found("a bare number 458026841 appears with no context", EntityType.SSN) == []


def test_ssn_rejects_000_area():
    assert texts_found("SSN 000-12-3456 invalid", EntityType.SSN) == []


def test_ssn_rejects_666_area():
    assert texts_found("SSN 666-12-3456 invalid", EntityType.SSN) == []


def test_ssn_rejects_9xx_area():
    assert texts_found("SSN 912-12-3456 invalid", EntityType.SSN) == []


def test_ssn_rejects_00_group():
    assert texts_found("SSN 123-00-4567 invalid", EntityType.SSN) == []


def test_ssn_rejects_0000_serial():
    assert texts_found("SSN 123-45-0000 invalid", EntityType.SSN) == []


def test_ssn_valid_area_boundary():
    # 458 is a normal, in-range area — the plan's designated fixture value.
    assert texts_found("SSN 458-02-6841", EntityType.SSN) == ["458-02-6841"]


# --------------------------------------------------------------------------
# ITIN
# --------------------------------------------------------------------------


def test_itin_positive():
    spans = texts_found("ITIN 912-70-1234 on the W-7", EntityType.ITIN)
    assert spans == ["912-70-1234"]


def test_itin_positive_8x_group():
    spans = texts_found("ITIN 987-85-4321", EntityType.ITIN)
    assert spans == ["987-85-4321"]


def test_itin_rejects_group_outside_70_89():
    assert texts_found("ITIN 912-50-1234", EntityType.ITIN) == []


def test_itin_rejects_non_9_area():
    assert texts_found("ITIN 812-70-1234", EntityType.ITIN) == []


# --------------------------------------------------------------------------
# EIN
# --------------------------------------------------------------------------


def test_ein_formatted_valid_prefix_positive():
    spans = texts_found("Our EIN is 12-3456789 for tax purposes", EntityType.EIN)
    assert spans == ["12-3456789"]


def test_ein_formatted_invalid_prefix_rejected():
    assert texts_found("EIN 07-3456789 is not real", EntityType.EIN) == []


def test_ein_bare_requires_context():
    assert texts_found("random number 123456789 with nothing else", EntityType.EIN) == []


def test_ein_bare_with_employer_context_positive():
    spans = texts_found("employer identification number: 123456789", EntityType.EIN)
    assert spans == ["123456789"]


# --------------------------------------------------------------------------
# CREDIT_CARD
# --------------------------------------------------------------------------


def test_credit_card_visa_test_number_positive():
    spans = texts_found("Card on file: 4111 1111 1111 1111", EntityType.CREDIT_CARD)
    assert spans == ["4111 1111 1111 1111"]


def test_credit_card_fails_luhn_rejected():
    assert texts_found("Card: 4111 1111 1111 1112", EntityType.CREDIT_CARD) == []


def test_credit_card_unknown_iin_prefix_rejected():
    # Luhn-valid 16-digit number but starting with a digit ('1') that is not
    # a recognized IIN prefix for any major network.
    assert texts_found("Card: 1234 5678 9012 3452", EntityType.CREDIT_CARD) == []


def test_credit_card_amex_prefix_positive():
    # 378282246310005 is the well-known Amex synthetic test number.
    spans = texts_found("Amex: 378282246310005", EntityType.CREDIT_CARD)
    assert spans == ["378282246310005"]


def test_credit_card_dashed_separators_positive():
    spans = texts_found("Card: 4111-1111-1111-1111", EntityType.CREDIT_CARD)
    assert spans == ["4111-1111-1111-1111"]


# --------------------------------------------------------------------------
# ROUTING_NUMBER
# --------------------------------------------------------------------------


def test_routing_number_with_context_positive():
    spans = texts_found("routing number 021000021", EntityType.ROUTING_NUMBER)
    assert spans == ["021000021"]


def test_routing_number_without_context_not_detected():
    assert texts_found("the number 021000021 shows up here", EntityType.ROUTING_NUMBER) == []


def test_routing_number_bad_checksum_rejected():
    assert texts_found("routing number 021000022", EntityType.ROUTING_NUMBER) == []


def test_routing_number_bare_digits_alone_not_a_routing_number():
    # From the spec: "123456789 alone is NOT a routing number without context"
    # (it also happens to fail the ABA checksum).
    assert texts_found("123456789", EntityType.ROUTING_NUMBER) == []


# --------------------------------------------------------------------------
# BANK_ACCOUNT
# --------------------------------------------------------------------------


def test_bank_account_with_context_positive():
    spans = texts_found("my account number is 88420123456", EntityType.BANK_ACCOUNT)
    assert spans == ["88420123456"]


def test_bank_account_abbreviation_context_positive():
    spans = texts_found("acct: 998877", EntityType.BANK_ACCOUNT)
    assert spans == ["998877"]


def test_bank_account_without_context_not_detected():
    assert texts_found("a number 88420123456 with nothing nearby", EntityType.BANK_ACCOUNT) == []


def test_bank_account_too_short_not_detected():
    assert texts_found("account 12345", EntityType.BANK_ACCOUNT) == []


# --------------------------------------------------------------------------
# PHONE
# --------------------------------------------------------------------------


def test_phone_nanp_parens_positive():
    assert texts_found("call (415) 555-2671 now", EntityType.PHONE) == ["(415) 555-2671"]


def test_phone_nanp_dashed_with_country_code_positive():
    spans = texts_found("call +1-415-555-2671 now", EntityType.PHONE)
    assert spans == ["+1-415-555-2671"]


def test_phone_extension_is_included():
    spans = texts_found("Work phone +1 (628) 555-0117 ext. 204", EntityType.PHONE)
    assert spans == ["+1 (628) 555-0117 ext. 204"]


def test_phone_extension_is_case_insensitive_and_can_be_longer():
    spans = texts_found("Work phone +1 (628) 555-0117 Ext. 1234567", EntityType.PHONE)
    assert spans == ["+1 (628) 555-0117 Ext. 1234567"]


def test_phone_international_positive():
    spans = texts_found("reach us at +44 20 7946 0958", EntityType.PHONE)
    assert spans == ["+44 20 7946 0958"]


def test_phone_invalid_nanp_exchange_leading_zero_rejected():
    # NANP forbids area code / exchange digits starting with 0 or 1.
    assert texts_found("not a phone: 041-055-2671", EntityType.PHONE) == []


def test_phone_short_digit_run_not_detected():
    assert texts_found("id 5551234", EntityType.PHONE) == []


# --------------------------------------------------------------------------
# EMAIL
# --------------------------------------------------------------------------


def test_email_positive():
    spans = texts_found("contact maria.garcia@example.com for details", EntityType.EMAIL)
    assert spans == ["maria.garcia@example.com"]


def test_email_plus_tag_and_subdomain_positive():
    spans = texts_found("send to test.user+tag@mail.example.co.uk", EntityType.EMAIL)
    assert spans == ["test.user+tag@mail.example.co.uk"]


def test_email_missing_tld_not_detected():
    assert texts_found("not an email: foo@bar", EntityType.EMAIL) == []


def test_email_no_at_sign_not_detected():
    assert texts_found("just text with a dot.com in it", EntityType.EMAIL) == []


# --------------------------------------------------------------------------
# URL
# --------------------------------------------------------------------------


def test_url_https_positive():
    assert texts_found("see https://example.com/path?x=1", EntityType.URL) == [
        "https://example.com/path?x=1"
    ]


def test_url_http_positive():
    assert texts_found("see http://example.org", EntityType.URL) == ["http://example.org"]


def test_url_ftp_not_detected():
    assert texts_found("see ftp://example.com/file", EntityType.URL) == []


def test_url_bare_domain_not_detected():
    assert texts_found("visit example.com today", EntityType.URL) == []


# --------------------------------------------------------------------------
# IP_ADDRESS
# --------------------------------------------------------------------------


def test_ipv4_positive():
    assert texts_found("server at 192.168.1.1 responded", EntityType.IP_ADDRESS) == [
        "192.168.1.1"
    ]


def test_ipv4_out_of_range_octet_rejected():
    assert texts_found("bad ip 999.1.1.1 here", EntityType.IP_ADDRESS) == []


def test_ipv4_version_string_permitted_by_design():
    # Spec explicitly allows this false-positive-shaped case: "version
    # strings like 1.2.3.4 handling for IP is fine to flag".
    assert texts_found("version 1.2.3.4 released", EntityType.IP_ADDRESS) == ["1.2.3.4"]


def test_ipv6_full_form_positive():
    spans = texts_found("addr 2001:0db8:85a3:0000:0000:8a2e:0370:7334 in use", EntityType.IP_ADDRESS)
    assert "2001:0db8:85a3:0000:0000:8a2e:0370:7334" in spans


def test_ipv6_compressed_form_positive():
    spans = texts_found("loopback ::1 is local", EntityType.IP_ADDRESS)
    assert "::1" in spans


# --------------------------------------------------------------------------
# MAC_ADDRESS
# --------------------------------------------------------------------------


def test_mac_address_colon_positive():
    assert texts_found("mac 00:1B:44:11:3A:B7 registered", EntityType.MAC_ADDRESS) == [
        "00:1B:44:11:3A:B7"
    ]


def test_mac_address_dash_positive():
    assert texts_found("mac 00-1B-44-11-3A-B7 registered", EntityType.MAC_ADDRESS) == [
        "00-1B-44-11-3A-B7"
    ]


def test_mac_address_too_few_octets_not_detected():
    assert texts_found("mac 00:1B:44:11:3A here", EntityType.MAC_ADDRESS) == []


# --------------------------------------------------------------------------
# Secrets: API_KEY
# --------------------------------------------------------------------------


def test_api_key_aws_positive():
    key = "AKIAABCDEFGHIJKLMNOP"
    assert texts_found(f"aws_key = {key}", EntityType.API_KEY) == [key]


def test_api_key_aws_wrong_prefix_not_detected():
    assert texts_found("aws_key = AKIZABCDEFGHIJKLMNOP", EntityType.API_KEY) == []


def test_api_key_github_pat_positive():
    key = "ghp_" + "a" * 36
    assert key in texts_found(f"token: {key}", EntityType.API_KEY)


def test_api_key_stripe_positive():
    key = "sk_live_" + "b" * 24
    assert key in texts_found(f"stripe secret {key}", EntityType.API_KEY)


def test_api_key_generic_high_entropy_assigned_positive():
    spans = texts_found("api_key = 'Zx91Qk3mPz88vBc2Lw77'", EntityType.API_KEY)
    assert spans == ["Zx91Qk3mPz88vBc2Lw77"]


def test_api_key_generic_low_entropy_not_detected():
    assert (
        texts_found("token = 'aaaaaaaaaaaaaaaaaaaaaaaa'", EntityType.API_KEY) == []
    )


def test_api_key_generic_no_assignment_context_not_detected():
    assert texts_found("just some text Zx91Qk3mPz88vBc2Lw77 wandering by", EntityType.API_KEY) == []


def test_labeled_pdf_fields_are_detected_without_colon_separators():
    text = (
        "Date of birth February 14, 1987 Driver license D123-4567-8901 "
        "Employee ID EMP-739184 Device ID 8F4C2A91-7B3D-4E10-A5C6-91D227E 4B830 "
        "Temporary password Temp!Redact#2026 API token sk_test_51N8xQp7d3L2m9V4c6A1 "
        "Verification PIN 739204"
    )
    spans = d.detect(text)
    values = {s.text for s in spans}
    assert {
        "February 14, 1987",
        "D123-4567-8901",
        "EMP-739184",
        "8F4C2A91-7B3D-4E10-A5C6-91D227E 4B830",
        "Temp!Redact#2026",
        "sk_test_51N8xQp7d3L2m9V4c6A1",
        "739204",
    } <= values


def test_labeled_security_answer_is_detected_before_numbered_section():
    values = texts_found(
        "Security answer Blue Harbor 5. Medical and insurance information",
        EntityType.CREDENTIAL,
    )
    assert values == ["Blue Harbor"]


def test_labeled_security_answer_stops_at_general_following_field():
    values = texts_found(
        "Security answer Blue Harbor Email alice@example.com",
        EntityType.CREDENTIAL,
    )
    assert values == ["Blue Harbor"]


def test_explicit_multiword_and_punctuation_only_passwords_are_detected():
    assert texts_found(
        'Password: "correct horse battery staple"', EntityType.CREDENTIAL
    ) == ["correct horse battery staple"]
    assert texts_found("Temporary password: !!!!!!", EntityType.CREDENTIAL) == ["!!!!!!"]


def test_delimiter_free_temporary_passphrase_is_detected_completely():
    assert texts_found(
        "Temporary password correct horse battery staple", EntityType.CREDENTIAL
    ) == ["correct horse battery staple"]


def test_numeric_labeled_medical_and_employee_ids_are_detected():
    text = "Employee ID 739184 Member ID 123456789 Group number 551204"
    spans = d.detect(text)
    assert {(s.entity_type, s.text) for s in spans} >= {
        (EntityType.GOVERNMENT_ID, "739184"),
        (EntityType.MEDICAL_ID, "123456789"),
        (EntityType.MEDICAL_ID, "551204"),
    }


def test_delimiter_free_field_words_in_prose_are_not_detected():
    samples = (
        "Password should contain eight characters.",
        "Update your username requirements before release.",
        "The clinic remains open on weekends.",
        "Her condition improved after medication adjustment.",
    )
    for text in samples:
        assert not {
            s.entity_type
            for s in d.detect(text)
        } & {EntityType.CREDENTIAL, EntityType.HEALTH_INFORMATION}


def test_labeled_financial_and_medical_values_are_detected():
    text = (
        "Expiration / CVV 12/29 / 123 Annual income $128,500 "
        "Tax filing status Married filing jointly Health plan Fictional Health PPO "
        "Member ID FHP-883920174 Group number GRP-551204 Primary physician Dr. Amara Chen "
        "Medical record no. MRN-002948173 Blood type O positive Condition Type 2 diabetes "
        "Medication Metformin 500 mg twice daily Recent visit June 3, 2026 "
        "Clinic Bayview Family Medicine 6. Free-form case notes"
    )
    values = {s.text for s in d.detect(text)}
    assert {
        "12/29 / 123",
        "$128,500",
        "Married filing jointly",
        "Fictional Health PPO",
        "FHP-883920174",
        "GRP-551204",
        "Dr. Amara Chen",
        "MRN-002948173",
        "O positive",
        "Type 2 diabetes",
        "Metformin 500 mg twice daily",
        "June 3, 2026",
        "Bayview Family Medicine",
    } <= values


def test_structured_street_address_is_detected_deterministically():
    values = texts_found(
        "Office address 100 Market Street, Suite 1200 San Francisco, CA 94105",
        EntityType.STREET_NAME,
    )
    assert values == ["100 Market Street, Suite 1200"]


def test_street_address_does_not_consume_preceding_phone_extension_or_label():
    values = texts_found(
        "Work phone +1 (628) 555-0117 ext. 204 Office address 100 Market Street, Suite 1200",
        EntityType.STREET_NAME,
    )
    assert values == ["100 Market Street, Suite 1200"]


# --------------------------------------------------------------------------
# Secrets: PRIVATE_KEY
# --------------------------------------------------------------------------


def test_private_key_block_positive():
    block = (
        "-----BEGIN RSA PRIVATE KEY-----\n"
        "MIIBOgIBAAJBAK3sQ2r7g9example000fakekeydata\n"
        "-----END RSA PRIVATE KEY-----"
    )
    spans = texts_found(block, EntityType.PRIVATE_KEY)
    assert spans == [block]


def test_private_key_openssh_variant_positive():
    block = (
        "-----BEGIN OPENSSH PRIVATE KEY-----\n"
        "b3BlbnNzaC1rZXktdjEAAAAABG5vbmU=\n"
        "-----END OPENSSH PRIVATE KEY-----"
    )
    assert texts_found(block, EntityType.PRIVATE_KEY) == [block]


def test_private_key_no_footer_not_detected():
    assert (
        texts_found("-----BEGIN RSA PRIVATE KEY----- but never closed", EntityType.PRIVATE_KEY)
        == []
    )


# --------------------------------------------------------------------------
# Secrets: JWT
# --------------------------------------------------------------------------


def test_jwt_positive():
    token = (
        "eyJhbGciOiJIUzI1NiJ9."
        "eyJzdWIiOiIxMjM0NTY3ODkwIn0."
        "dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U"
    )
    assert texts_found(f"Authorization: Bearer {token}", EntityType.JWT) == [token]


def test_jwt_not_starting_with_eyJ_not_detected():
    assert texts_found("looks.like.segments but not a jwt: abc.def.ghi", EntityType.JWT) == []


def test_jwt_only_two_segments_not_detected():
    assert texts_found("eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0", EntityType.JWT) == []


# --------------------------------------------------------------------------
# Overlap handling / longest-match-wins
# --------------------------------------------------------------------------


def test_no_double_report_of_same_range():
    # A dashed EIN-shaped number and a formatted SSN never both fire on the
    # exact same characters — but generic overlap safety: no span in the
    # output should overlap any other span from the same detect() call.
    spans = d.detect(
        "SSN 458-02-6841, EIN 12-3456789, ITIN 912-70-1234, "
        "card 4111 1111 1111 1111, routing number 021000021"
    )
    for i, a in enumerate(spans):
        for b in spans[i + 1 :]:
            assert not a.overlaps(b)


def test_mixed_document_finds_expected_types():
    text = (
        "Name: Maria Garcia\n"
        "SSN: 458-02-6841\n"
        "Email: maria.garcia@example.com\n"
        "Phone: (415) 555-2671\n"
        "Card: 4111 1111 1111 1111\n"
        "Routing number: 021000021, Account number: 88420123456\n"
    )
    found = types_found(text)
    assert found == {
        "SSN",
        "EMAIL",
        "PHONE",
        "CREDIT_CARD",
        "ROUTING_NUMBER",
        "BANK_ACCOUNT",
    }

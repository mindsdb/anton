"""The browser contract fixtures, held against the code that reads and writes them.

tests/fixtures/browser_contract pins what the browser tool sends, what both
browser hosts answer, the turn request's browser block and the cloud turn's
session line. The hosts' tests keep copies of the same files, so every
example here is read by the anton code it describes: answers by the models
the client parses them with, requests by the client calls that send them.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Literal, NamedTuple

import httpx2 as httpx
import pytest
from pydantic import BaseModel, ConfigDict, ValidationError

from anton.core.browser import client as client_module
from anton.core.browser import tool as browser_tool
from anton.core.browser.client import BrowserClient, BrowserError
from anton.core.browser.config import BrowserConfig
from anton.core.browser.models import (
    ActionResult,
    Control,
    Element,
    ErrorBody,
    Frame,
    HeldStep,
    Page,
    SavedTabAnswer,
    SavedTabsAnswer,
    SessionInfo,
    SiteAnswer,
    SitesAnswer,
    TabAnswer,
    TabsAnswer,
    WaitFor,
)
from anton.core.browser.page import (
    FRAME_TEXT_HEADING,
    PAGE_TEXT_HEADING,
    PAYMENT_FRAME_HOSTS,
    SIGN_IN_FRAME_HOSTS,
    TEXT_PER_FRAME,
    format_page,
    human_check,
)

FIXTURES = Path(__file__).parent / "fixtures" / "browser_contract"
ROUTER = "https://app.example.test/browser"
PROFILE = "t-0123456789abcdef01234567"
PROJECT = "p-89abcdef0123456789abcdef"


class ContractExample(BaseModel):
    model_config = ConfigDict(extra="forbid")

    where: str
    kind: Literal["request", "response", "event", "turn_block", "constructor", "classification"]
    status: int | None = None
    refused: bool = False
    body: dict


class ContractFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    description: str
    limits: dict[str, int | str | list[str]] = {}
    examples: dict[str, ContractExample]


def _load(name: str) -> ContractFile:
    return ContractFile.model_validate_json((FIXTURES / f"{name}.json").read_text())


def _examples(kind: str) -> list[tuple[str, str, ContractExample]]:
    found = []
    for path in sorted(FIXTURES.glob("*.json")):
        for name, example in _load(path.stem).examples.items():
            if example.kind == kind:
                found.append((path.stem, name, example))
    return found


def test_the_contract_is_exactly_these_files():
    names = sorted(p.stem for p in FIXTURES.glob("*.json"))
    assert names == sorted([
        "action", "browser_config", "commit_steps", "control", "error", "page", "saved_tabs", "session_event",
        "sessions", "sites", "tabs",
    ])
    for name in names:
        assert _load(name).examples, name


#: The dotted names the descriptions use. Any other word, a period and a
#: word with no space between them is two sentences run together.
_DOTTED_NAMES = {"params.browser", "BrowserConfig.to_dict", "BrowserConfig.from_dict", "limits.saved_tabs_per_project"}


def test_no_description_runs_two_sentences_together():
    for path in sorted(FIXTURES.glob("*.json")):
        joined = set(re.findall(r"[A-Za-z_]+\.[A-Za-z_]+", _load(path.stem).description)) - _DOTTED_NAMES
        assert not joined, path.name


def test_every_host_keeps_this_folder_byte_for_byte_readme_included():
    """The README is one of the files all three copies share byte for byte,
    so it says so, and a host keeps its own notes outside its copy."""
    readme = " ".join((FIXTURES / "README.md").read_text().split())
    assert "Each host's tests keep a copy of these files" not in readme
    assert "Each host's tests keep a byte-for-byte copy of this folder, this README included" in readme
    assert "a host keeps its own notes outside its copy" in readme


def test_the_readme_describes_the_card_field_rule_s_limits_and_examples():
    readme = " ".join((FIXTURES / "README.md").read_text().split())
    assert (
        "a number, such as `saved_tabs_per_project`, a regular expression's source, such as `card_field_name`, which "
        "a host compares with its own pattern's source and reads ignoring case, or a list, such as "
        "`payment_frame_hosts`."
    ) in readme
    assert (
        "`card_field_word_breaks` is a list of regular expressions' sources, which a host compares with its own and "
        "reads case-sensitive."
    ) in readme
    assert "or a field's attribute values, and whether they name the field for a card" in readme
    assert "`value_hidden` on secret fields, `focus` for the focused node, `challenges` for visible check frames" in readme
    assert "`frames` for child frames, and the card-field rule both hosts hold" in readme


# ── answers ────────────────────────────────────────────────────────────────

class _OpenedSession(BaseModel):
    """``POST /sessions``'s answer. Anton discards it, so this pins the one
    shape both hosts give, url always present and null before a page loads."""

    model_config = ConfigDict(extra="forbid")

    session_id: str
    resumed: bool
    url: str | None


#: The model the client reads each route's answer with. A DELETE answer's
#: body is never read.
ANSWER_MODELS: dict[str, type[BaseModel]] = {
    "POST /sessions": _OpenedSession,
    "GET /sessions/{id}/page": Page,
    "POST /sessions/{id}/action": ActionResult,
    "GET /sessions/{id}": SessionInfo,
    "POST /sessions/{id}/control": Control,
    "GET /sessions/{id}/tabs": TabsAnswer,
    "POST /sessions/{id}/tabs": TabAnswer,
    "POST /sessions/{id}/tabs/{tab}/activate": TabAnswer,
    "GET /sessions/{id}/sites": SitesAnswer,
    "POST /sessions/{id}/sites": SiteAnswer,
    "GET /saved-tabs?project={project}": SavedTabsAnswer,
    "POST /saved-tabs": SavedTabAnswer,
}


@pytest.mark.parametrize(
    "example", [e for _, _, e in _examples("response") if e.status is not None and e.status < 400],
    ids=lambda e: e.where,
)
def test_every_answer_parses_into_the_model_the_client_reads_it_with(example):
    created = example.where == "POST /sessions" and not example.body["resumed"]
    assert example.status == (201 if created else 200)
    if example.where.startswith("DELETE "):
        assert example.body == {"ok": True}
        return
    ANSWER_MODELS[example.where].model_validate(example.body)


def test_the_page_fixture_reads_as_the_model_sees_it():
    pages = _load("page").examples
    signin = Page.model_validate(pages["a sign-in page with a check frame"].body)
    cart = Page.model_validate(pages["a page with no check"].body)
    assert human_check(signin) == "Cloudflare"
    assert human_check(cart) == ""
    text = format_page(signin)
    assert '[3] textbox "Password" (password)' in text
    assert '[4] textbox "One-time code" (secret field)' in text
    assert '[2] textbox "Email" value="ada@example.test"' in text
    assert PAGE_TEXT_HEADING in text
    assert [el.id for el in signin.elements if el.secret] == [3, 4]
    assert "… more elements not shown" in format_page(cart)


def test_the_frame_fixture_reads_as_the_model_sees_it():
    """A read frame's controls carry no number, its text sits under its own
    untrusted heading, an unread frame is one line, and all frames together
    show at most FRAMES_TEXT_CHARS of text."""
    page = Page.model_validate(_load("page").examples["a page with a frame"].body)
    text = format_page(page)
    form = text.index('Frame "https://forms.example.test/newsletter" (no numbers: act by pixel):')
    assert text.index('- textbox "Email"', form) < text.index('- button "Subscribe"', form)
    heading = text.index(FRAME_TEXT_HEADING, form)
    assert text.index("Get the weekly digest. Email Subscribe", heading) == heading + len(FRAME_TEXT_HEADING) + 1
    assert 'Frame "https://js.stripe.com/v3/elements-inner-card" (not read)' in text
    assert 'Frame "about:srcdoc" (not read)' in text
    # Frame controls are reached only by position, so none carries a number.
    assert not [line for line in text.splitlines() if line.startswith("- [")]
    assert all(not hasattr(el, "id") for frame in page.frames for el in frame.elements)
    # The sign-in page lists its check frame under frames, not as an element.
    signin = Page.model_validate(_load("page").examples["a sign-in page with a check frame"].body)
    assert [f.read for f in signin.frames] == [False] and len({el.id for el in signin.elements}) == len(signin.elements)

    many = page.model_copy(update={"frames": [
        Frame(url=f"https://forms.example.test/form-{i}", read=True, text=letter * 1000) for i, letter in enumerate("QXZ")
    ]})
    shown = format_page(many)
    assert "Q" * 1000 in shown and "X" * 1000 in shown and "Z" not in shown
    assert 'Frame "https://forms.example.test/form-2" (no numbers: act by pixel):' in shown


def test_the_model_reads_as_much_of_a_frame_as_a_host_sends():
    assert TEXT_PER_FRAME == _load("page").limits["text_per_frame"]


def test_anton_holds_the_frame_hosts_no_host_reads():
    """A frame on one of these hosts is a payment or a sign-in, so anton
    watches no hand-over on a page that lists one (tool.py _wait_for)."""
    limits = _load("page").limits
    assert list(PAYMENT_FRAME_HOSTS) == limits["payment_frame_hosts"]
    assert list(SIGN_IN_FRAME_HOSTS) == limits["sign_in_frame_hosts"]


def test_the_page_fixture_names_every_frame_limit_the_hosts_hold():
    """Each host pins its frame constants against these, the time a host
    gives a read's frames included, and the card-field rule that keeps a
    frame unread and a focused type out of a field."""
    limits = _load("page").limits
    assert set(limits) == {
        "frames_read", "elements_per_frame", "text_per_frame", "frame_min_read_px", "frames_phase_ms",
        "payment_frame_hosts", "sign_in_frame_hosts", "card_field_name", "card_field_exempt", "card_field_conjunction",
        "card_field_payment", "card_field_word_breaks", "card_field_attributes",
    }
    assert limits["frames_phase_ms"] == 2000
    assert "frames_phase_ms" in _load("page").description


#: The words for a card's number after card: number, num, no or # ('Card #').
_NUMBER = r"(numbers?|num|no|#)"
#: The one card-field rule both hosts hold, character for character: a field
#: is named for a card when this, ignoring case, matches its attributes'
#: values joined with spaces, or their split forms joined so, once
#: CARD_FIELD_EXEMPT's matches are taken out of each value on its own (but
#: those CARD_FIELD_CONJUNCTION keeps), unless CARD_FIELD_PAYMENT matches one
#: of them.
CARD_FIELD_NAME = (
    r"(^|[^a-z0-9]|card|cc)(cvc|cvv2?|csc)([^a-z0-9]|$)|card[\s._-]*" + _NUMBER + r"(?![a-z])"
    r"|(^|[^a-z0-9])cc[\s._-]*(numbers?|num|#)(?![a-z])|security[\s._-]*code"
)
#: CARD_FIELD_EXEMPT's qualifiers: the words that name a card that isn't a
#: payment card. A qualifier needs no word boundary, so none may occur in a
#: payment card's name (PAYMENT_CARD_NAMES): club is left out, since Diners
#: Club is a payment card, and benefits, since an EBT, FSA or HSA benefits card
#: is one. An id counts only with no letter or digit right before it, so
#: prepaid's id never does. So does the e of an eGift or e-gift card, so a
#: match starts at that e and the e of 'the gift card' is never taken. resident
#: covers a permanent resident card. green is a green card's: Green Dot's card
#: holds no match (Dot sits between), and a label that names Amex's Green Card
#: holds amex, a payment word. pan is India's PAN card, a tax identity card,
#: and counts only with no letter or digit right before it, so Japan's never
#: does.
_EXEMPT_QUALIFIERS = (
    r"(((?<![a-z0-9])e[\s._-]*)?gift|loyalty|rewards?|library|members?(hip)?|insurance|health|medicare|medicaid"
    r"|student|transit|sim|national[\s._-]*id|identity|identification|(?<![a-z0-9])id|green|residen(t|ce|cy)"
    r"|social[\s._-]*security|aadhaa?r|(?<![a-z0-9])pan|citizen(ship)?|voter)"
)
#: The number of a card that isn't a payment card (a gift, loyalty, library,
#: insurance or identity card), read ignoring case. Each match is replaced with
#: a space in each value and its split form before the join, so a card's CVV,
#: or a card number in another attribute, still names the field.
CARD_FIELD_EXEMPT = _EXEMPT_QUALIFIERS + r"[\s._-]*card[\s._-]*" + _NUMBER + r"(?![a-z])"
#: Names of things that aren't payment cards, which a form joins to an
#: ordinary card's number: an ID document (passport, license or licence,
#: permit, DL, SSN), a contact (name, phone, telephone, mobile, email,
#: username), a checkout code (promo, promotional, coupon, voucher, discount,
#: certificate) or a barcode (a library card's). They are exempt names only in
#: CARD_FIELD_CONJUNCTION, since the exempt pattern holds no number of theirs
#: to take out. Passport can't be a qualifier, since it would take the number
#: of Cash Passport, a prepaid card.
_OTHER_NAMES = (
    r"(passport|licen[cs]e|permit|dl|ssn|name|(tele)?phone|mobile|e[\s._-]*mail|user[\s._-]*name|promo(tion(al)?)?"
    r"|coupon|voucher|discount|certificate|bar[\s._-]*code)"
)
#: A conjunction: or or and with no letter or digit right before or after it,
#: &, +, / or a comma.
_CONJUNCTION = r"((?<![a-z0-9])(or|and)(?![a-z0-9])|[&+/,])"
#: An article or possessive with no letter or digit right before it.
_ARTICLE = r"(?<![a-z0-9])(a|an|the|your|any)"
#: A join: one or more conjunctions, each followed only by spaces or
#: separators ('Gift, loyalty, or', 'and/or'), then at most one article with a
#: space or separator after it.
_JOIN = r"(" + _CONJUNCTION + r"[\s._-]*)+(" + _ARTICLE + r"[\s._-]+)?"
#: An exempt name, which starts a word: one of the exempt pattern's
#: qualifiers or one of _OTHER_NAMES, alone or followed by card, code or
#: address, then by a number (number, num, no or #) or not.
_EXEMPT_NAME = (
    r"(?<![a-z0-9])(" + _EXEMPT_QUALIFIERS + "|" + _OTHER_NAMES + r")([\s._-]*(card|code|address))?"
    r"([\s._-]*" + _NUMBER + r")?"
)
#: Text that holds no conjunction.
_NO_CONJUNCTION = r"((?!" + _CONJUNCTION + r")[\s\S])*"
#: Text that holds no conjunction, no card and no cc with no letter right
#: before or after it (cc as the payment pattern reads it).
_NO_CONJUNCTION_CARD_OR_CC = r"((?!" + _CONJUNCTION + r"|card|(?<![a-z])cc(?![a-z]))[\s\S])*"
#: An identifier's number at the end of a name: a letter or digit, then
#: number, num or no after at least one space or separator ('Account number',
#: 'A-number', 'Order no.'), or # or ID right after it or after spaces or
#: separators ('Account#', 'ICCID'). Num and no need the space, so a word
#: that merely ends in them (Platinum, Casino) names no number.
_IDENTIFIER_NUMBER = r"[a-z0-9]([\s._-]+(numbers?|num|no)|[\s._-]*(#|id))"
#: A name that ends in an exempt name, then the run of conjunctions after it:
#: text that starts with no space or separator and holds no conjunction, whose
#: last words are an exempt name followed only by spaces or separators, or,
#: when it holds neither card nor cc, an identifier's number followed so. A
#: name that holds card or cc keeps the card number ('Card account number or
#: gift card number', 'CC no. or gift card number'). The lookahead finds the
#: exempt name, and a lookahead never gives back what it matched, so each name
#: is read one way and a long list takes time in step with its length.
_NAME_ENDING_IN_AN_EXEMPT_NAME = (
    r"(?![\s._-])(?=(" + _NO_CONJUNCTION + _EXEMPT_NAME + "|" + _NO_CONJUNCTION_CARD_OR_CC + _IDENTIFIER_NUMBER
    + r")[\s._-]*" + _CONJUNCTION + r")" + _NO_CONJUNCTION + r"(" + _CONJUNCTION + r"[\s._-]*)+"
)
#: A character no join holds: neither a conjunction, a space nor a separator,
#: and not the end of or or and with no letter or digit right before it.
_OUTSIDE_A_JOIN = r"[^&+/,\s._-](?<!(?<![a-z0-9])or)(?<!(?<![a-z0-9])and)"
#: The text before an exempt match that keeps it, read ignoring case: text
#: that ends in a join, unless each name its conjunctions separate ends in an
#: exempt name or, holding neither card nor cc, an identifier's number. So
#: 'Card or gift card number', 'Card or gift certificate or gift card number'
#: and 'Card account number or gift card number' keep their card number, and
#: 'Gift or loyalty card number', 'Passport, residence permit or ID card
#: number' and 'Account number or rewards card number' drop it. Python's re
#: takes only a fixed-width lookbehind, so the pattern is anchored at both ends
#: of the text before the match and looks ahead. The join at the end starts at
#: the text's start or right after a character no join holds, then spaces or
#: separators, so it is read from one start and a long run of joins takes
#: time in step with its length too.
CARD_FIELD_CONJUNCTION = (
    r"^(?![\s._-]*(" + _NAME_ENDING_IN_AN_EXEMPT_NAME + r")+(" + _ARTICLE + r"[\s._-]+)?$)([\s\S]*" + _OUTSIDE_A_JOIN
    + r")?[\s._-]*" + _JOIN + r"$"
)
#: A word for a payment card with no letter right before or after it, read
#: ignoring case. When it matches any of a field's values or their split
#: forms, the field skips the exempt step, so 'Visa gift card number' keeps its
#: card number. Payment, bank and charge also name a checkout's section, so
#: they aren't payment words. AmEx and RuPay split to Am Ex and Ru Pay.
CARD_FIELD_PAYMENT = (
    r"(?<![a-z])(credit|debit|prepaid|cc|atm|ebt|hsa|fsa|visa|master[\s._-]*card|am[\s._-]*ex"
    r"|american[\s._-]*express|discover|diners|jcb|union[\s._-]*pay|maestro|ru[\s._-]*pay|interac)(?![a-z])"
)
#: The payment words CARD_FIELD_PAYMENT holds, in the spellings a page uses,
#: and AmEx's and RuPay's split forms.
PAYMENT_WORDS = [
    "credit", "debit", "prepaid", "cc", "atm", "ebt", "hsa", "fsa", "visa", "mastercard", "master card", "amex",
    "am ex", "american express", "discover", "diners", "jcb", "unionpay", "union pay", "maestro", "rupay", "ru pay",
    "interac",
]
#: The word breaks both hosts split the joined value at, applied in order and
#: case-sensitive, each replacing every match with its group and a space
#: (JavaScript's "$1 "): cardNumberInput reads as card Number Input, and
#: CVVCode as CVV Code.
CARD_FIELD_WORD_BREAKS = [r"([a-z0-9])(?=[A-Z])", r"([A-Z])(?=[A-Z][a-z])"]
#: The attributes both hosts read a field's name from, in the order they join them.
CARD_FIELD_ATTRIBUTES = ["name", "id", "aria-label", "placeholder"]
#: A card's number written with # ('Card #'): the card pattern takes # after
#: card and cc, as an exempt name does.
HASH_CARD_NUMBERS = ["Card #", "Card#", "Credit card #", "CC #", "cc#"]
#: A gift, loyalty or library card's number written with #, which the exempt
#: pattern takes as it takes number.
HASH_ORDINARY_CARD_NUMBERS = ["Gift card #", "Loyalty card #", "Library card#"]
#: India's PAN card, a tax identity card, whose number a KYC form asks for.
PAN_CARD_NUMBERS = ["PAN card number", "panCardNumber", "pan_card_number", "PAN Card No."]
#: Names that must stay named for a card (on the page its value is hidden and
#: a type, select or editing key answers 403 password_field, a focused type
#: answers the same, and a frame holding one is never read) and names that
#: must not, so a host types into them.
CARD_NAMED = [
    "cardnumber", "card_number", "card.number", "cardNo", "card_no", "cc_num", "cc-number", "cvc", "CVC", "cvv2",
    "card_cvc", "cardCvc", "cc_cvv", "billing_cvv2", "cvv_code", "csc", "security_code", "Card number", "Security code",
    "cardNumberInput", "CardNumberTextBox", "ctl00_Main_CardNumberTextBox", "ccNumberInput", "creditCardNumberInput",
    "cardNumberField", "cardNumbers", "cvvInput", "txtCVV", "CVVCode", "CCNumber", *HASH_CARD_NUMBERS,
]
#: Payment cards' names, each card-named on both hosts. None holds a match of
#: CARD_FIELD_EXEMPT in either form: its qualifiers need no word boundary, so a
#: qualifier inside one of these would take a payment card for an ordinary
#: card. giftCardCvv is a gift card's CVV, which the exempt pattern never takes.
#: An EBT, HSA or FSA card, and any benefits card, is a payment card.
PAYMENT_CARD_NAMES = [
    "creditCardNumber", "prepaidCardNumber", "storeCardNumber", "fleetCardNumber", "businessCardNumber",
    "debitCardNumber", "cardNumberInput", "giftCardCvv", "paymentCardNumber", "virtualCardNumber",
    "corporateCardNumber", "bankCardNumber", "chargeCardNumber", "purchasingCardNumber", "procurementCardNumber",
    "travelCardNumber", "cc-number", "card_number", "visaCardNumber", "mastercardNumber", "amexCardNumber",
    "discoverCardNumber", "dinersClubCardNumber", "Diners Club card number", "jcbCardNumber", "unionPayCardNumber",
    "maestroCardNumber", "ebtCardNumber", "EBT card number", "HSA card number", "hsaCardNumber", "FSA card number",
    "fsaCardNumber", "FSA/HSA benefits card number", "Benefits card number", "benefitsCardNumber",
    "Transit benefits card number", "Gift card or credit card number", "Green Dot card number",
    "greenDotCardNumber", "Citizens Bank card number", "Direct Express card number", "ATM card number",
    "RuPay card number", "Interac card number",
]
#: An ordinary card's number named for a payment card with no conjunction
#: between, so only CARD_FIELD_PAYMENT keeps it: a Visa gift card is a prepaid
#: Visa, and Amex's Green Card is a charge card. AmEx and RuPay name the card
#: only in their split forms, Am Ex and Ru Pay.
PAYMENT_GIFT_CARD_LABELS = [
    "Visa gift card number", "visaGiftCardNumber", "Mastercard gift card number", "prepaidGiftCardNumber",
    "AmExGiftCardNumber", "RuPayGiftCardNumber", "Amex Green Card number",
]
#: Labels for a field that takes a payment card or an ordinary one, joined by a
#: conjunction and naming the payment card by a payment word, so both
#: CARD_FIELD_PAYMENT and CARD_FIELD_CONJUNCTION keep the card number. Visa
#: keeps a travel form's 'Passport, visa or ID card number' card-named.
COMBINED_PAYMENT_LABELS = [
    "Credit card or gift card number", "Credit/Gift Card Number", "Credit, debit or gift card number",
    "Visa, Mastercard or gift card number", "creditOrGiftCardNumber", "ATM card or gift card number",
    "CC or gift card number", "ccOrGiftCardNumber", "RuPay or gift card number", "Interac or gift card number",
    "Passport, visa or ID card number",
]
#: Labels that join an ordinary card's number by a conjunction (or, and, &, +,
#: / or a comma) to a card that isn't an exempt name and no payment word
#: names, so only CARD_FIELD_CONJUNCTION keeps the card number. A chain keeps
#: it too when its first card isn't exempt, and so does a join that ends in an
#: article. An eGift or e-gift card's match starts at its e, so the text before
#: it ends in the conjunction. cardOrGiftCardNumber names it only in its split
#: form, where Or stands alone.
CONJOINED_CARD_LABELS = [
    "Card or gift card number", "Payment or rewards card number", "Bank card or gift card number",
    "Store card / gift card number", "Card/Gift card number", "cardOrGiftCardNumber", "card_or_gift_card_number",
    "Store and loyalty card number", "Payment & gift card number", "Card + gift card number",
    "Store card, gift card number", "Card, gift or loyalty card number", "Store card, gift card or loyalty card number",
    "Card or eGift card number", "Card or e-gift card number", "Payment card or eGift card number",
    "paymentCardOrEgiftCardNumber", "payment[card_or_egift_card_number]", "cardOrEGiftCardNumber",
    "Card or eGift or loyalty card number", "Card or a gift card number", "Card or an eGift card number",
    "Card or your gift card number", "Card or the gift card number", "Card or any gift card number",
    "cardOrAGiftCardNumber",
    "Card or a gift or loyalty card number", "Card and/or gift card number", "Card or license or ID card number",
    "Card / gift card #",
]
#: A member ID card's number that stays card-named: a letter sits right before
#: its Id, so the exempt pattern takes nothing out of the value itself, which
#: keeps CardNo. Its split form alone, member Id Card No, would be ordinary.
#: Taking an id after a letter would also take prepaid's.
MEMBER_ID_CARD_NO = "memberIdCardNo"
NOT_CARD_NAMED = [
    "name", "email", "acc_number", "account_number", "cardholder_name", "card_type", "cardnotes", "discount_code",
    "promo_code", "postal_code", "recv_count", "Name on card",
    "cardholderName", "cardNotes", "cardNoteText", "cardType", "CardHolder", "Credit or debit card",
]
#: Numbers of cards that aren't payment cards, which a host reads and types
#: into: at least one for each qualifier of CARD_FIELD_EXEMPT, in the casings
#: forms give their fields.
ORDINARY_CARD_NUMBERS = [
    "giftCardNumber", "txtGiftCardNumber", "ctl00_Main_GiftCardNumberTextBox", "loyalty_card_number",
    "Library card number", "membershipCardNumber", "insuranceCardNumber", "nationalIdCardNumber", "rewardsCardNumber",
    "memberCardNo", "health_card_number", "Medicare card number", "medicaidCardNumber", "studentCardNumber",
    "transitCardNum", "simCardNumber", "ID card number", "Identity card number", "Student ID card number",
    "idCardNumber", "id_card_number", "ID Card No.", "Green card number", "greenCardNumber",
    "Permanent resident card number", "Resident card number", "Residence card number", "residenceCardNumber",
    "Residency card number", "Social Security card number", "socialSecurityCardNumber", "Aadhaar card number",
    "Aadhar card number", "aadhaarCardNumber", "Citizen card number", "citizenCardNumber", "Citizenship card number",
    "Voter card number", "voterCardNo", "Voter ID card number", "eGift card number", "e-gift card number",
    "E-Gift Card No.", "eGiftCardNumber", "egift_card_number", "Identification card number",
    "State identification card number", "identificationCardNumber", "identification_card_number",
    *PAN_CARD_NUMBERS, *HASH_ORDINARY_CARD_NUMBERS,
]
#: Ordinary cards' numbers whose labels hold a payment word with a letter
#: right after it (Visalia) or right before it (Rediscover), which isn't one.
PAYMENT_WORD_LOOKALIKES = ["Visalia Library card number", "Rediscover Rewards card number"]
#: Ordinary cards' numbers in a checkout's payment section: payment, bank and
#: charge also name the section, so they aren't payment words.
PAYMENT_SECTION_GIFT_CARDS = [
    "payment[gift_card_number]", "paymentGiftCardNumber", "payment_gift_card_number",
    "checkout-payment-giftcard-number",
]
#: Ordinary cards' numbers joined only to other exempt cards' names, a
#: qualifier alone or followed by card, a number or both, so each is ordinary:
#: by a conjunction or a run of them (an Oxford comma, and/or), after an
#: article too. A conjunction after the match ('and PIN') changes nothing.
EXEMPT_CARD_LISTS = [
    "Gift or loyalty card number", "Gift/Loyalty card number", "Gift card number and PIN",
    "Gift card or library card number", "Loyalty card or gift card number", "Gift card number or loyalty card number",
    "Gift, loyalty or rewards card number", "giftOrLoyaltyCardNumber", "Student ID or library card number",
    "Membership & rewards card number", "Gift, loyalty, or rewards card number", "Gift and/or loyalty card number",
    "Gift card, or loyalty card number", "Gift card, loyalty card, or rewards card number",
    "Student ID number or library card number", "Member number or rewards card number", "Gift card / eGift card number",
    "Gift card or eGift card number", "eGift or loyalty card number", "Gift or a loyalty card number",
    "Enter a gift or loyalty card number", "PAN or Aadhaar card number", "Oregon ID or library card number",
]
#: Ordinary cards' numbers joined to a barcode, a name or an SSN, which end
#: in no identifier's number, so only _OTHER_NAMES makes each ordinary.
BARCODE_NAME_AND_SSN_LISTS = [
    "Barcode or library card number", "Library barcode or library card number", "Bar code or library card number",
    "Name or library card number", "Last name or library card number", "Name, email or loyalty card number",
    "SSN or green card number",
]
#: Ordinary cards' numbers joined only to names of things that aren't payment
#: cards and that the exempt pattern doesn't take (_OTHER_NAMES): an ID
#: document, a contact, a checkout code or a barcode. Each is ordinary.
OTHER_NAME_LISTS = [
    *BARCODE_NAME_AND_SSN_LISTS,
    "Driver's license or ID card number", "Drivers license or ID card number", "Driver license/ID card number",
    "DL or ID card number", "Driver license number or ID card number", "Driver's license / identity card number",
    "Driver license or identification card number", "license_or_id_card_number", "Passport or ID card number",
    "Passport/ID card number", "Passport or identity card number", "Passport or national ID card number",
    "Passport number or national ID card number", "Passport card or ID card number", "Passport no. or ID card no.",
    "Driver license # or ID card number",
    "Residence permit or ID card number", "Driving licence or identity card number",
    "Phone number or loyalty card number", "Mobile number or rewards card number", "Email or loyalty card number",
    "Email address or rewards card number", "Username or library card number", "Promo code or gift card number",
    "Coupon or gift card number", "Gift certificate or gift card number", "Voucher code or gift card number",
    "Telephone or loyalty card number", "Promotional code or gift card number", "Discount code or gift card number",
]
#: Ordinary cards' numbers joined to names that end in an exempt name after
#: other words ('residence permit', "driver's license", 'mobile phone', 'gift
#: certificate'), in any name of the list. Each is ordinary.
WORDED_NAME_LISTS = [
    "Passport, residence permit or ID card number", "Passport number, driver's license or ID card number",
    "Gift card, mobile phone or loyalty card number", "Email, mobile phone number or loyalty card number",
    "Promo code, gift certificate or gift card number", "Gift card or gift certificate or gift card number",
    "Enter your gift, loyalty or rewards card number",
]
#: A payment card named first, then names that end in an exempt name after
#: another word ('gift certificate', 'mobile phone', 'store loyalty'). The
#: first name doesn't end in an exempt name, so each keeps its card number.
CARD_FIRST_LISTS = [
    "Card or gift certificate or gift card number", "Card, gift certificate or gift card number",
    "Card, gift voucher or gift card number", "Card or gift voucher or gift card number",
    "Card, gift voucher, or gift card number", "Bank card, gift certificate or gift card number",
    "Store card, gift certificate or gift card number", "Card, e-voucher or gift card number",
    "Card or mobile phone or gift card number", "Card or store loyalty or gift card number",
    "Card or student ID or library card number", "Card, gift certificate, or eGift card number",
    "cardOrGiftCertificateOrGiftCardNumber",
]
#: Card numbers whose labels hold pan inside a longer word (Japan, Span,
#: Company) or name a payment card's PAN, not a PAN card, so each is
#: card-named.
PAN_LOOKALIKES = ["Japan card number", "Span card number", "Company card number", "Card number (PAN)"]
#: Ordinary cards' numbers after a word that ends in or or and, which isn't a
#: conjunction.
CONJUNCTION_LOOKALIKES = ["Vendor gift card number", "Brand gift card number"]
#: Ordinary cards' numbers joined to an identifier's number (an account,
#: customer, policy, group, order or registration number, an A-number, an
#: ICCID) in a name that holds neither card nor cc. Each is ordinary.
IDENTIFIER_LISTS = [
    "Account number or rewards card number", "Policy number or insurance card number",
    "Customer number or loyalty card number", "Group number or insurance card number", "A-number or green card number",
    "USCIS number or green card number", "Alien registration number or green card number",
    "Voter registration number or voter card number", "Account # or rewards card number",
    "Account# or rewards card number", "Order no. or gift card number", "ICCID or SIM card number",
    "MemberID or rewards card number", "accountNumberOrRewardsCardNumber", "account_number_or_rewards_card_number",
    "Account number, phone or loyalty card number", "Enter your account number or gift card number",
]
#: Card numbers joined to a name that holds card ('Card account number'), to
#: a number word with no word before it, or to a word that merely ends in num
#: or no (Platinum, Casino), so each keeps its card number.
IDENTIFIER_LOOKALIKES = [
    "Card number or gift card number", "Card no. or gift card number", "Card account number or gift card number",
    "Bank card account number or gift card number", "cardAccountNumberOrGiftCardNumber", "Number or gift card number",
    "Platinum or gift card number", "Casino or gift card number", "Card, account number or rewards card number",
]
#: Payment cards' numbers joined to an ordinary card's number, each named
#: with a payment word, so the payment pattern keeps each card number. CC no.
#: and the names that hold card keep it by the conjunction pattern too.
IDENTIFIER_PAYMENT_LABELS = [
    "CC number or gift card number", "CC no. or gift card number", "Credit card number or gift card number",
    "Visa number or gift card number", "Debit card no. or gift card no.",
]
#: Card names the pattern alone misses on the joined value: only its
#: word-split form names them, so a host that skips the split types into them.
#: CVVCode needs the second break.
NAMED_ONLY_WHEN_SPLIT = [
    "cardNumberInput", "CardNumberTextBox", "ctl00_Main_CardNumberTextBox", "ccNumberInput", "creditCardNumberInput",
    "cardNumberField", "cvvInput", "txtCVV", "CVVCode",
]


def _word_split(value: str, breaks: list[str]) -> str:
    """The value with each word break applied in order, case-sensitive, every
    match replaced by its group and a space."""
    for pattern in breaks:
        value = re.sub(pattern, r"\1 ", value)
    return value


class _ValueForms(NamedTuple):
    """One attribute's value and its word-split form."""

    value: str
    split: str


class _NameForms(NamedTuple):
    """A field's two name forms: its values joined, and its split forms joined."""

    joined: str
    split: str


def _value_forms(attributes: dict[str, str]) -> list[_ValueForms]:
    """Each of page.json's card_field_attributes in order, a missing one
    empty, with its word-split form."""
    limits = _load("page").limits
    values = (attributes.get(name, "") for name in limits["card_field_attributes"])
    return [_ValueForms(value, _word_split(value, limits["card_field_word_breaks"])) for value in values]


def _holds_payment_word(pair: _ValueForms) -> bool:
    """Whether page.json's payment pattern, ignoring case, matches the value
    or its split form."""
    payment = _load("page").limits["card_field_payment"]
    return any(re.search(payment, form, re.IGNORECASE) for form in pair)


def _kept_by_a_conjunction(before: str) -> bool:
    """Whether page.json's conjunction pattern, ignoring case, matches the text
    before an exempt match, which keeps the match."""
    return re.search(_load("page").limits["card_field_conjunction"], before, re.IGNORECASE) is not None


def _exempted(
    pair: _ValueForms, keeps: Callable[[str], bool] = _kept_by_a_conjunction, exempt: str | None = None,
) -> _ValueForms:
    """The value and its split form, each with every match of the exempt
    pattern (`exempt`, page.json's), ignoring case, replaced with a space, but
    a match the text before it keeps (`keeps`, page.json's conjunction
    pattern), read in the form as it was before any replacement."""
    exempt = exempt or _load("page").limits["card_field_exempt"]

    def take_out(match: re.Match[str]) -> str:
        return match.group(0) if keeps(match.string[: match.start()]) else " "

    return _ValueForms(*(re.sub(exempt, take_out, form, flags=re.IGNORECASE) for form in pair))


def _joined(forms: list[_ValueForms]) -> _NameForms:
    """The values joined in order with a space, and the split forms joined so."""
    return _NameForms(" ".join(pair.value for pair in forms), " ".join(pair.split for pair in forms))


def _names_a_card(forms: _NameForms) -> bool:
    """Whether page.json's card pattern, ignoring case, matches either name form."""
    name = _load("page").limits["card_field_name"]
    return any(re.search(name, form, re.IGNORECASE) for form in forms)


def _name_forms(attributes: dict[str, str]) -> _NameForms:
    """A field's two name forms as page.json gives them: unless the payment
    pattern matches one of its values or split forms, every exempt match the
    conjunction pattern doesn't keep is replaced with a space in each value and
    each split form on its own; then the values are joined in order with a
    space, and the split forms so."""
    forms = _value_forms(attributes)
    if not any(_holds_payment_word(pair) for pair in forms):
        forms = [_exempted(pair) for pair in forms]
    return _joined(forms)


def _card_named(attributes: dict[str, str]) -> bool:
    """The rule as page.json gives it: the card pattern on either name form."""
    return _names_a_card(_name_forms(attributes))


#: page.json's card rule, word for word.
CARD_RULE = (
    "A field's card rule reads each value of its card_field_attributes (name, id, aria-label and placeholder), a "
    "missing one as empty, and that value's word-split form: each of card_field_word_breaks applied in order, "
    "case-sensitive, replacing every match with its first group and a space ($1 ), so cardNumberInput reads as card "
    "Number Input and CVVCode as CVV Code. Then every match of card_field_exempt, ignoring case, is replaced with a "
    "space in each value and in each split form on its own, so an exempt match never runs from one attribute into "
    "the next, but a match stays when card_field_conjunction, ignoring case, matches the text before it in its form, "
    "read before any replacement. A field skips that step when card_field_payment, ignoring case, matches any of its "
    "values or their split forms. The field's two name forms are then its values joined in that order with a space "
    "and its split forms joined the same way, and the field is card-named when card_field_name, ignoring case, "
    "matches either name form. The exempt pattern matches the number of a card that isn't a payment card, such as a "
    "gift or e-gift, loyalty, library or insurance card or an identity card (an ID, identification, green, "
    "residence, Social Security, Aadhaar, PAN, citizen or voter card). The conjunction pattern matches text that "
    "ends in a join: one or more conjunctions (or, and, &, +, / or a comma), each followed only by spaces or "
    "separators (. _ -), then at most one article (a, an, the, your or any) followed by spaces or separators. The "
    "text's names are its text before its first join and between one join and the next. A name is exempt when it "
    "ends in an exempt name, or holds neither card nor cc and ends in an identifier's number, followed only by "
    "spaces or separators, and the pattern doesn't match when each name is exempt. An exempt name is one of the "
    "exempt pattern's qualifiers or the name of an ID document (passport, license, licence, permit, DL or SSN), a "
    "contact (name, phone, telephone, mobile, email or username), a checkout code (promo, promotional, coupon, "
    "voucher, discount or certificate) or a barcode, alone or followed by card, code or address, and then by a "
    "number (number, num, no or #) or not. An identifier's number is a letter or digit, then number, num or no "
    "after one or more spaces or separators, or # or ID right after it or after spaces or separators, as in "
    "'Account number', 'A-number', 'Order no.', 'Account#' and 'ICCID'. Or and and count only with no letter or "
    "digit right before or after them, an article and an exempt name only with none right before them, and cc "
    "only with no letter right before or after it. So the card number stays when a conjunction before it joins it "
    "to a name that isn't exempt: it stays in 'Card or gift card number', 'Card or eGift card number', 'Card or a "
    "gift card number', 'Payment or rewards card number', 'Card, gift or loyalty card number', 'Card or gift "
    "certificate or gift card number' and 'Card account number or gift card number', and not in 'Gift or loyalty "
    "card number', 'Gift, loyalty, or rewards card number', 'Gift card or library card number', 'Driver license or "
    "ID card number', 'Passport, residence permit or ID card number', 'Phone number or loyalty card number', "
    "'Account number or rewards card number' or 'Gift card number and PIN'. "
    "The payment pattern matches a word for a payment card with no letter right before or after it: "
    "credit, debit, prepaid, CC, ATM, EBT, HSA or FSA, or one of the card networks Visa, Mastercard, Amex, American "
    "Express, Discover, Diners, JCB, UnionPay, Maestro, RuPay and Interac. Payment, bank and charge aren't payment "
    "words, since they also name a checkout's section. So a field that names a card number is ordinary only when "
    "the exempt pattern takes out each card number its values name and no value holds a payment word: "
    "giftCardNumber, payment[gift_card_number], 'Library card number', 'Gift card #', 'ID card number', 'PAN card "
    "number' and 'Green card number' are ordinary fields, while giftCardCvv, 'Card #', a benefits card's number, "
    "'Japan card number', 'Credit card or gift card number', "
    "creditOrGiftCardNumber, 'Visa gift card number', 'Green Dot card number', a field named giftCardNumber with the "
    "placeholder 'Card number', one named cardNumber with the aria-label 'Gift card number' and one with the id "
    "member and the placeholder 'Card number' stay card-named. A select labeled 'Credit or debit card' names no card "
    "number, so it isn't card-named."
)


def _attribute_example(name: str, card_named: bool, attributes: dict[str, str]) -> None:
    """The page.json example `name` is a field with these attributes that
    the card rule names for a card, or not, as card_named says."""
    example = _load("page").examples[name]
    assert (example.kind, example.where) == ("classification", "a field's name, id, aria-label or placeholder")
    assert example.body == {"card_named": card_named, "attributes": attributes}
    assert _card_named(attributes) is card_named, name


def test_the_page_fixture_pins_the_one_card_field_rule_both_hosts_hold():
    """Both hosts hide a card-named field's value, refuse a type into it and
    never read a frame that holds one, by this one rule, so a name gets the
    same answer on either host. Each value is named for a card, or not, in
    any of the attributes alone, and one field is named only by two
    attributes together, which pins the join. The camelCase names an ASP.NET
    or React form gives its card fields are named only by the word split.
    A benefits card is a payment card, so benefitsCardNumber is card-named."""
    page = _load("page")
    assert page.limits["card_field_name"] == CARD_FIELD_NAME
    assert page.limits["card_field_exempt"] == CARD_FIELD_EXEMPT
    assert page.limits["card_field_conjunction"] == CARD_FIELD_CONJUNCTION
    assert page.limits["card_field_payment"] == CARD_FIELD_PAYMENT
    assert page.limits["card_field_word_breaks"] == CARD_FIELD_WORD_BREAKS
    assert page.limits["card_field_attributes"] == CARD_FIELD_ATTRIBUTES
    for gone in (
        "(name, id, aria-label and placeholder) joined in that order with a space, a missing one as empty.",
        "A field is card-named when card_field_name, ignoring case, matches the values of",
        "In each form, every match of card_field_exempt, ignoring case, is replaced with a space",
        "such as a gift, loyalty, library or insurance card, so giftCardNumber",
        "credit, debit, prepaid, payment, bank, charge, EBT, HSA or FSA",
        "where it joins an ordinary card to one the exempt pattern doesn't name",
        "An exempt card's name is one of the exempt pattern's qualifiers, alone or followed by card or a card number",
        "It doesn't match text that reads as text that doesn't end in a conjunction or an article",
        "Or, and, an article and an exempt name count only with no letter or digit right before them.",
        "and the pattern doesn't match when each name ends in an exempt name followed only by spaces or separators.",
    ):
        assert gone not in page.description, gone
    assert CARD_RULE in page.description
    for value in NAMED_ONLY_WHEN_SPLIT:
        assert re.search(CARD_FIELD_NAME, value, re.IGNORECASE) is None, value
        assert re.search(CARD_FIELD_NAME, _word_split(value, CARD_FIELD_WORD_BREAKS), re.IGNORECASE), value
    assert re.search(CARD_FIELD_NAME, _word_split("CVVCode", CARD_FIELD_WORD_BREAKS[:1]), re.IGNORECASE) is None
    named = page.examples["values that name a field for a card"]
    unnamed = page.examples["values that don't name a field for a card"]
    for example, card_named, required in (
        (named, True, CARD_NAMED + PAYMENT_CARD_NAMES + PAYMENT_GIFT_CARD_LABELS + COMBINED_PAYMENT_LABELS
         + CONJOINED_CARD_LABELS + CARD_FIRST_LISTS + PAN_LOOKALIKES + IDENTIFIER_LOOKALIKES + IDENTIFIER_PAYMENT_LABELS
         + [MEMBER_ID_CARD_NO]),
        (unnamed, False, NOT_CARD_NAMED + ORDINARY_CARD_NUMBERS + PAYMENT_WORD_LOOKALIKES + PAYMENT_SECTION_GIFT_CARDS
         + EXEMPT_CARD_LISTS + OTHER_NAME_LISTS + WORDED_NAME_LISTS + CONJUNCTION_LOOKALIKES + IDENTIFIER_LISTS),
    ):
        assert (example.kind, example.where) == ("classification", "a field's name, id, aria-label or placeholder")
        assert example.body["card_named"] is card_named
        values = example.body["values"]
        assert set(required) <= set(values) and len(set(values)) == len(values)
        for value in values:
            for attribute in CARD_FIELD_ATTRIBUTES:
                assert _card_named({attribute: value}) is card_named, (attribute, value)
    assert "benefitsCardNumber" in named.body["values"] and "benefitsCardNumber" not in unnamed.body["values"]
    together = page.examples["a field named for a card only by two of its attributes together"]
    assert together.body["card_named"] is True
    attributes = together.body["attributes"]
    assert _card_named(attributes)
    assert not any(_card_named({name: value}) for name, value in attributes.items())


def test_the_exempt_pattern_takes_out_only_an_ordinary_card_s_number_and_never_a_payment_card_s_name():
    """A gift, loyalty, library, insurance or identity card's number is an
    ordinary field on both hosts: readable, and typed into. The exempt
    pattern's qualifiers need no word boundary (txtGiftCardNumber), so each
    payment name in page.json is checked for a match in both forms: with club
    among the qualifiers, Diners Club's names failed here, and with benefits,
    the benefits cards'. A host that skips the exempt step reads
    giftCardNumber as card-named; one that skips it on the split form reads
    ctl00_Main_GiftCardNumberTextBox so; one that drops the field when the
    exempt pattern matches anywhere, instead of taking the match out, misses
    cardNumber beside the label 'Gift card number'. An id counts only with no
    letter or digit right before it: counting every id reads prepaidCardNumber
    as ordinary."""
    page = _load("page")
    exempt = page.limits["card_field_exempt"]
    named_despite_an_exempt_match = (
        PAYMENT_GIFT_CARD_LABELS + COMBINED_PAYMENT_LABELS + CONJOINED_CARD_LABELS + CARD_FIRST_LISTS
        + IDENTIFIER_LOOKALIKES + IDENTIFIER_PAYMENT_LABELS + [MEMBER_ID_CARD_NO]
    )
    for value in page.examples["values that name a field for a card"].body["values"]:
        if value in named_despite_an_exempt_match:
            continue
        for attribute in CARD_FIELD_ATTRIBUTES:
            for form in (form for pair in _value_forms({attribute: value}) for form in pair):
                assert re.search(exempt, form, re.IGNORECASE) is None, (attribute, value, form)
    for value in ORDINARY_CARD_NUMBERS:
        assert any(re.search(exempt, form, re.IGNORECASE) for form in _value_forms({"name": value})[0]), value
    assert re.search(CARD_FIELD_NAME, "giftCardNumber", re.IGNORECASE)
    value, split = _value_forms({"name": "ctl00_Main_GiftCardNumberTextBox"})[0]
    assert re.search(exempt, value, re.IGNORECASE) is None and re.search(CARD_FIELD_NAME, split, re.IGNORECASE)

    beside = page.examples["a card field whose aria-label names a gift card"]
    assert (beside.kind, beside.where) == ("classification", "a field's name, id, aria-label or placeholder")
    assert beside.body == {
        "card_named": True, "attributes": {"name": "cardNumber", "aria-label": "Gift card number"},
    }
    assert _card_named(beside.body["attributes"])
    assert _card_named({"name": "cardNumber"}) and not _card_named({"aria-label": "Gift card number"})
    beside_forms = _value_forms(beside.body["attributes"])
    assert any(re.search(exempt, form, re.IGNORECASE) for pair in beside_forms for form in pair)

    unbounded = exempt.replace("|(?<![a-z0-9])id|", "|id|")
    assert unbounded != exempt
    left = re.sub(unbounded, " ", "prepaidCardNumber", flags=re.IGNORECASE)
    assert re.search(CARD_FIELD_NAME, left, re.IGNORECASE) is None
    assert re.search(exempt, "prepaidCardNumber", re.IGNORECASE) is None
    value, split = _value_forms({"name": MEMBER_ID_CARD_NO})[0]
    assert re.search(exempt, value, re.IGNORECASE) is None and re.search(CARD_FIELD_NAME, value, re.IGNORECASE)
    assert re.search(CARD_FIELD_NAME, re.sub(exempt, " ", split, flags=re.IGNORECASE), re.IGNORECASE) is None


def test_the_exempt_step_runs_on_each_attribute_value_on_its_own():
    """An exempt match never runs from one attribute into the next, so a
    field is ordinary only when each card number its attributes name is an
    ordinary card's. A host that runs the exempt step on the joined values
    reads a field with the id member and the placeholder 'Card number' as
    ordinary; each attribute alone is ordinary but for the placeholder."""
    _attribute_example(
        "a field whose id is member and whose placeholder names a card number",
        True, {"id": "member", "placeholder": "Card number"},
    )
    _attribute_example(
        "a gift card's field whose placeholder names a card number",
        True, {"name": "giftCardNumber", "placeholder": "Card number"},
    )
    _attribute_example(
        "a gift card's number in every attribute",
        False, {
            "name": "gift_card_number", "id": "giftCardNumber", "aria-label": "Gift card number",
            "placeholder": "Gift card number",
        },
    )
    assert not _card_named({"id": "member"}) and not _card_named({"name": "giftCardNumber"})
    assert _card_named({"placeholder": "Card number"})

    def exempt_after_the_join(attributes: dict[str, str]) -> bool:
        exempt = _load("page").limits["card_field_exempt"]
        joined = _joined(_value_forms(attributes))
        return _names_a_card(_NameForms(*(re.sub(exempt, " ", form, flags=re.IGNORECASE) for form in joined)))

    assert not exempt_after_the_join({"id": "member", "placeholder": "Card number"})


def test_a_payment_word_in_any_attribute_keeps_the_field_s_card_number():
    """A label that names a payment card over an ordinary card's number names
    a payment card, so a payment word anywhere in the field skips the exempt
    step: in any attribute, in its split form (visaGiftCardNumber reads visa
    only once split), and only with no letter right before or after it. A host
    that skips the payment check reads 'Visa gift card number' as ordinary;
    one that checks only the values reads visaGiftCardNumber so; one that skips
    the exempt step only in the attribute that holds the payment word reads a
    gift card's id beside the label 'Credit or gift card' so; and one that
    matches the word inside a longer one reads 'Visalia Library card number' as
    card-named. A combined label with a conjunction keeps its card number by
    the payment word and by the conjunction alike, so only a host that skips
    both reads it as ordinary. A payment word alone names no card number."""
    limits = _load("page").limits
    payment = limits["card_field_payment"]
    for word in PAYMENT_WORDS:
        assert re.fullmatch(payment, word, re.IGNORECASE), word
        for around in (f"9{word}_", f"{word.upper()} card", f"{word.title()}/Gift"):
            assert re.search(payment, around, re.IGNORECASE), around
        for around in (f"x{word}", f"{word}x", f"X{word.upper()}", f"{word.upper()}S"):
            assert re.search(payment, around, re.IGNORECASE) is None, around

    def without_payment_check(attributes: dict[str, str]) -> bool:
        return _names_a_card(_joined([_exempted(pair) for pair in _value_forms(attributes)]))

    def payment_check_on_values_only(attributes: dict[str, str]) -> bool:
        forms = _value_forms(attributes)
        if any(re.search(payment, pair.value, re.IGNORECASE) for pair in forms):
            return _names_a_card(_joined(forms))
        return without_payment_check(attributes)

    def payment_check_per_attribute(attributes: dict[str, str]) -> bool:
        forms = _value_forms(attributes)
        return _names_a_card(_joined([pair if _holds_payment_word(pair) else _exempted(pair) for pair in forms]))

    def without_payment_check_or_conjunction(attributes: dict[str, str]) -> bool:
        forms = _value_forms(attributes)
        return _names_a_card(_joined([_exempted(pair, keeps=lambda before: False) for pair in forms]))

    for label in PAYMENT_GIFT_CARD_LABELS:
        for attribute in CARD_FIELD_ATTRIBUTES:
            assert _card_named({attribute: label}), (attribute, label)
            assert not without_payment_check({attribute: label}), (attribute, label)
    for label in COMBINED_PAYMENT_LABELS:
        for attribute in CARD_FIELD_ATTRIBUTES:
            assert _card_named({attribute: label}), (attribute, label)
            assert any(_holds_payment_word(pair) for pair in _value_forms({attribute: label})), (attribute, label)
            assert without_payment_check({attribute: label}), (attribute, label)
            assert not without_payment_check_or_conjunction({attribute: label}), (attribute, label)
    for value in ("visaGiftCardNumber", "AmExGiftCardNumber", "RuPayGiftCardNumber"):
        assert re.search(payment, value, re.IGNORECASE) is None, value
        assert not payment_check_on_values_only({"name": value}), value
    _attribute_example(
        "a gift card's id on a field labeled for a credit or gift card",
        True, {"id": "giftCardNumber", "aria-label": "Credit or gift card"},
    )
    assert not _card_named({"id": "giftCardNumber"}) and not _card_named({"aria-label": "Credit or gift card"})
    assert not payment_check_per_attribute({"id": "giftCardNumber", "aria-label": "Credit or gift card"})
    _attribute_example(
        "a payment-type select labeled for a credit or debit card",
        False, {"name": "payment_type", "aria-label": "Credit or debit card"},
    )
    assert _holds_payment_word(_value_forms({"aria-label": "Credit or debit card"})[2])

    inside_words = payment.removeprefix("(?<![a-z])").removesuffix("(?![a-z])")
    assert inside_words != payment

    def payment_check_inside_words(attributes: dict[str, str]) -> bool:
        forms = _value_forms(attributes)
        if any(re.search(inside_words, form, re.IGNORECASE) for pair in forms for form in pair):
            return _names_a_card(_joined(forms))
        return without_payment_check(attributes)

    for value in PAYMENT_WORD_LOOKALIKES:
        assert payment_check_inside_words({"aria-label": value}), value


def test_the_card_rule_lists_each_payment_word_its_pattern_holds_and_no_other():
    """Both hosts' docs copy the card rule's sentence, so it lists the words
    card_field_payment holds, each card network by name. A sentence that says
    any card network's name tells a reader 'Elo gift card number' is
    card-named, while the rule reads it as ordinary. Each listed word matches
    exactly one of the pattern's words, and each of the pattern's words is
    listed exactly once."""
    page = _load("page")
    payment = page.limits["card_field_payment"]
    lead = "The payment pattern matches a word for a payment card with no letter right before or after it: "
    clause = page.description.split(lead, 1)[1].split(". ", 1)[0]
    listed = re.split(r", or one of the card networks |, | or | and ", clause)
    held = re.fullmatch(r"\(\?<!\[a-z\]\)\((.*)\)\(\?!\[a-z\]\)", payment).group(1).split("|")
    for word in listed:
        assert re.fullmatch(payment, word, re.IGNORECASE), word
        assert len([one for one in held if re.fullmatch(one, word, re.IGNORECASE)]) == 1, word
    for one in held:
        assert len([word for word in listed if re.fullmatch(one, word, re.IGNORECASE)]) == 1, one
    assert not _card_named({"placeholder": "Elo gift card number"})


def _kept_as_worded(before: str) -> bool:
    """The conjunction rule as page.json words it, read plainly: the text
    before an exempt match keeps it when its last join ends it, unless each of
    its names (the text before its first join and between one join and the
    next) ends in an exempt name, or holds neither card nor cc and ends in an
    identifier's number, followed only by spaces or separators."""
    joins = list(re.finditer(_JOIN, before, re.IGNORECASE))
    if not joins or joins[-1].end() != len(before):
        return False
    ends_in_an_exempt_name = re.compile(_EXEMPT_NAME + r"[\s._-]*$", re.IGNORECASE)
    ends_in_an_identifier_number = re.compile(_IDENTIFIER_NUMBER + r"[\s._-]*$", re.IGNORECASE)

    def exempt(start: int, end: int) -> bool:
        if ends_in_an_exempt_name.search(before, start, end):
            return True
        name = before[start:end].lower()
        holds_card_or_cc = "card" in name or re.search(r"(?<![a-z])cc(?![a-z])", name) is not None
        return not holds_card_or_cc and bool(ends_in_an_identifier_number.search(before, start, end))

    names = zip([0] + [join.end() for join in joins], [join.start() for join in joins])
    return not all(exempt(start, end) for start, end in names)


def _card_named_keeping(attributes: dict[str, str], keeps: Callable[[str], bool], exempt: str | None = None) -> bool:
    """The rule as page.json gives it, but with `keeps` in place of the
    conjunction pattern, and `exempt`, when given, in place of the exempt
    pattern."""
    forms = _value_forms(attributes)
    if any(_holds_payment_word(pair) for pair in forms):
        return _names_a_card(_joined(forms))
    return _names_a_card(_joined([_exempted(pair, keeps=keeps, exempt=exempt) for pair in forms]))


def _keeping_by(conjunction: str) -> Callable[[str], bool]:
    """A conjunction check by `conjunction`, read ignoring case, in place of
    page.json's."""
    return lambda before: re.search(conjunction, before, re.IGNORECASE) is not None


def test_a_conjunction_keeps_an_ordinary_card_s_number_joined_to_a_name_that_isn_t_an_exempt_name():
    """A label that joins a gift card's number to another name by or, and, &,
    +, / or a comma takes that other thing too, so unless it is an exempt name
    as well, the exempt match stays and the field is card-named: 'Card or gift
    card number' and 'Payment or rewards card number' name a payment card with
    no payment word. The conjunction pattern reads the text before each exempt
    match in its form, as the form was before any replacement, and holds the
    exempt pattern's own qualifiers. A host that skips it reads each conjoined
    label as ordinary; one that keeps a match after any conjunction reads 'Gift
    or loyalty card number' as card-named; one that looks only at the word
    right before the last conjunction reads 'Card, gift or loyalty card number'
    as ordinary and 'Gift card or library card number' as card-named; one that
    reads or and and at the end of a longer word reads 'Vendor gift card
    number' as card-named, and one that reads them at its start reads 'Oregon
    ID or library card number' so, taking the Or of Oregon for a conjunction;
    and one that reads the text before a match after the earlier matches are
    taken out reads 'Gift card number or loyalty card number' so."""
    limits = _load("page").limits
    conjunction = limits["card_field_conjunction"]
    assert CARD_FIELD_EXEMPT.startswith(_EXEMPT_QUALIFIERS)
    assert conjunction.count(_EXEMPT_QUALIFIERS) == 1 and limits["card_field_exempt"].startswith(_EXEMPT_QUALIFIERS)
    for label in CONJOINED_CARD_LABELS:
        for attribute in CARD_FIELD_ATTRIBUTES:
            forms = _value_forms({attribute: label})
            assert _card_named({attribute: label}), (attribute, label)
            assert not any(_holds_payment_word(pair) for pair in forms), (attribute, label)
            assert not _card_named_keeping({attribute: label}, lambda before: False), (attribute, label)
    for word in ("or", "and", "&", "+", "/", ","):
        assert _kept_by_a_conjunction(f"Card {word} ") and _kept_by_a_conjunction(f"Card_{word}_"), word
        assert not _kept_by_a_conjunction(f"Gift {word} ") and not _kept_by_a_conjunction(f"Gift card {word} "), word
    for word in ("&", "+", "/", ","):
        assert _kept_by_a_conjunction(f"Card{word}") and not _kept_by_a_conjunction(f"Gift{word}"), word
    for word in ("or", "and"):
        assert not _kept_by_a_conjunction(f"Card{word} "), word
    assert not _kept_by_a_conjunction("cardOr") and _kept_by_a_conjunction("card Or ")
    assert not _kept_by_a_conjunction("") and not _kept_by_a_conjunction("Enter the ")

    after_any_conjunction = re.compile(_CONJUNCTION + r"[\s._-]*$", re.IGNORECASE).search
    assert _card_named_keeping({"aria-label": "Gift or loyalty card number"}, after_any_conjunction)
    last_word_only = re.compile(
        r"^(?![\s\S]*" + _EXEMPT_QUALIFIERS + r"[\s._-]*" + _CONJUNCTION + r"[\s._-]*$)[\s\S]*" + _CONJUNCTION
        + r"[\s._-]*$", re.IGNORECASE,
    ).search
    assert not _card_named_keeping({"aria-label": "Card, gift or loyalty card number"}, last_word_only)
    assert _card_named_keeping({"aria-label": "Gift card or library card number"}, last_word_only)
    inside_words = conjunction.replace("(?<![a-z0-9])(or|and)", "(or|and)")
    assert inside_words != conjunction
    for label in CONJUNCTION_LOOKALIKES:
        assert _card_named_keeping({"aria-label": label}, _keeping_by(inside_words))
    word_starts = conjunction.replace("(or|and)(?![a-z0-9])", "(or|and)")
    assert word_starts != conjunction
    assert not _card_named({"aria-label": "Oregon ID or library card number"})
    assert _card_named_keeping({"aria-label": "Oregon ID or library card number"}, _keeping_by(word_starts))

    def taken_out_in_turn(form: str) -> str:
        exempt = re.compile(limits["card_field_exempt"], re.IGNORECASE)
        start = 0
        while match := exempt.search(form, start):
            if _kept_by_a_conjunction(form[: match.start()]):
                start = match.end()
            else:
                form, start = form[: match.start()] + " " + form[match.end():], match.start() + 1
        return form

    label = "Gift card number or loyalty card number"
    assert not _card_named({"aria-label": label})
    assert _names_a_card(_NameForms(*(taken_out_in_turn(form) for form in _value_forms({"aria-label": label})[2])))

    exempt = re.compile(limits["card_field_exempt"], re.IGNORECASE)
    befores = {
        "Card, gift card or ", "Card or national ID or ", "xgift or ", "Gift card no. & ", "gift_or_", "Gift or\n",
        "Card or a ", "Card or a gift or ", "Gift or a ", "Enter a gift or ", "Gift, loyalty, or ", "Gift and/or ",
        "Card or eGift or ", "Card or e", "Driver's license or ", "Passport number or ", "Card or license or ",
        "Phone number or ", "Promo code or ", "Visa or ", "the ", "a ", "Card, or ", "Card or an", "Card or any ",
        "Account number or ", "A-number or ", "ICCID or ", "Account# or ", "Order no. or ", "Card account number or ",
        "Number or ", "Platinum or ", "Casino or ", "Barcode or ", "Last name or ", "SSN or ", "Cardholder ID or ",
        "Discard number or ", "x number, card or ", ", " * 50 + "x", "or " * 50 + "x", "x and/or " * 20,
        "CC no. or ", "cc_no_or_", "ccid or ", "Acct no. or ", "Visa number or ", "ICC ID or ",
    }
    for example in _load("page").examples.values():
        if example.kind != "classification":
            continue
        fields = [{"name": value} for value in example.body.get("values", [])] + [example.body.get("attributes", {})]
        for field in fields:
            for form in (form for pair in _value_forms(field) for form in pair):
                befores |= {form[: match.start()] for match in exempt.finditer(form)}
    assert len(befores) > 20
    for before in befores:
        assert _kept_by_a_conjunction(before) is _kept_as_worded(before), before


def test_an_e_gift_card_s_match_starts_at_its_e_so_the_conjunction_before_it_keeps_the_card_number():
    """An eGift or e-gift card is a gift card, so its number is ordinary on
    both hosts. The exempt match starts at its e, so in 'Card or eGift card
    number' the text before the match ends in the conjunction, which keeps the
    card number: the field takes a payment card's number too. A host whose
    gift qualifier takes no e starts the match at Gift, reads the text before
    it as 'Card or e' and the field as ordinary. The e counts only with no
    letter or digit right before it: a host that takes any e starts the match
    in 'Card or the gift card number' at the e of the, and reads the field as
    ordinary. An exempt name in the conjunction pattern starts a word too, so
    a qualifier inside a longer word names no exempt card: 'Cardmember or gift
    card number' keeps its card number, where a host whose names needn't start
    a word reads Cardmember as ending in the exempt name member, and the field
    as ordinary. The cost is a qualifier glued to the word before it ('MyGift
    or loyalty card number'), which reads as card-named."""
    limits = _load("page").limits
    exempt, conjunction = limits["card_field_exempt"], limits["card_field_conjunction"]
    for value in ("eGift card number", "e-gift card number", "E-Gift Card No.", "Enter your eGift card number"):
        assert not _card_named({"aria-label": value}), value
    for label in ("Card or eGift card number", "Card or e-gift card number", "Payment card or eGift card number"):
        assert _card_named({"aria-label": label}), label
        assert label[: re.search(exempt, label, re.IGNORECASE).start()].endswith(" or "), label
    split = _value_forms({"id": "paymentCardOrEgiftCardNumber"})[1].split
    assert split[: re.search(exempt, split, re.IGNORECASE).start()] == "payment Card Or "

    no_e = exempt.replace(r"((?<![a-z0-9])e[\s._-]*)?gift", "gift")
    any_e = exempt.replace(r"((?<![a-z0-9])e[\s._-]*)?gift", r"(e[\s._-]*)?gift")
    assert no_e != exempt and any_e != exempt
    for label in ("Card or eGift card number", "Card or e-gift card number", "paymentCardOrEgiftCardNumber"):
        assert not _card_named_keeping({"aria-label": label}, _kept_by_a_conjunction, exempt=no_e), label
    assert _card_named({"aria-label": "Card or the gift card number"})
    assert not _card_named_keeping({"aria-label": "Card or the gift card number"}, _kept_by_a_conjunction, exempt=any_e)

    mid_word = conjunction.replace(r"(?<![a-z0-9])(" + _EXEMPT_QUALIFIERS, "(" + _EXEMPT_QUALIFIERS)
    assert mid_word != conjunction
    for label in ("Cardmember or gift card number", "MyGift or loyalty card number"):
        assert _card_named({"aria-label": label}), label
        assert not _card_named_keeping({"aria-label": label}, _keeping_by(mid_word)), label
    for label in ("Card or eGift or loyalty card number", "Card, eGift or loyalty card number"):
        assert _card_named({"aria-label": label}), label
    assert not _card_named({"aria-label": "eGift or loyalty card number"})


def test_a_join_holds_a_run_of_conjunctions_and_may_end_in_an_article():
    """A join is one or more conjunctions, each followed only by spaces or
    separators, then at most one article (a, an, the, your or any). So 'Gift,
    loyalty, or rewards card number', with an Oxford comma, and 'Gift and/or
    loyalty card number' join only exempt names and are ordinary, where a host
    that takes one conjunction between names reads them as card-named. 'Card or a
    gift card number' keeps its card number, where a host that takes no
    article reads it as ordinary. Before an exempt name, an article is a word
    like any other, so 'Enter a gift or loyalty card number' stays ordinary.
    'Card or a gift or loyalty card number' keeps its card number, since its
    first name, Card, doesn't end in an exempt name."""
    runs = ("Gift, loyalty, or rewards card number", "Gift and/or loyalty card number")
    articles = ("Card or a gift card number", "Card or an eGift card number", "Card or your gift card number")
    for label in runs:
        assert not _card_named({"aria-label": label}), label
    for label in articles:
        assert _card_named({"aria-label": label}), label
    assert _card_named({"aria-label": "Card or a gift or loyalty card number"})
    assert not _card_named({"aria-label": "Enter a gift or loyalty card number"})
    for before in ("Visa or ", "Enter a ", "Card or another "):
        assert _kept_by_a_conjunction(before) is (before == "Visa or "), before

    conjunction = _load("page").limits["card_field_conjunction"]
    one_conjunction = conjunction.replace("(" + _CONJUNCTION + r"[\s._-]*)+", _CONJUNCTION + r"[\s._-]*", 1)
    no_article = conjunction.replace("(" + _ARTICLE + r"[\s._-]+)?", "")
    assert len({conjunction, one_conjunction, no_article}) == 3
    for label in runs:
        assert _card_named_keeping({"aria-label": label}, _keeping_by(one_conjunction)), label
    for label in articles:
        assert not _card_named_keeping({"aria-label": label}, _keeping_by(no_article)), label


def test_a_card_number_joined_to_an_id_document_a_contact_or_a_checkout_code_is_ordinary():
    """ID and residence forms pair two documents in one field ("Driver's
    license or ID card number", 'Passport or ID card number', 'SSN or green
    card number'), a loyalty lookup takes a phone number, an email or a card's
    number ('Phone number or loyalty card number'), a library's sign-in takes a
    barcode or a name ('Barcode or library card number', 'Last name or library
    card number'), and a checkout takes a code or a gift card's number ('Promo
    code or gift card number'). None names a payment card, so each is ordinary
    on both hosts: the conjunction pattern reads the name of an ID document, a
    contact, a checkout code or a barcode as an exempt name too
    (_OTHER_NAMES), and an exempt name may be followed by card, code or
    address, then by a number. With identifiers' numbers taken out of the
    pattern (they cover a number after any word), a host without those names
    reads each as card-named, and one whose names take a number only after
    card reads 'Driver license number or ID card number' and 'Student ID
    number or library card number' so. A host without SSN, name or barcode
    reads a label with it as card-named. The exempt pattern holds none of them,
    since passport there would take the number of Cash Passport, a prepaid card. The
    cost is that 'Cash Passport or gift card number' reads as ordinary, a label
    no real form writes. A payment word ('Passport, visa or ID card number') or
    a card ahead of the names ('Card or license or ID card number') still keeps
    the card number."""
    numbers_alone = (
        "Driver license number or ID card number", "Student ID number or library card number",
        "Driver license # or ID card number",
    )
    for label in OTHER_NAME_LISTS + list(numbers_alone):
        for attribute in CARD_FIELD_ATTRIBUTES:
            assert not _card_named({attribute: label}), (attribute, label)
    kept = ("Passport, visa or ID card number", "Card or license or ID card number", "Card or phone or loyalty card number")
    for label in kept:
        assert _card_named({"aria-label": label}), label
    limits = _load("page").limits
    exempt, conjunction = limits["card_field_exempt"], limits["card_field_conjunction"]
    assert re.search(exempt, "Cash Passport card number", re.IGNORECASE) is None
    assert _card_named({"aria-label": "Cash Passport card number"})
    assert not _card_named({"aria-label": "Cash Passport or gift card number"})

    no_identifiers = conjunction.replace("|" + _NO_CONJUNCTION_CARD_OR_CC + _IDENTIFIER_NUMBER, "")
    no_other_names = no_identifiers.replace("|" + _OTHER_NAMES, "")
    number_after_card_only = no_identifiers.replace(
        r"([\s._-]*(card|code|address))?([\s._-]*(numbers?|num|no|#))?",
        r"([\s._-]*(card|code|address)([\s._-]*(numbers?|num|no|#))?)?",
    )
    assert len({conjunction, no_identifiers, no_other_names, number_after_card_only}) == 4
    for label in OTHER_NAME_LISTS:
        assert _card_named_keeping({"aria-label": label}, _keeping_by(no_other_names)), label
    for label in numbers_alone:
        assert _card_named_keeping({"aria-label": label}, _keeping_by(number_after_card_only)), label
    others = _alternatives(_OTHER_NAMES)
    for name in ("ssn", "name", r"bar[\s._-]*code"):
        assert name in others, name
        without = conjunction.replace(_OTHER_NAMES, "(" + "|".join(one for one in others if one != name) + ")")
        assert without != conjunction
        assert any(
            _card_named_keeping({"aria-label": label}, _keeping_by(without)) for label in BARCODE_NAME_AND_SSN_LISTS
        ), name


def test_a_card_number_joined_to_an_identifier_s_number_is_ordinary_unless_the_name_holds_card_or_cc():
    """A loyalty, insurance, green card or SIM form pairs an ordinary card's
    number with an identifier's ('Account number or rewards card number',
    'Policy number or insurance card number', 'A-number or green card
    number', 'ICCID or SIM card number'). None names a payment card, so each
    is ordinary on both hosts: before a join, a name that holds neither card
    nor cc and ends in an identifier's number (number, num or no after a word,
    # or ID right after one) counts as an exempt name. A host without that
    reads each as card-named. A name that holds card or cc keeps the card
    number, so 'Card account number or gift card number' is card-named, where a
    host that lets the name hold card reads it as ordinary; 'Card number or
    gift card number' is card-named by its own card number too. 'CC no. or
    gift card number' keeps its card number by the payment word and by the
    conjunction alike, where a host that lets the name hold cc and skips the
    payment check reads it as ordinary; 'Visa number or gift card number' keeps
    it by the payment word alone. Num and no need a space or separator before
    them and a word before that, so 'Platinum', 'Casino' and a lone 'Number'
    name no identifier, where a host that takes them glued or alone reads those
    labels as ordinary. # and ID may be glued to the word, so 'ICCID' and
    'Account#' do, where a host that needs a space reads them as card-named."""
    for label in IDENTIFIER_LISTS:
        for attribute in CARD_FIELD_ATTRIBUTES:
            assert not _card_named({attribute: label}), (attribute, label)
    for label in IDENTIFIER_LOOKALIKES + IDENTIFIER_PAYMENT_LABELS:
        for attribute in CARD_FIELD_ATTRIBUTES:
            assert _card_named({attribute: label}), (attribute, label)

    def without_payment_check(label: str, conjunction: str) -> bool:
        forms = _value_forms({"aria-label": label})
        return _names_a_card(_joined([_exempted(pair, keeps=_keeping_by(conjunction)) for pair in forms]))

    conjunction = _load("page").limits["card_field_conjunction"]
    identifier = _NO_CONJUNCTION_CARD_OR_CC + _IDENTIFIER_NUMBER
    assert conjunction.count(identifier) == 1
    for label in IDENTIFIER_PAYMENT_LABELS:
        assert without_payment_check(label, conjunction) is (label != "Visa number or gift card number"), label

    no_identifiers = conjunction.replace("|" + identifier, "")
    may_hold_card = conjunction.replace(identifier, _NO_CONJUNCTION + _IDENTIFIER_NUMBER)
    may_hold_cc = conjunction.replace(identifier, r"((?!" + _CONJUNCTION + r"|card)[\s\S])*" + _IDENTIFIER_NUMBER)
    glued = conjunction.replace(identifier, identifier.replace(r"[\s._-]+(numbers?", r"[\s._-]*(numbers?"))
    alone = conjunction.replace(identifier, identifier.replace(r"[a-z0-9]([\s._-]+(numbers?", r"([\s._-]*(numbers?"))
    spaced = conjunction.replace(identifier, identifier.replace(r"[\s._-]*(#|id)", r"[\s._-]+(#|id)"))
    assert len({conjunction, no_identifiers, may_hold_card, may_hold_cc, glued, alone, spaced}) == 7
    for label in IDENTIFIER_LISTS:
        assert _card_named_keeping({"aria-label": label}, _keeping_by(no_identifiers)), label
    for label in ("Card account number or gift card number", "Bank card account number or gift card number"):
        assert not _card_named_keeping({"aria-label": label}, _keeping_by(may_hold_card)), label
    assert not without_payment_check("CC no. or gift card number", may_hold_cc)
    for label in ("Platinum or gift card number", "Casino or gift card number"):
        assert not _card_named_keeping({"aria-label": label}, _keeping_by(glued)), label
    assert not _card_named_keeping({"aria-label": "Number or gift card number"}, _keeping_by(alone))
    for label in ("ICCID or SIM card number", "Account# or rewards card number"):
        assert _card_named_keeping({"aria-label": label}, _keeping_by(spaced)), label


def test_a_checkout_s_payment_section_leaves_a_gift_card_s_field_ordinary():
    """Payment, bank and charge also name a checkout's section, so they aren't
    payment words, and a gift card's field in that section is ordinary, as it
    was before the payment check: payment[gift_card_number] or
    paymentGiftCardNumber with the placeholder 'Gift card number'. A host that
    keeps payment among the payment words reads each as card-named."""
    payment = _load("page").limits["card_field_payment"]
    for word in ("payment", "bank", "charge", "PAYMENT", "Bank", "Payment card"):
        assert re.search(payment, word, re.IGNORECASE) is None, word
    _attribute_example(
        "a gift card's field in a checkout's payment section",
        False, {"name": "payment[gift_card_number]", "placeholder": "Gift card number"},
    )
    _attribute_example(
        "a gift card's id in a checkout's payment section",
        False, {"id": "paymentGiftCardNumber", "placeholder": "Gift card number"},
    )
    with_payment = payment.replace("(credit|", "(credit|payment|")
    assert with_payment != payment
    for value in PAYMENT_SECTION_GIFT_CARDS:
        forms = _value_forms({"name": value, "placeholder": "Gift card number"})
        assert any(re.search(with_payment, form, re.IGNORECASE) for pair in forms for form in pair), value
        assert _names_a_card(_joined(forms)), value


def _alternatives(group: str) -> list[str]:
    """The top-level alternatives of a parenthesized pattern group."""
    found, depth, part = [], 0, ""
    for char in group[1:-1]:
        depth += {"(": 1, ")": -1}.get(char, 0)
        if char == "|" and depth == 0:
            found, part = [*found, part], ""
        else:
            part += char
    return [*found, part]


def test_an_identity_card_s_number_is_ordinary_and_no_qualifier_takes_a_payment_card_s_name():
    """An identification, green, permanent resident, residence, Social
    Security, Aadhaar, citizen or voter card's number is an identity card's, so
    it is ordinary on both hosts, and each qualifier of the exempt pattern has
    an ordinary example that needs it. No qualifier occurs in a payment card's name in
    either form, checked for every card-named value in page.json: Green Dot's
    card stays card-named, with Dot between green and card, and a label naming
    Amex's Green Card keeps its card number by amex. A travel form's
    'Passport, visa or ID card number' stays card-named by visa."""
    qualifiers = _alternatives(_EXEMPT_QUALIFIERS)
    assert "green" in qualifiers and "residen(t|ce|cy)" in qualifiers and "voter" in qualifiers
    assert "identification" in qualifiers
    for qualifier in qualifiers:
        without = CARD_FIELD_EXEMPT.replace(
            _EXEMPT_QUALIFIERS, "(" + "|".join(one for one in qualifiers if one != qualifier) + ")",
        )
        needs_it = [
            value for value in ORDINARY_CARD_NUMBERS
            if _names_a_card(_NameForms(*(
                re.sub(without, " ", form, flags=re.IGNORECASE) for form in _value_forms({"name": value})[0]
            )))
        ]
        assert needs_it, qualifier
    exempt = _load("page").limits["card_field_exempt"]
    for value in ("Green Dot card number", "greenDotCardNumber"):
        assert _card_named({"placeholder": value}) and re.search(exempt, value, re.IGNORECASE) is None, value
    for value in ("Amex Green Card number", "Passport, visa or ID card number"):
        assert re.search(exempt, value, re.IGNORECASE) and _holds_payment_word(_value_forms({"name": value})[0])
        assert _card_named({"placeholder": value}), value


def test_a_conjunction_keeps_the_card_number_unless_each_name_it_joins_ends_in_an_exempt_name():
    """A list that names a payment card first and an exempt card's number
    last takes the person's payment card, whatever names stand between:
    'Card or gift certificate or gift card number'. So the conjunction pattern
    reads the names the text's joins separate, and drops the match only when
    each of them ends in an exempt name; a name holds no conjunction. A host
    whose first name may hold one reads 'Card or gift' as that name and
    'certificate or' as the next, and the field as ordinary. Words may come
    before the exempt name in any name of the list, so 'Passport, residence
    permit or ID card number' and 'Gift card, mobile phone or loyalty card
    number' stay ordinary: a host that takes words only before the first
    name's exempt name reads them as card-named, and one that takes none reads
    'Enter a gift or loyalty card number' and 'Gift card, mobile phone or
    loyalty card number' so."""
    for label in CARD_FIRST_LISTS + ["Card or a gift or loyalty card number"]:
        for attribute in CARD_FIELD_ATTRIBUTES:
            assert _card_named({attribute: label}), (attribute, label)
            assert not any(_holds_payment_word(pair) for pair in _value_forms({attribute: label})), (attribute, label)
    worded = WORDED_NAME_LISTS + ["Enter a gift or loyalty card number", "Student ID or library card number"]
    for label in worded:
        for attribute in CARD_FIELD_ATTRIBUTES:
            assert not _card_named({attribute: label}), (attribute, label)

    conjunction = _load("page").limits["card_field_conjunction"]
    one_way = "(" + _NAME_ENDING_IN_AN_EXEMPT_NAME + ")+"
    holding_a_conjunction = conjunction.replace("(?=(" + _NO_CONJUNCTION, r"(?=([\s\S]*", 1)
    names_alone = _NAME_ENDING_IN_AN_EXEMPT_NAME.replace("(?=(" + _NO_CONJUNCTION, "(?=(", 1)
    words_first_only = conjunction.replace(one_way, _NAME_ENDING_IN_AN_EXEMPT_NAME + "(" + names_alone + ")*")
    no_words = conjunction.replace(one_way, "(" + names_alone + ")+")
    assert one_way in conjunction
    assert len({conjunction, holding_a_conjunction, words_first_only, no_words}) == 4
    for label in ("Card or gift certificate or gift card number", "Card or a gift or loyalty card number"):
        assert not _card_named_keeping({"aria-label": label}, _keeping_by(holding_a_conjunction)), label
    for label in WORDED_NAME_LISTS[:3]:
        assert _card_named_keeping({"aria-label": label}, _keeping_by(words_first_only)), label
    for label in ("Enter a gift or loyalty card number", "Gift card, mobile phone or loyalty card number"):
        assert _card_named_keeping({"aria-label": label}, _keeping_by(no_words)), label


#: Texts before an exempt match that a host reads at once, each with whether
#: it keeps the match: a long list of names, a long run of separators, and a
#: long run of joins with no join at its end.
LONG_BEFORES = [
    ("national ID or " * 18 + "x or ", True), ("e-gift or " * 2000 + "x or ", True), ("-" * 5000 + "or", True),
    ("account number or " * 1500 + "x or ", True), (", " * 4000 + "x", False), ("or " * 4000 + "x", False),
    ("and/or " * 2000 + "x", False), (" ,- " * 3000 + "x", False),
]


def test_the_conjunction_pattern_answers_a_long_list_or_a_long_run_of_joins_at_once():
    """The page writes its own labels, so the conjunction pattern reads each
    text before an exempt match in time in step with its length, on both
    hosts. A name never starts with a space or separator, and a lookahead
    never gives back what it matched, so each name is read one way: with a
    name that may start with a space, 'national ID or ' 18 times and then 'x
    or ' took 7 s in Python's re, four times as long per two more names. The
    join at the end starts at the text's start or right after a character no
    join holds, so a run of joins is read from one start: where any character
    may come before it, ', ' 4000 times and then x took 4 s in Python's re and
    0.2 s in node, four times as long at twice the length."""
    for text, kept in LONG_BEFORES:
        started = time.perf_counter()
        assert _kept_by_a_conjunction(text) is kept, text[:20]
        assert time.perf_counter() - started < 1, text[:20]
    conjunction = _load("page").limits["card_field_conjunction"]
    from_any_start = conjunction.replace(r"([\s\S]*" + _OUTSIDE_A_JOIN + r")?[\s._-]*", r"[\s\S]*")
    assert from_any_start != conjunction
    for before in ("Card or ", "Gift or ", "Card, or a ", "x", ", " * 20 + "x", "Card or" + " " * 20):
        assert _kept_by_a_conjunction(before) is (re.search(from_any_start, before, re.IGNORECASE) is not None), before


def test_a_pan_card_s_number_is_an_identity_card_s_and_pan_counts_only_at_a_word_s_start():
    """India's PAN card is a tax identity card, and a KYC form asks for its
    number ('PAN card number', panCardNumber, 'PAN or Aadhaar card number'), so
    it is ordinary on both hosts. Pan counts only with no letter or digit right
    before it, so 'Japan card number' and 'Span card number' keep their card
    number, where a host whose pan needs no word start reads them as ordinary.
    A PAN is also a payment card's number, but the card pattern names no PAN
    alone, so 'Card number (PAN)' is card-named by its card number, and 'PAN or
    gift card number' reads as ordinary, as 'PAN' does."""
    for value in PAN_CARD_NUMBERS + ["PAN or Aadhaar card number"]:
        for attribute in CARD_FIELD_ATTRIBUTES:
            assert not _card_named({attribute: value}), (attribute, value)
    for value in PAN_LOOKALIKES:
        for attribute in CARD_FIELD_ATTRIBUTES:
            assert _card_named({attribute: value}), (attribute, value)
    exempt = _load("page").limits["card_field_exempt"]
    unbounded = exempt.replace("|(?<![a-z0-9])pan|", "|pan|")
    assert unbounded != exempt
    for value in ("Japan card number", "Span card number"):
        assert not _card_named_keeping({"aria-label": value}, _kept_by_a_conjunction, exempt=unbounded), value
    assert not _card_named({"aria-label": "PAN or gift card number"}) and not _card_named({"aria-label": "PAN"})


def test_a_hash_after_card_names_a_card_s_number():
    """A form writes a card's number as 'Card #' ('Credit card #', 'CC #'), so
    # is a number word in the card pattern after card and cc, as in an exempt
    name, where a host without it types into 'Card #'. The exempt pattern takes
    # too, so 'Gift card #' is ordinary, where a host whose exempt pattern
    lacks it reads it as card-named, and 'Card / gift card #' keeps its card
    number by the conjunction."""
    for value in HASH_CARD_NUMBERS + ["Card / gift card #"]:
        for attribute in CARD_FIELD_ATTRIBUTES:
            assert _card_named({attribute: value}), (attribute, value)
    for value in HASH_ORDINARY_CARD_NUMBERS:
        for attribute in CARD_FIELD_ATTRIBUTES:
            assert not _card_named({attribute: value}), (attribute, value)
    limits = _load("page").limits
    name, exempt = limits["card_field_name"], limits["card_field_exempt"]
    assert name.count("|#)") == 2 and exempt.count("|#)") == 1
    for value in HASH_CARD_NUMBERS:
        assert re.search(name.replace("|#)", ")"), value, re.IGNORECASE) is None, value
    for value in HASH_ORDINARY_CARD_NUMBERS:
        assert _card_named_keeping({"placeholder": value}, _kept_by_a_conjunction, exempt=exempt.replace("|#)", ")"))


def test_a_card_named_field_on_the_page_is_a_secret_field():
    """A field the card rule names is secret on the page itself, on both
    hosts, as a cc- autocomplete field is: its value never reaches a read,
    and a type, a select or an editing key into it answers 403
    password_field. The checkout example's Card field is <input
    name=card_number aria-label=Card>, named for a card by its name alone,
    and the tool refuses a type into it before a host sees it."""
    fixture = _load("page")
    assert "A type with focused true never types into a card-named field (403 password_field)." not in fixture.description
    assert (
        "A card-named field is a secret field: it carries value_hidden and never its value, and a type into it, by "
        "element or with focused true, a select of it or a press of a key that edits text in it answers 403 "
        "password_field."
    ) in fixture.description
    checkout = fixture.examples["a checkout page with a card field named card_number"]
    assert (checkout.where, checkout.kind, checkout.status) == ("GET /sessions/{id}/page", "response", 200)
    assert _card_named({"name": "card_number", "aria-label": "Card"})
    assert not _card_named({"aria-label": "Card"}) and not _card_named({"aria-label": "Name on card"})
    raw = {el["name"]: el for el in checkout.body["elements"]}
    assert raw["Card"] == {"id": 3, "role": "textbox", "name": "Card", "value_hidden": True}
    page = Page.model_validate(checkout.body)
    card, holder = (next(el for el in page.elements if el.name == name) for name in ("Card", "Name on card"))
    assert card.secret and card.value is None
    assert not holder.secret and holder.value
    shown = format_page(page)
    assert '[3] textbox "Card" (secret field)' in shown
    assert f'[{holder.id}] textbox "Name on card" value={json.dumps(holder.value)}' in shown
    assert _load("error").examples["password_field"].status == 403

    session = SimpleNamespace(_browser_config=BrowserConfig(base_url=ROUTER, profile=PROFILE))
    state = browser_tool._BrowserState(client=_client(_Recorder()), session_id=PROFILE)
    body = browser_tool._step_body({"step": {"action": "type", "element": card.id, "text": "4242"}})
    outcome = asyncio.run(browser_tool._gate(session, state=state, body=body, page=page, risky=False))
    assert isinstance(outcome, browser_tool.ToolOutcome) and outcome.reason == "browser_secret_field"


def test_the_page_fixture_lets_a_host_name_the_focused_node():
    """A page may carry focus, an opaque token for the node that has the
    focus, followed down through the frames the host can read. Inside a frame
    it can't read (a payment frame, another site's frame inside another
    site's), the token names that frame's element, so a focus that moves into
    such a frame, or from one to another, changes the token where an omitted
    focus would have read the same. The desktop app's browser sends it, so the
    tool can tell that the focus moved while a card waited; the web's
    browser, which reads the focus again when a step runs, leaves it out."""
    fixture = _load("page")
    assert "through frames. Equal tokens mean the same node, and a host may omit" not in fixture.description
    assert (
        "A page may carry focus: an opaque token naming the focused node, the deepest one through frames that the "
        "host can read; when the focus is inside a frame the host can't read, the token names that frame's element. "
        "Equal tokens mean the same node. A host may omit focus, and omits it when nothing is focused."
    ) in fixture.description
    framed = fixture.examples["a page with a frame"].body
    assert isinstance(framed["focus"], str) and framed["focus"]
    unchecked = fixture.examples["a page with no check"].body
    assert "focus" not in unchecked
    assert Page.model_validate(framed).focus == framed["focus"]
    assert Page.model_validate(unchecked).focus is None


def test_the_page_fixture_gives_the_element_order_both_hosts_use():
    """Past the cap both hosts list what the person sees first. One lists an
    open dialog's elements ahead of everything, even when all fit, and the
    other keeps page order when all fit, so the fixture allows both."""
    description = _load("page").description
    assert "It keeps page order when every shown element fits" not in description
    assert "Past the cap it lists the on-screen elements ahead of the off-screen ones, each group in page order" in description
    assert "when every shown element fits, a host may keep page order or use the same order" in description
    assert "A host may list an open dialog's elements (in_dialog) ahead of both groups." in description


def test_a_page_from_a_host_that_predates_frames_says_nothing_about_frames():
    """A host that sends no ``frames`` hasn't said the page has none."""
    body = {key: value for key, value in _load("page").examples["a page with no check"].body.items() if key != "frames"}
    page = Page.model_validate(body)
    assert page.frames is None
    assert "Frame " not in format_page(page)


def test_the_action_fixture_says_whether_the_step_changed_anything():
    actions = _load("action").examples
    assert ActionResult.model_validate(actions["a step that changed nothing"].body).changed is False
    assert ActionResult.model_validate(actions["the answer with the page after"].body).changed is True
    # An older host sends no `changed`: the tool can't tell.
    assert ActionResult.model_validate({"ok": True, "url": "https://shop.example.test/cart"}).changed is None


def test_the_action_fixture_says_a_step_that_changed_the_page_in_place_settles_too():
    """The settle both hosts run: no settle ends for quiet, or for a step
    that started nothing, inside the first settle_quiet_ms (the request
    window, for a debounced search); a step that started nothing answers at
    its end; one that started a request or a navigation, or finds the page
    aria-busy, waits for quiet up to settle_max_ms; one that only changed the
    page in place waits for quiet a moment at most. A navigation's cap is
    each host's own: the router's browser lets it run past settle_max_ms
    while the new page loads, and the desktop app's waits for a navigate
    step's load before the settle and caps every settle at settle_max_ms,
    or sooner inside the step's own time budget, so a navigate whose load
    took its full 30 s settles for about 2.5 s at most."""
    description = _load("action").description
    assert "answers at once" not in description
    assert "a host may first wait up to settle_quiet_ms" not in description
    for sentence in (
        "Neither host ends a settle for quiet, or for a step that started nothing, before settle_quiet_ms from its "
        "start, so a request the step starts by then (a search box's debounced suggestions) is waited for.",
        "A step that by then has changed nothing, started no main-frame request or navigation and left nothing "
        "shown marked aria-busy answers there.",
        "One that started a main-frame request or a navigation, or finds the page aria-busy, waits until those "
        "requests are done, nothing shown is marked aria-busy and the page has shown no change for "
        "settle_quiet_ms, at most settle_max_ms from the settle's start.",
        "One that only changed the page in place waits for that quiet at most settle_quiet_ms past the host's "
        "first look at the page after the step, so a page that changes on its own (a ticker) doesn't hold every "
        "step to the cap.",
        "A navigation's cap is each host's own.",
        "The router's browser waits up to 5 s each for a navigation to commit and for its new page's "
        "DOMContentLoaded (a navigate step's own load has waited up to 30 s for it), then for the new page's "
        "network to go idle too, and stops at the later of settle_max_ms from the settle's start and 1.5 s after "
        "that DOMContentLoaded wait, never past 31.5 s from the step's start.",
        "The desktop app's browser waits for a navigate, back, forward or reload step's page to finish loading, "
        "up to 30 s, before the settle, and stops every settle, a navigation's too, at settle_max_ms from its start "
        "or 0.5 s before the step's 33 s budget ends, whichever comes first.",
    ):
        assert sentence in description, sentence
    assert "a navigation's too, at settle_max_ms from its start." not in description


def test_the_action_fixture_says_how_a_focused_type_reaches_a_field_in_a_frame():
    """A frame's fields carry no number, so the one type step without an
    element types into the field that has the focus, wherever it is. Each
    host types it only into a text entry that isn't secret and refuses
    anything else with its own code, and into_frame, which is about where a
    press of the mouse lands, never stops it."""
    description = _load("action").description
    for sentence in (
        "A type step with focused true, its text and submit, and no element or point, types into the field that "
        "has the focus, inside a frame too.",
        "The host types it only when that field is a text entry that isn't secret (a text, search, email, url, tel "
        "or number input, a textarea, or an editor); otherwise it types nothing and refuses it: 403 password_field "
        "for a secret, card or password field, 409 not_a_text_field for anything else, and 409 focus_unreadable "
        "when it can't read what has the focus.",
        "The desktop app's browser, which holds no steps itself, refuses (409) a mouse press, or a drag's release, "
        "that lands in a frame when into_frame is false or missing; into_frame never stops a focused type, which "
        "has no point.",
    ):
        assert sentence in description, sentence
    body = _load("action").examples["type into the focused field"].body
    assert body == {"action": "type", "focused": True, "text": "ada@example.test", "submit": False, "include_page": True}
    errors = _load("error").examples
    assert [errors[code].status for code in ("password_field", "not_a_text_field", "focus_unreadable")] == [403, 409, 409]
    held = {name: e.body["held"] for name, e in _load("commit_steps").examples.items() if e.body["step"].get("focused")}
    assert held == {"type into the focused field": None, "type into the focused field and submit": "commit"}


def test_a_focused_type_whose_focus_leaves_the_field_mid_type_answers_stale_element():
    """Both hosts read the focus again before each key and before submit's
    Enter, and stop when it has left the field the type began in: 409
    stale_element, or 403 password_field when it is now on a secret or
    card-named field."""
    sentence = (
        "Before each key, and before the Enter submit adds, the host reads the focus again: when it has left that "
        "field the host stops typing, keeps what it typed, and answers 409 stale_element, or 403 password_field when "
        "the focus is now on a secret, card or password field."
    )
    assert sentence in _load("action").description
    errors = _load("error")
    assert errors.examples["stale_element"].status == 409
    assert (
        "One whose focus leaves that field before the text and its Enter are all typed stops there and answers "
        "stale_element, or password_field when the focus moved onto a secret, card or password field."
    ) in errors.description


def test_the_page_fixture_says_a_frame_left_unread_takes_none_of_frames_read():
    """Both hosts count the frames they read after their refusals, so a
    sign-in or covered frame on-screen first never costs a plain one its read."""
    description = _load("page").description
    assert "or comes after the first frames_read." not in description
    assert (
        "A host reads at most frames_read frames, in the order frames lists them, and a frame it leaves unread for "
        "any of these reasons takes none of them."
    ) in description
    assert "and a host leaves unread a frame it finds the person can't see." in description


def test_the_page_fixture_lets_a_host_list_the_frames_inside_a_frame_it_read():
    """A host that reads a same-origin frame may list the frames inside it
    after the page's own, unread, so a step by position knows a frame lies
    there."""
    description = _load("page").description
    assert "frames lists the page's rendered child frames, on-screen first:" not in description
    assert (
        "frames lists the page's rendered child frames, on-screen first, and a host may list after them, unread, the "
        "frames inside a frame it read."
    ) in description
    assert (
        "Each frame has url (origin and path for http and https, about:blank or about:srcdoc, otherwise the scheme "
        "alone), read, box"
    ) in description


#: page.json's definition of a frame's at, which both hosts compute: the
#: scrolls of the frame's containing-block chain, so a frame fixed to the view
#: keeps its at whatever scrolls, and only a frame a sticky element holds
#: stuck to the view moves it by the scroll.
FRAME_AT = (
    "A host sends at with every frame whose box isn't null: the top-left {x, y} of the frame element's content box "
    "(the box that box gives in screenshot pixels) in its document's content coordinates, in CSS pixels rounded "
    "to whole ones. That is the "
    "box's top-left in the viewport plus the scroll offset of each element in the frame element's containing-block "
    "chain, plus its document's scroll unless that chain ends at an element fixed to the view. The chain holds the "
    "elements whose scroll moves it: an element in flow (static, relative or sticky) moves with its parent's "
    "scroll, and an absolute or fixed one with its containing block's as CSS defines it (an absolute one's nearest "
    "positioned ancestor, a fixed one's nearest ancestor with a transform, a filter or the like, else none), and so "
    "on up from each. So no scroll, its document's or an inner column's, changes the at of a frame that scrolls with "
    "the content, or of one fixed to the view. A frame that a sticky element holds stuck to the view keeps its box "
    "as its document scrolls, so its at moves by the scroll."
)


def test_the_page_fixture_places_each_frame_on_its_page_whatever_scrolls():
    """A frame's box is where it is on the screenshot, so the page's scroll
    and an inner column's both move it. at is where the frame element sits in
    its document, so neither does, and it tells a frame that moved on the
    page from one that only scrolled. Only the scrolls that move the frame
    count, those of its containing-block chain, so a frame fixed to the view
    keeps its at too, inside a scrolling column as well, and a frame stuck to
    the view by a sticky element moves it by the scroll. A host sends it with
    every frame it places, in whole CSS pixels, and a frame with no box
    carries none."""
    fixture = _load("page")
    assert "plus the page's scroll plus the scroll offset of every scrolling element it sits in" not in fixture.description
    assert FRAME_AT in fixture.description
    placed = 0
    for name, example in fixture.examples.items():
        for frame in example.body.get("frames", []) if example.kind == "response" else []:
            if frame["box"] is None:
                assert "at" not in frame, (name, frame["url"])
                continue
            placed += 1
            assert set(frame["at"]) == {"x", "y"}, (name, frame["url"])
            assert all(type(frame["at"][axis]) is int for axis in ("x", "y")), (name, frame["url"])
    assert placed == 3
    framed = fixture.examples["a page with a frame"].body
    assert [frame.get("at") for frame in framed["frames"]] == [{"x": 112, "y": 1500}, {"x": 112, "y": 1720}, None]
    assert [frame.model_dump().get("at") for frame in Page.model_validate(framed).frames] == [
        {"x": 112, "y": 1500}, {"x": 112, "y": 1720}, None,
    ]


def test_the_page_model_says_what_the_page_fixture_says_of_its_frames():
    """Page.frames says what page.json says: the page's rendered child frames,
    on-screen first, and after them, unread, the frames inside a frame the
    host read."""
    import inspect

    source = " ".join(line.strip().removeprefix("#:").strip() for line in inspect.getsource(Page).splitlines())
    assert "The page's child frames, on-screen first. Their" not in source
    assert (
        "The page's rendered child frames, on-screen first, and after them a host may list, unread, the frames "
        "inside a frame it read."
    ) in source


def test_the_frame_model_says_which_scrolls_page_json_adds_to_a_frame_s_at():
    """Frame.at says what page.json's FRAME_AT says: the scrolls of the
    frame's containing-block chain, and its document's scroll unless that
    chain ends fixed to the view, so a frame fixed to the view keeps its at
    and a frame stuck to the view by a sticky element moves it."""
    import inspect

    source = " ".join(line.strip().removeprefix("#:").strip() for line in inspect.getsource(Frame).splitlines())
    assert "plus the page's scroll plus the scroll offset of every scrolling element it sits in" not in source
    assert (
        "plus the scroll offset of each element in its containing-block chain, plus its document's scroll unless "
        "that chain ends fixed to the view."
    ) in source
    assert "A frame fixed to the view keeps its at; one a sticky element holds stuck to the view moves it by the scroll." in source


def test_the_control_fixture_tells_done_by_the_person_from_any_other_agent_record():
    controls = _load("control").examples
    fresh = SessionInfo.model_validate(controls["a session the agent holds"].body).control
    handed = Control.model_validate(controls["handed to the person"].body)
    done = SessionInfo.model_validate(controls["done by the person"].body).control
    assert (handed.holder, handed.by) == ("human", "agent")
    assert handed.watching is False
    # A host that doesn't say it watches isn't taken to watch.
    unsaid = {k: v for k, v in controls["handed to the person"].body.items() if k != "watching"}
    assert Control.model_validate(unsaid).watching is False
    assert Control.model_validate(controls["handed to the person, watching"].body).watching is True
    assert (done.holder, done.by) == ("agent", "person") and done.seq > handed.seq
    assert (fresh.holder, fresh.by) == ("agent", "agent") and fresh.seq < handed.seq
    assert SessionInfo.model_validate(controls["a session before a page loads"].body).url is None


def test_a_project_keeps_at_most_the_pinned_number_of_saved_tabs():
    saved = _load("saved_tabs")
    assert saved.limits == {"saved_tabs_per_project": 100}
    assert "100" in _load("error").examples["too_many_saved_tabs"].body["message"]


def test_a_held_step_reads_what_the_browser_says_it_holds():
    body = _load("error").examples["approval_required"].body
    assert set(body["held"]) == set(HeldStep.model_fields)
    held = ErrorBody.model_validate(body).held
    assert held == HeldStep(kind="commit", action="click", names=["Pay $42.00"], url="https://shop.example.test/cart")


def _held(example: ContractExample) -> str | None:
    body = example.body
    element = Element.model_validate({"id": 7, **body["element"]}) if body["element"] is not None else None
    return browser_tool._classify(body["step"], url=body["url"], requested=body.get("requested"), element=element)


@pytest.mark.parametrize("name", sorted(_load("commit_steps").examples))
def test_the_commit_list_holds_what_the_fixture_says(name):
    example = _load("commit_steps").examples[name]
    assert example.kind == "classification"
    assert _held(example) == example.body["held"], name


def test_the_commit_list_covers_every_word_and_both_holds():
    held = [e.body["held"] for e in _load("commit_steps").examples.values()]
    assert {"commit", "position", None} <= set(held)
    names = " ".join(
        (e.body["element"] or {}).get("name", "") for e in _load("commit_steps").examples.values() if e.body["held"] == "commit"
    ).lower()
    for word in browser_tool.COMMIT_WORDS:
        assert word in names, word


# ── requests ───────────────────────────────────────────────────────────────


class _Recorder:
    """A host that records each request and answers with any fixture answer
    the client can read, so every client call completes. ``page`` names the
    page example it reads back."""

    def __init__(self, page: str = "a page with no check"):
        self.sent: list[tuple[str, str, dict | None]] = []
        self.page = page

    def handler(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else None
        path = request.url.path.removeprefix("/browser")
        self.sent.append((request.method, path, body))
        answers = {
            ("POST", f"/sessions/{PROFILE}/control"): _load("control").examples["handed to the person"].body,
            ("GET", f"/sessions/{PROFILE}"): _load("control").examples["a session the agent holds"].body,
            ("POST", f"/sessions/{PROFILE}/tabs"): _load("tabs").examples["the opened tab"].body,
            ("GET", f"/sessions/{PROFILE}/tabs"): _load("tabs").examples["the open tabs"].body,
            ("POST", f"/sessions/{PROFILE}/tabs/t2/activate"): _load("tabs").examples["the activated tab"].body,
            ("POST", f"/sessions/{PROFILE}/sites"): _load("sites").examples["the noted site"].body,
            ("GET", f"/sessions/{PROFILE}/sites"): _load("sites").examples["the sites on record"].body,
            ("POST", "/saved-tabs"): _load("saved_tabs").examples["the saved tab"].body,
            ("GET", "/saved-tabs"): _load("saved_tabs").examples["the project's saved tabs"].body,
            ("GET", f"/sessions/{PROFILE}/page"): _load("page").examples[self.page].body,
            ("POST", f"/sessions/{PROFILE}/action"): _load("action").examples["the answer with the page after"].body,
        }
        answer = answers.get((request.method, path), {"ok": True})
        return httpx.Response(200, json=answer, headers={"x-image-width": "1024", "x-image-height": "640"})


async def _k():
    return "k"


def _client(recorder: _Recorder) -> BrowserClient:
    return BrowserClient(
        base_url=ROUTER, credential=_k, trust_env=True, first_call_timeout_s=45.0,
        http=httpx.AsyncClient(transport=httpx.MockTransport(recorder.handler)),
    )


async def _every_client_call(client: BrowserClient) -> None:
    await client.open_session(session_id=PROFILE, url="https://shop.example.test/")
    await client.page(session_id=PROFILE)
    await client.act(session_id=PROFILE, step={"action": "click", "element": 7, "include_page": True})
    await client.screenshot(session_id=PROFILE)
    await client.sites(session_id=PROFILE)
    await client.note_site(session_id=PROFILE, site="github.com", account_hint="ada")
    await client.control(session_id=PROFILE)
    await client.hand_to_person(session_id=PROFILE, reason="Please sign in to Example Shop")
    await client.tabs(session_id=PROFILE)
    await client.new_tab(session_id=PROFILE, url="https://docs.example.test/")
    await client.activate_tab(session_id=PROFILE, tab_id="t2")
    await client.close_tab(session_id=PROFILE, tab_id="t2")
    await client.saved_tabs(project=PROJECT)
    await client.save_tab(project=PROJECT, name="Q3 pipeline report", url="https://crm.example.test/reports/q3", notes="start here")
    await client.delete_saved_tab(project=PROJECT, name="Q3 pipeline report")
    await client.close_session(session_id=PROFILE)


#: The client call (or, for steps, the tool's step body) that sends each
#: request example. Every request example the hosts accept is here.
REQUESTS = {
    ("sessions", "open a session"): lambda c: c.open_session(session_id=PROFILE, url="https://shop.example.test/"),
    ("control", "hand to the person"): lambda c: c.hand_to_person(session_id=PROFILE, reason="Please sign in to Example Shop"),
    ("control", "hand to the person, waiting for a box they check"): lambda c: c.hand_to_person(
        session_id=PROFILE, reason="Check the unaccompanied-minor policy box", wait_for=WaitFor(element=12, state="checked"),
    ),
    ("control", "hand to the person, waiting for the page to change"): lambda c: c.hand_to_person(
        session_id=PROFILE, reason="Please sign in to Example Shop", wait_for=WaitFor(page="changes"),
    ),
    ("tabs", "open a tab"): lambda c: c.new_tab(session_id=PROFILE, url="https://docs.example.test/"),
    ("sites", "note a site"): lambda c: c.note_site(session_id=PROFILE, site="github.com", account_hint="ada"),
    ("saved_tabs", "save a tab"): lambda c: c.save_tab(
        project=PROJECT, name="Q3 pipeline report", url="https://crm.example.test/reports/q3", notes="start here",
    ),
}
#: The step the model writes for each action request example.
STEPS = {
    "click an element": {"action": "click", "element": 7},
    "type and submit": {"action": "type", "element": 1, "text": "MacBook Air", "submit": True},
    "click at a point on the screenshot": {"action": "click", "x": 412, "y": 230},
    "press Enter in an element": {"action": "press", "key": "Enter", "element": 2},
    "drag from an element to a point": {"action": "drag", "from": {"element": 5}, "to": {"x": 600, "y": 300}},
    "type into the focused field": {"action": "type", "focused": True, "text": "ada@example.test"},
}
#: The step the model writes for each action request example the tool's gate
#: sends to the desktop app's browser once the person approves its card, on
#: the page with a frame.
APPROVED_IN_APP_STEPS = {
    "click at a point the person approved as one step that may land in a frame": {"action": "click", "x": 200, "y": 350},
}


async def _approved_in_app(step: dict, monkeypatch: pytest.MonkeyPatch) -> dict:
    """The body the gate sends to the desktop app's browser when the person
    approves the step's card, on a page that holds still while the card
    waits: the gate reads it again after a card for a point in a frame."""
    async def approve(*args, **kwargs):
        return None

    monkeypatch.setattr(browser_tool, "ask_approval_card", approve)
    session = SimpleNamespace(_browser_config=BrowserConfig(base_url="http://127.0.0.1:51234", profile=PROFILE, token="k"))
    state = browser_tool._BrowserState(client=_client(_Recorder(page="a page with a frame")), session_id=PROFILE)
    page = Page.model_validate(_load("page").examples["a page with a frame"].body)
    return await browser_tool._gate(session, state=state, body=browser_tool._step_body({"step": step}), page=page, risky=False)


@pytest.mark.parametrize(("reason", "sent"), [
    ("a" * 299 + "\U0001F600", "a" * 299),
    ("\U0001F600" * 150, "\U0001F600" * 150),
    ("a" * 300 + "b", "a" * 300),
])
async def test_the_hand_over_reason_fits_the_hosts_utf16_limit(reason, sent):
    """The hosts count the reason in UTF-16 units and answer a longer one
    400, so an emoji counts 2 and the cut never splits one."""
    recorder = _Recorder()
    await _client(recorder).hand_to_person(session_id=PROFILE, reason=reason)
    body = recorder.sent[0][2]
    assert body["reason"] == sent
    assert len(body["reason"].encode("utf-16-le")) // 2 <= client_module.REASON_UNITS


def test_every_accepted_request_is_one_the_client_sends(monkeypatch):
    accepted = {(f, n) for f, n, e in _examples("request") if not e.refused}
    assert accepted == set(REQUESTS) | {("action", n) for n in [*STEPS, *APPROVED_IN_APP_STEPS]}
    for (file, name), call in REQUESTS.items():
        recorder = _Recorder()
        asyncio.run(call(_client(recorder)))
        assert recorder.sent[-1][2] == _load(file).examples[name].body, name
    for name, step in STEPS.items():
        assert browser_tool._step_body({"step": step}) == _load("action").examples[name].body, name
    for name, step in APPROVED_IN_APP_STEPS.items():
        assert asyncio.run(_approved_in_app(step, monkeypatch)) == _load("action").examples[name].body, name


#: The focused types and presses both hosts answer 400 bad_request, each
#: for one field the accepted focused type lacks or has otherwise.
REFUSED_STEPS = {
    "a focused type that names an element": {"element": 1},
    "a focused type at a point": {"x": 200, "y": 350, "coords": "screenshot"},
    "a focused type that clears the field first": {"clear": True},
    "a type with focused false": {"focused": False, "element": 1},
}


def test_the_action_fixture_names_the_focused_steps_both_hosts_refuse_and_the_tool_never_sends():
    """A focused type types at the caret of whatever has the focus, so it
    names no element or point and clears nothing; focused is true or absent,
    and only a type step carries it. Both hosts answer each of these 400
    bad_request, and the tool refuses each step before it sends it."""
    actions = _load("action")
    refused = {name: e for name, e in actions.examples.items() if e.refused}
    assert set(refused) == {*REFUSED_STEPS, "a press with focused"}
    accepted = actions.examples["type into the focused field"].body
    for name, change in REFUSED_STEPS.items():
        assert refused[name].where == "POST /sessions/{id}/action" and refused[name].kind == "request"
        assert refused[name].body == {**accepted, **change}, name
    press = {"action": "press", "focused": True, "key": "a", "include_page": True}
    assert refused["a press with focused"].body == press
    assert (
        "A type with focused true that also names an element or a point, or clear other than false, a type with "
        "focused other than true, and focused on any other action are refused with 400 bad_request."
    ) in actions.description
    for example in refused.values():
        step = {k: v for k, v in example.body.items() if k not in ("include_page", "coords")}
        with pytest.raises(ValueError):
            browser_tool._step_body({"step": step})


def test_every_wait_for_the_hosts_take_reads_as_the_model_the_client_sends():
    waits = [e.body["wait_for"] for _, _, e in _examples("request") if "wait_for" in e.body]
    assert len(waits) == 2
    for body in waits:
        assert WaitFor.model_validate(body, strict=True).model_dump(exclude_none=True) == body


@pytest.mark.parametrize("body", [
    {"element": 12}, {"state": "checked"}, {"element": 12, "state": "checked", "page": "changes"}, {"page": "loads"},
    {"element": 12, "state": "checked", "value": "x"}, {},
])
def test_a_wait_for_is_one_element_and_its_state_or_the_page_and_never_a_value(body):
    with pytest.raises(ValidationError):
        WaitFor.model_validate(body)


def test_the_client_never_sends_a_refused_request():
    refused = [e.body for _, _, e in _examples("request") if e.refused]
    assert [e.body for e in _load("control").examples.values() if e.refused] == [{"holder": "agent", "reason": ""}]
    assert {b.get("action") for b in refused} == {None, "type", "press"}
    recorder = _Recorder()
    asyncio.run(_every_client_call(_client(recorder)))
    assert len(recorder.sent) == 16
    sent = [body for _, _, body in recorder.sent]
    for body in refused:
        assert body not in sent
    assert not any(isinstance(b, dict) and b.get("holder") == "agent" for b in sent)


# ── errors ─────────────────────────────────────────────────────────────────

#: The reason the tool answers each host error with. Each refusal has its
#: own, so none reads as a bad credential unless it is one.
ERROR_REASONS = {
    "unauthorized": "browser_unauthorized",
    "unscoped": "browser_unscoped",
    "not_entitled": "browser_not_entitled",
    "person_only": "browser_person_only",
    "password_field": "browser_secret_field",
    "bad_request": "browser_bad_input",
    "tab_not_found": "browser_error",
    "tab_limit": "browser_error",
    "human_in_control": "browser_human_in_control",
    "stale_element": "browser_stale_element",
    "page_blocked": "browser_page_blocked",
    "page_busy": "browser_page_blocked",
    "daily_limit": "browser_daily_limit",
    "browser_unavailable": "browser_unreachable",
    "entitlements_unavailable": "browser_entitlements_unavailable",
    "out_of_scope": "browser_out_of_scope",
    "forbidden": "browser_forbidden",
    "at_capacity": "browser_at_capacity",
    "approval_required": "browser_step_not_approved",
    "too_many_saved_tabs": "browser_error",
    "browser_starting": "browser_unreachable",
    "too_many_requests": "browser_at_capacity",
    "not_a_text_field": "browser_bad_input",
    "focus_unreadable": "browser_bad_input",
}


def test_every_error_body_is_named_by_its_code_and_mapped():
    errors = _load("error").examples
    assert set(errors) == set(ERROR_REASONS)
    for name, example in errors.items():
        assert example.kind == "response" and example.status >= 400
        assert example.body["error"] == name and example.body["message"]


@pytest.mark.parametrize("code", sorted(ERROR_REASONS))
async def test_each_host_error_reaches_the_model_with_its_own_reason(code, monkeypatch):
    monkeypatch.setattr(client_module, "BUSY_RETRY_S", (0.0, 0.0, 0.0))
    example = _load("error").examples[code]
    client = BrowserClient(
        base_url=ROUTER, credential=_k, trust_env=True, first_call_timeout_s=45.0,
        http=httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(example.status, json=example.body))),
    )
    with pytest.raises(BrowserError) as raised:
        await client.page(session_id=PROFILE)
    assert (raised.value.status, raised.value.code, raised.value.message) == (example.status, code, example.body["message"])

    session = SimpleNamespace(
        _browser_config=BrowserConfig(base_url=ROUTER, profile=PROFILE),
        _browser_state=browser_tool._BrowserState(client=client, session_id=PROFILE),
    )
    result = await browser_tool.handle_browser(session, {"action": "page"})
    assert result.ok is False and result.reason == ERROR_REASONS[code]
    if code not in ("unauthorized",):
        assert "reconnect MindsHub" not in result.content


# ── the session line and the browser block ─────────────────────────────────


def test_the_session_line_is_what_the_cloud_turn_writes():
    from anton.cloud_turn.__main__ import stream_turn
    from anton.core.llm.provider import StreamBrowserSession

    (example,) = _load("session_event").examples.values()

    class _Session:
        async def turn_stream(self, user_input, **kwargs):
            yield StreamBrowserSession(session_id=example.body["session_id"])

        def close(self):
            pass

    lines: list[dict] = []
    request = json.dumps({"protocol_version": 1, "conversation_id": "c", "input": "hi"})
    asyncio.run(stream_turn(request, lines.append, session_builder=lambda r: _Session()))
    assert [line for line in lines if line.get("kind") == "browser"] == [example.body]


@pytest.mark.parametrize(
    "name,example", [(n, e) for _, n, e in _examples("turn_block")], ids=lambda v: v if isinstance(v, str) else "",
)
def test_turn_blocks_read_back_as_the_block_written(name, example):
    config = BrowserConfig.from_dict(example.body)
    if example.refused:
        assert config is None, name
        return
    assert config is not None and config.to_dict() == example.body


@pytest.mark.parametrize(
    "name,example", [(n, e) for _, n, e in _examples("constructor")], ids=lambda v: v if isinstance(v, str) else "",
)
def test_constructor_arguments_build_only_the_two_forms(name, example):
    if example.refused:
        with pytest.raises(ValidationError):
            BrowserConfig(**example.body)
        return
    config = BrowserConfig(**example.body)
    assert config.in_app and "launch-key-example" not in repr(config)
    assert BrowserConfig.from_dict(example.body) is None
    with pytest.raises(ValueError):
        config.to_dict()

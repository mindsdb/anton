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
from pathlib import Path
from types import SimpleNamespace
from typing import Literal

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
)
from anton.core.browser.page import FRAME_TEXT_HEADING, PAGE_TEXT_HEADING, TEXT_PER_FRAME, format_page, human_check

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


def test_the_page_fixture_names_every_frame_limit_the_hosts_hold():
    """Each host pins its frame constants against these, the time a host
    gives a read's frames included, and the card-field rule that keeps a
    frame unread and a focused type out of a field."""
    limits = _load("page").limits
    assert set(limits) == {
        "frames_read", "elements_per_frame", "text_per_frame", "frame_min_read_px", "frames_phase_ms",
        "payment_frame_hosts", "sign_in_frame_hosts", "card_field_name", "card_field_exempt", "card_field_word_breaks",
        "card_field_attributes",
    }
    assert limits["frames_phase_ms"] == 2000
    assert "frames_phase_ms" in _load("page").description


#: The one card-field rule both hosts hold, character for character: a field
#: is named for a card when this, ignoring case, matches what is left of its
#: attributes' values joined with spaces, or of that joined value split into
#: words, once CARD_FIELD_EXEMPT's matches are taken out.
CARD_FIELD_NAME = (
    r"(^|[^a-z0-9]|card|cc)(cvc|cvv2?|csc)([^a-z0-9]|$)|card[\s._-]*(numbers?|num|no)(?![a-z])"
    r"|(^|[^a-z0-9])cc[\s._-]*(numbers?|num)(?![a-z])|security[\s._-]*code"
)
#: The number of a card that isn't a payment card (a gift, loyalty, library or
#: insurance card), read ignoring case. In each form every match is replaced
#: with a space before CARD_FIELD_NAME is tested, so a card's CVV, or a card
#: number in another attribute, still names the field. A qualifier needs no
#: word boundary, so none may occur in a payment card's name
#: (PAYMENT_CARD_NAMES): club is left out, since Diners Club is a payment card.
CARD_FIELD_EXEMPT = (
    r"(gift|loyalty|rewards?|library|members?(hip)?|insurance|health|medicare|medicaid|student|transit|benefits?|sim"
    r"|national[\s._-]*id)[\s._-]*card[\s._-]*(numbers?|num|no)(?![a-z])"
)
#: The word breaks both hosts split the joined value at, applied in order and
#: case-sensitive, each replacing every match with its group and a space
#: (JavaScript's "$1 "): cardNumberInput reads as card Number Input, and
#: CVVCode as CVV Code.
CARD_FIELD_WORD_BREAKS = [r"([a-z0-9])(?=[A-Z])", r"([A-Z])(?=[A-Z][a-z])"]
#: The attributes both hosts read a field's name from, in the order they join them.
CARD_FIELD_ATTRIBUTES = ["name", "id", "aria-label", "placeholder"]
#: Names that must stay named for a card (on the page its value is hidden and
#: a type, select or editing key answers 403 password_field, a focused type
#: answers the same, and a frame holding one is never read) and names that
#: must not, so a host types into them.
CARD_NAMED = [
    "cardnumber", "card_number", "card.number", "cardNo", "card_no", "cc_num", "cc-number", "cvc", "CVC", "cvv2",
    "card_cvc", "cardCvc", "cc_cvv", "billing_cvv2", "cvv_code", "csc", "security_code", "Card number", "Security code",
    "cardNumberInput", "CardNumberTextBox", "ctl00_Main_CardNumberTextBox", "ccNumberInput", "creditCardNumberInput",
    "cardNumberField", "cardNumbers", "cvvInput", "txtCVV", "CVVCode", "CCNumber",
]
#: Payment cards' names, each card-named on both hosts. None holds a match of
#: CARD_FIELD_EXEMPT in either form: its qualifiers need no word boundary, so a
#: qualifier inside one of these would take a payment card for an ordinary
#: card. giftCardCvv is a gift card's CVV, which the exempt pattern never takes.
PAYMENT_CARD_NAMES = [
    "creditCardNumber", "prepaidCardNumber", "storeCardNumber", "fleetCardNumber", "businessCardNumber",
    "debitCardNumber", "cardNumberInput", "giftCardCvv", "paymentCardNumber", "virtualCardNumber",
    "corporateCardNumber", "bankCardNumber", "chargeCardNumber", "purchasingCardNumber", "procurementCardNumber",
    "travelCardNumber", "cc-number", "card_number", "visaCardNumber", "mastercardNumber", "amexCardNumber",
    "discoverCardNumber", "dinersClubCardNumber", "Diners Club card number", "jcbCardNumber", "unionPayCardNumber",
    "maestroCardNumber", "ebtCardNumber",
]
NOT_CARD_NAMED = [
    "name", "email", "acc_number", "account_number", "cardholder_name", "card_type", "cardnotes", "discount_code",
    "promo_code", "postal_code", "recv_count", "Name on card",
    "cardholderName", "cardNotes", "cardNoteText", "cardType", "CardHolder",
]
#: Numbers of cards that aren't payment cards, which a host reads and types
#: into: at least one for each qualifier of CARD_FIELD_EXEMPT, in the casings
#: forms give their fields.
ORDINARY_CARD_NUMBERS = [
    "giftCardNumber", "txtGiftCardNumber", "ctl00_Main_GiftCardNumberTextBox", "loyalty_card_number",
    "Library card number", "membershipCardNumber", "insuranceCardNumber", "nationalIdCardNumber", "rewardsCardNumber",
    "memberCardNo", "health_card_number", "Medicare card number", "medicaidCardNumber", "studentCardNumber",
    "transitCardNum", "benefitsCardNumber", "simCardNumber",
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


def _name_forms(attributes: dict[str, str]) -> tuple[str, str]:
    """A field's two name forms as page.json gives them: the attributes'
    values joined in order with a space, a missing one empty, and that joined
    value's word-split form."""
    limits = _load("page").limits
    joined = " ".join(attributes.get(name, "") for name in limits["card_field_attributes"])
    return joined, _word_split(joined, limits["card_field_word_breaks"])


def _card_named(attributes: dict[str, str]) -> bool:
    """The rule as page.json gives it: in each name form, every match of the
    exempt pattern replaced with a space, ignoring case, then the card
    pattern, ignoring case, on what is left of either form."""
    limits = _load("page").limits
    left = (re.sub(limits["card_field_exempt"], " ", form, flags=re.IGNORECASE) for form in _name_forms(attributes))
    return any(re.search(limits["card_field_name"], form, re.IGNORECASE) for form in left)


def test_the_page_fixture_pins_the_one_card_field_rule_both_hosts_hold():
    """Both hosts hide a card-named field's value, refuse a type into it and
    never read a frame that holds one, by this one rule, so a name gets the
    same answer on either host. Each value is named for a card, or not, in
    any of the attributes alone, and one field is named only by two
    attributes together, which pins the join. The camelCase names an ASP.NET
    or React form gives its card fields are named only by the word split."""
    page = _load("page")
    assert page.limits["card_field_name"] == CARD_FIELD_NAME
    assert page.limits["card_field_exempt"] == CARD_FIELD_EXEMPT
    assert page.limits["card_field_word_breaks"] == CARD_FIELD_WORD_BREAKS
    assert page.limits["card_field_attributes"] == CARD_FIELD_ATTRIBUTES
    assert (
        "(name, id, aria-label and placeholder) joined in that order with a space, a missing one as empty."
    ) not in page.description
    assert "A field is card-named when card_field_name, ignoring case, matches the values of" not in page.description
    assert (
        "A field's two name forms are the values of its card_field_attributes (name, id, aria-label and placeholder) "
        "joined in that order with a space, a missing one as empty, and that joined value's word-split form: each of "
        "card_field_word_breaks applied in order, case-sensitive, replacing every match with its first group and a "
        "space ($1 ), so cardNumberInput reads as card Number Input and CVVCode as CVV Code. In each form, every "
        "match of card_field_exempt, ignoring case, is replaced with a space, and the field is card-named when "
        "card_field_name, ignoring case, matches what is left of either form. The exempt pattern matches the number "
        "of a card that isn't a payment card, such as a gift, loyalty, library or insurance card, so giftCardNumber "
        "and 'Library card number' are ordinary fields, while giftCardCvv and a field named cardNumber with the "
        "aria-label 'Gift card number' stay card-named."
    ) in page.description
    for value in NAMED_ONLY_WHEN_SPLIT:
        assert re.search(CARD_FIELD_NAME, value, re.IGNORECASE) is None, value
        assert re.search(CARD_FIELD_NAME, _word_split(value, CARD_FIELD_WORD_BREAKS), re.IGNORECASE), value
    assert re.search(CARD_FIELD_NAME, _word_split("CVVCode", CARD_FIELD_WORD_BREAKS[:1]), re.IGNORECASE) is None
    named = page.examples["values that name a field for a card"]
    unnamed = page.examples["values that don't name a field for a card"]
    for example, card_named, required in (
        (named, True, CARD_NAMED + PAYMENT_CARD_NAMES),
        (unnamed, False, NOT_CARD_NAMED + ORDINARY_CARD_NUMBERS),
    ):
        assert (example.kind, example.where) == ("classification", "a field's name, id, aria-label or placeholder")
        assert example.body["card_named"] is card_named
        values = example.body["values"]
        assert set(required) <= set(values) and len(set(values)) == len(values)
        for value in values:
            for attribute in CARD_FIELD_ATTRIBUTES:
                assert _card_named({attribute: value}) is card_named, (attribute, value)
    together = page.examples["a field named for a card only by two of its attributes together"]
    assert together.body["card_named"] is True
    attributes = together.body["attributes"]
    assert _card_named(attributes)
    assert not any(_card_named({name: value}) for name, value in attributes.items())


def test_the_exempt_pattern_takes_out_only_an_ordinary_card_s_number_and_never_a_payment_card_s_name():
    """A gift, loyalty, library or insurance card's number is an ordinary
    field on both hosts: readable, and typed into. The exempt pattern's
    qualifiers need no word boundary (txtGiftCardNumber), so each payment
    name in page.json is checked for a match in both forms: with club among
    the qualifiers, Diners Club's names failed here. A host that skips the
    exempt step reads giftCardNumber as card-named; one that skips it on the
    split form reads ctl00_Main_GiftCardNumberTextBox so; one that drops the
    field when the exempt pattern matches anywhere, instead of taking the
    match out, misses cardNumber beside the label 'Gift card number'."""
    page = _load("page")
    exempt = page.limits["card_field_exempt"]
    for value in page.examples["values that name a field for a card"].body["values"]:
        for attribute in CARD_FIELD_ATTRIBUTES:
            for form in _name_forms({attribute: value}):
                assert re.search(exempt, form, re.IGNORECASE) is None, (attribute, value, form)
    for value in ORDINARY_CARD_NUMBERS:
        assert any(re.search(exempt, form, re.IGNORECASE) for form in _name_forms({"name": value})), value
    assert re.search(CARD_FIELD_NAME, "giftCardNumber", re.IGNORECASE)
    joined, split = _name_forms({"name": "ctl00_Main_GiftCardNumberTextBox"})
    assert re.search(exempt, joined, re.IGNORECASE) is None and re.search(CARD_FIELD_NAME, split, re.IGNORECASE)

    beside = page.examples["a card field whose aria-label names a gift card"]
    assert (beside.kind, beside.where) == ("classification", "a field's name, id, aria-label or placeholder")
    assert beside.body == {
        "card_named": True, "attributes": {"name": "cardNumber", "aria-label": "Gift card number"},
    }
    assert _card_named(beside.body["attributes"])
    assert _card_named({"name": "cardNumber"}) and not _card_named({"aria-label": "Gift card number"})
    assert any(re.search(exempt, form, re.IGNORECASE) for form in _name_forms(beside.body["attributes"]))


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


def test_the_page_fixture_places_each_frame_on_its_page_whatever_scrolls():
    """A frame's box is where it is on the screenshot, so the page's scroll
    and an inner column's both move it. at is where the frame element sits in
    its document, so neither does, and it tells a frame that moved on the
    page from one that only scrolled. A host sends it with every frame it
    places, in whole CSS pixels, and a frame with no box carries none."""
    fixture = _load("page")
    assert (
        "A host sends at with every frame whose box isn't null: the frame element's top-left {x, y} in its "
        "document's content coordinates, in CSS pixels rounded to whole ones, which is its top-left in the viewport "
        "plus the page's scroll plus the scroll offset of every scrolling element it sits in, so no scroll, the "
        "page's or an inner column's, changes it."
    ) in fixture.description
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


def test_the_control_fixture_tells_done_by_the_person_from_any_other_agent_record():
    controls = _load("control").examples
    fresh = SessionInfo.model_validate(controls["a session the agent holds"].body).control
    handed = Control.model_validate(controls["handed to the person"].body)
    done = SessionInfo.model_validate(controls["done by the person"].body).control
    assert (handed.holder, handed.by) == ("human", "agent")
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
        session_id=PROFILE, reason="Check the unaccompanied-minor policy box", wait_for={"element": 12, "state": "checked"},
    ),
    ("control", "hand to the person, waiting for the page to change"): lambda c: c.hand_to_person(
        session_id=PROFILE, reason="Please sign in to Example Shop", wait_for={"page": "changes"},
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

"""Tests for the shared speech-safety sanitiser (clarif_eye.speech).

THE OUTPUT IS READ ALOUD to a blind user who cannot cross-check it, so the
overriding rule under test here is: ordinary content (filenames, passwords,
product codes, arithmetic) must survive completely unchanged, and only
actual markup/tag syntax gets removed.
"""

from clarif_eye.speech import to_spoken_text


# --- FIX 1: lone `_`/`*` must never be stripped -----------------------------


def test_filename_with_underscores_is_preserved_verbatim():
    assert (
        to_spoken_text("The file is named my_file_name.txt on the label.")
        == "The file is named my_file_name.txt on the label."
    )


def test_arithmetic_with_lone_asterisk_is_preserved_verbatim():
    assert to_spoken_text("5 * 3 = 15 written on a chalkboard.") == "5 * 3 = 15 written on a chalkboard."


def test_wifi_password_with_underscores_is_preserved_verbatim():
    assert to_spoken_text("WIFI_PASSWORD_2026") == "WIFI_PASSWORD_2026"


def test_product_code_with_underscores_is_preserved_verbatim():
    assert to_spoken_text("model_no_A4471") == "model_no_A4471"


def test_paired_double_asterisk_bold_still_collapses():
    assert to_spoken_text("This is **bold** text.") == "This is bold text."


def test_paired_underscore_italic_still_collapses():
    assert to_spoken_text("This is _italic_ text.") == "This is italic text."


def test_paired_double_underscore_bold_still_collapses():
    assert to_spoken_text("This is __bold__ text.") == "This is bold text."


def test_paired_single_asterisk_italic_still_collapses():
    assert to_spoken_text("This is *italic* text.") == "This is italic text."


def test_repeated_multiplication_with_spaces_is_preserved():
    # Two lone asterisks, both used as multiplication signs with spaces on
    # both sides - neither forms a valid paired emphasis marker.
    assert to_spoken_text("5 * 3 * 4 = 60") == "5 * 3 * 4 = 60"


# --- FIX 2: markdown links --------------------------------------------------


def test_inline_markdown_link_keeps_text_drops_url():
    result = to_spoken_text("Check [our menu](https://example.com/menu) for prices.")
    assert result == "Check our menu for prices."
    assert "example.com" not in result
    assert "[" not in result and "]" not in result
    assert "(" not in result and ")" not in result


def test_reference_style_link_keeps_text_drops_reference():
    result = to_spoken_text("Check [our menu][1] for prices.")
    assert result == "Check our menu for prices."
    assert "[" not in result and "]" not in result


# --- FIX 3: tables -----------------------------------------------------------


def test_table_becomes_connected_row_phrases_not_comma_soup():
    table = "| Item | Price |\n|------|-------|\n| Burger | $8.00 |\n"

    result = to_spoken_text(table)

    # Exact pin: the separator row contributes NOTHING - not dashes, not an
    # extra empty sentence - between the two real data rows.
    assert result == "Item, Price. Burger, $8.00."
    assert ", ," not in result
    assert not result.startswith(",")
    assert not result.endswith(",")


def test_table_with_empty_cells_does_not_produce_bare_comma_runs():
    table = "| Item | Note |\n|------|------|\n| Burger |  |\n"

    result = to_spoken_text(table)

    assert ", ," not in result
    assert not result.startswith(",")
    assert not result.endswith(",")


# --- FIX 4: HTML -------------------------------------------------------------


def test_html_tags_are_stripped_keeping_inner_text():
    result = to_spoken_text("The sign says <b>OPEN</b> and <i>24 hours</i>.")
    assert result == "The sign says OPEN and 24 hours."
    assert "<" not in result and ">" not in result


def test_genuine_less_than_sign_is_preserved():
    assert to_spoken_text("5 < 10") == "5 < 10"


# --- Already-clean prose is untouched ---------------------------------------


def test_already_clean_prose_is_unchanged():
    text = "The image shows a coffee cup label that reads Open 9 to 5."
    assert to_spoken_text(text) == text

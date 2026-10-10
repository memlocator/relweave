from relweave.tidy import tidy


def test_joins_wrapped_lines_and_hyphenated_words():
    raw = "Acme Corp, a maker of tur-\nbines, said its chief executive\nJane Roe would step down.\n\nShe joined in 2001."
    assert tidy(raw) == "Acme Corp, a maker of turbines, said its chief executive Jane Roe would step down.\n\nShe joined in 2001."


def test_drops_page_numbers_and_running_headers():
    pages = [f"ACME ANNUAL REPORT\n\nThe group sold stake number {k}.\nPage {k}" for k in (1, 2, 3)]
    out = tidy("\n\n".join(pages))
    assert "ACME ANNUAL REPORT" not in out and "Page" not in out
    assert all(f"The group sold stake number {k}." in out for k in (1, 2, 3))


def test_keeps_a_sentence_repeated_without_page_numbers():
    assert tidy("Prices rose.\n\nPrices rose.\n\nPrices rose.").count("Prices rose.") == 3


def test_keeps_list_items_and_decodes_entities():
    out = tidy("Board members:\n• Jane Roe\n• John Doe\nJohnson &amp; Johnson owns it .")
    assert out.splitlines()[:3] == ["Board members:", "• Jane Roe", "• John Doe"]
    assert out.endswith("Johnson & Johnson owns it.")

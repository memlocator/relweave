from relweave.tables import verbalize


def test_rows_carry_their_headers():
    t = "Board below.\n\n| Name | Title |\n|---|---|\n| Ana Varga | Chair |\n| Ben Okafor | CFO |\n\nThe end."
    assert verbalize(t) == "Board below.\n\nName: Ana Varga; Title: Chair.\nName: Ben Okafor; Title: CFO.\n\nThe end."


def test_repeated_header_after_page_break_is_dropped():
    t = "| Name | Title |\n|---|---|\n| Ana Varga | Chair |\n| Name | Title |\n| Ben Okafor | CFO |"
    assert verbalize(t) == "Name: Ana Varga; Title: Chair.\nName: Ben Okafor; Title: CFO."


def test_two_columns_without_separator_are_key_value():
    t = "| Founded | 1998 |\n| Headquarters | Ghent |"
    assert verbalize(t) == "Founded: 1998.\nHeadquarters: Ghent."


def test_no_separator_uses_first_row_as_header():
    t = "| City | Subsidiary | Staff |\n| Lyon | Zelland Textiles | 642 |"
    assert verbalize(t) == "City: Lyon; Subsidiary: Zelland Textiles; Staff: 642."

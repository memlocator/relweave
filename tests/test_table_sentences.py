from relweave.formats import sentence_bounds


def test_each_table_row_is_a_sentence():
    t = "The board is listed below.\n\n| Name | Role |\n|---|---|\n| Ana Varga | Chair |\n| Ben Okafor | CFO |\n\nThe end."
    assert [t[a:b] for a, b in sentence_bounds(t)] == [
        "The board is listed below.\n\n", "| Name | Role |", "| Ana Varga | Chair |", "| Ben Okafor | CFO |",
        "\n\nThe end."]


def test_prose_without_tables_is_unchanged():
    t = "Acme hired Jane Doe. She moved to Oslo. Mr. Smith stayed."
    assert [t[a:b] for a, b in sentence_bounds(t)] == ["Acme hired Jane Doe.", "She moved to Oslo.", "Mr. Smith stayed."]

import pytest

from relweave.serving import HEAD_GB, plan, resident_share


def test_auto_picks_the_most_precise_format_that_fits_with_a_cache():
    assert plan(23.0, 24.0).format == "bf16"
    assert plan(10.0, 12.0).format == "fp8"
    assert plan(7.3, 8.0).format == "4bit"  # an 8 GB desktop card
    with pytest.raises(RuntimeError):
        plan(3.0, 4.0)


def test_parallelism_is_maximal_unless_the_cache_is_small():
    assert plan(23.0, 24.0).max_num_seqs == 256
    assert plan(7.3, 8.0).max_num_seqs == 32
    assert plan(7.3, 8.0, max_num_seqs=8).max_num_seqs == 8  # a knob wins
    assert plan(23.0, 24.0, fmt="4bit").format == "4bit"


def test_the_head_stays_resident_only_on_a_big_card():
    assert resident_share(7.3, 8.0) is None
    share = resident_share(23.0, 24.0)
    assert share is not None and share * 24.0 <= 23.0 - HEAD_GB


def test_a_served_lora_generator_is_asked_by_name():
    from relweave.remote import pick_model
    assert pick_model(["unsloth/qwen3-4b-unsloth-bnb-4bit", "generator"]) == "generator"
    assert pick_model(["chrullis/relweave-4b-base"]) == "chrullis/relweave-4b-base"

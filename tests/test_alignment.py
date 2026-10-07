from egip.alignment import diff_spans, masked_source_token_indices
from egip.factuality import life_format_evidence


def test_life_mask_alignment() -> None:
    source = "The Second Punic War ended in 201 BC."
    masked = "The Second * * ended in * BC ."
    indices, accuracy = masked_source_token_indices(source, masked)
    assert indices == {2, 3, 6}
    assert accuracy == 1.0


def test_diff_targets_error_value() -> None:
    result = diff_spans(
        "The Second Punic War ended in 201 BC.",
        "The Second Punic War ended in 301 BC.",
    )
    assert [span.text for span in result.target_spans] == ["301"]
    assert result.operations[0]["tag"] == "replace"


def test_life_verifier_evidence_format() -> None:
    assert life_format_evidence(["Julianne_Moore\tFrom -LRB- 1985 -RRB-."]) == (
        "title: Julianne Moore context: From ( 1985 )."
    )

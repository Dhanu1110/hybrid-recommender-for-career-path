from src.ingest.text_to_esco_mapper import map_text_to_occupations


def test_low_confidence_mapping_returns_none():
    # Use a very high threshold so any realistic match is rejected
    esco_id, score = map_text_to_occupations("totally-nonsense-unique-string-xyz-12345", top_k=1, score_threshold=0.99)
    assert esco_id is None

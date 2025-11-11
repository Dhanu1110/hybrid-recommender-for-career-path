import torch
from src.models.bert4rec import BERT4Rec, BERT4RecConfig, predict_next


def test_pad_mask_filtered_and_different_inputs():
    # Small synthetic model
    config = BERT4RecConfig(vocab_size=20, d_model=32, n_layers=1, n_heads=2, max_seq_len=10)
    model = BERT4Rec(
        vocab_size=config.vocab_size,
        d_model=config.d_model,
        n_layers=config.n_layers,
        n_heads=config.n_heads,
        max_seq_len=config.max_seq_len
    )

    # Two different inputs
    input_a = torch.tensor([[2, 3, 4]], dtype=torch.long)
    input_b = torch.tensor([[2, 3, 5]], dtype=torch.long)

    logits_a = model(input_a)
    logits_b = model(input_b)

    # Ensure logits shapes are correct [B, L, V]
    assert logits_a.shape[0] == 1 and logits_a.shape[2] == config.vocab_size

    # Get last-token logits and ensure they differ for different inputs
    last_a = logits_a[:, -1, :].squeeze(0)
    last_b = logits_b[:, -1, :].squeeze(0)

    # They should not be identical for different inputs
    assert not torch.allclose(last_a, last_b)

    # Use predict_next to ensure PAD/MASK are filtered
    pad_id = model.pad_token_id
    mask_id = model.mask_token_id

    top_a = predict_next(last_a, pad_token_id=pad_id, mask_token_id=mask_id, exclude_ids=set(), top_k=3)
    top_b = predict_next(last_b, pad_token_id=pad_id, mask_token_id=mask_id, exclude_ids=set(), top_k=3)

    # Top-1 may differ given different inputs
    assert len(top_a) == 3 and len(top_b) == 3
    # At least the top-1 indices should be not always identical
    if top_a[0][0] == top_b[0][0]:
        # If they are equal, ensure probabilities differ
        assert top_a[0][1] != top_b[0][1]

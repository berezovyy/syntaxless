"""Direct checks for the tagger: padding neutrality and QAT finite gradients.

Padding neutrality is the property that makes length bucketing safe: appending
unused rows must not change a single logged probability for the real tokens.
Finite gradients under QAT prove the straight-through estimator keeps training.
Run with `python training/cron/test_model.py`.
"""
from __future__ import annotations

import torch

from model import Tagger

FEATURE_ROWS = 580
ROWS_PER_TOKEN = 17
ROLE_CLASSES = 15
HIDDEN = 16


def make_batch(content_length: int, padding_length: int, seed: int = 0):
    """A single sequence padded with `FEATURE_ROWS` rows and masked `False`."""
    generator = torch.Generator().manual_seed(seed)
    width = content_length + padding_length

    rows = torch.full((1, width, ROWS_PER_TOKEN), FEATURE_ROWS, dtype=torch.long)
    content = torch.randint(
        0, FEATURE_ROWS, (content_length, ROWS_PER_TOKEN), generator=generator
    )
    rows[0, :content_length] = content

    valid = torch.zeros(1, width, dtype=torch.bool)
    valid[0, :content_length] = True

    neighbors = torch.full((1, width, 2), -1, dtype=torch.long)
    for position in range(content_length):
        if position > 0:
            neighbors[0, position, 0] = position - 1
        if position + 1 < content_length:
            neighbors[0, position, 1] = position + 1
    return rows, valid, neighbors


def build_model() -> Tagger:
    torch.manual_seed(0)
    return Tagger(
        feature_rows=FEATURE_ROWS,
        rows_per_token=ROWS_PER_TOKEN,
        role_classes=ROLE_CLASSES,
        hidden=HIDDEN,
    )


def test_padding_neutrality() -> None:
    model = build_model().eval()
    content = 6
    base = make_batch(content, 0, seed=7)
    padded = make_batch(content, 9, seed=7)
    assert base[0].shape[1] != padded[0].shape[1]

    for qat in (False, True):
        model.qat = qat
        with torch.no_grad():
            base_roles, base_boundary = model(*base)
            padded_roles, padded_boundary = model(*padded)
        assert torch.allclose(
            base_roles[0, :content],
            padded_roles[0, :content],
            atol=1e-5,
            rtol=1e-4,
        ), f"role logits changed under padding (qat={qat})"
        assert torch.allclose(
            base_boundary[0, :content],
            padded_boundary[0, :content],
            atol=1e-5,
            rtol=1e-4,
        ), f"boundary logits changed under padding (qat={qat})"
    print(f"padding neutrality: ok (content={content}, padded width {padded[0].shape[1]})")


def test_finite_gradients() -> None:
    model = build_model().train()
    model.qat = True
    batch = make_batch(8, 5, seed=11)
    roles, boundary = model(*batch)

    target = torch.randint(0, ROLE_CLASSES, roles.shape[:-1])
    role_loss = torch.nn.functional.cross_entropy(
        roles.reshape(-1, ROLE_CLASSES), target.reshape(-1)
    )
    boundary_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        boundary, torch.zeros_like(boundary)
    )
    (role_loss + boundary_loss).backward()

    checked = 0
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        assert torch.isfinite(parameter.grad).all(), f"non-finite gradient: {name}"
        checked += 1
    assert checked > 0, "no parameter received a gradient"
    print(f"finite qat gradients: ok ({checked} tensors)")


if __name__ == "__main__":
    test_padding_neutrality()
    test_finite_gradients()
    print("test_model: all checks passed")

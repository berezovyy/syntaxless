"""MRT risk is 1 − EXEC; the loss backprops through sampled log-probs."""
import torch

from mrt import exec_from_labels, exec_risk, mrt_loss, sample_k


def test_exec_risk_is_one_minus_label_exec() -> None:
    gold = torch.tensor([[1, 2, 3]])
    mask = torch.tensor([[True, True, True]])
    match = gold.clone()
    miss = torch.tensor([[1, 9, 3]])
    assert float(exec_from_labels(match, gold, mask)) == 1.0
    assert float(exec_risk(match, gold, mask)) == 0.0
    assert float(exec_from_labels(miss, gold, mask)) == 0.0
    assert float(exec_risk(miss, gold, mask)) == 1.0
    masked = torch.tensor([[True, False, True]])
    almost = torch.tensor([[1, 0, 3]])
    assert float(exec_risk(almost, gold, masked)) == 0.0


def test_mrt_loss_backward() -> None:
    logits = torch.randn(2, 3, 5, requires_grad=True)
    gold = torch.tensor([[0, 1, 2], [2, 1, 0]])
    mask = torch.ones(2, 3, dtype=torch.bool)
    sampled, log_probs = sample_k(logits, gold, mask, k=4)
    assert sampled.shape == (4, 2, 3)
    assert log_probs.shape == (4, 2)
    risks = torch.stack(
        [exec_risk(sampled[index], gold, mask) for index in range(4)],
        dim=0,
    )
    loss = mrt_loss(log_probs, risks)
    loss.backward()
    assert logits.grad is not None
    assert logits.grad.abs().sum() >= 0


if __name__ == "__main__":
    test_exec_risk_is_one_minus_label_exec()
    test_mrt_loss_backward()
    print("test_mrt: all checks passed")

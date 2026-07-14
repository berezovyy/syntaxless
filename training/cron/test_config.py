"""Focused command-line checks for corpus population mixing."""
from __future__ import annotations

from config import parse_args


def test_population_mix_is_recorded_in_config() -> None:
    config = parse_args(["--device", "cpu", "--population-mix", "68,22,10"])
    assert config.population_mix == (68.0, 22.0, 10.0)
    assert config.to_dict()["population_mix"] == (68.0, 22.0, 10.0)


def test_population_mix_rejects_ambiguous_combinations() -> None:
    for conflicting_flag in (
        "--balance-frames",
        "--balance-kinds",
        "--balance-schedule-kinds",
    ):
        try:
            parse_args(
                [
                    "--device",
                    "cpu",
                    "--population-mix",
                    "0.68,0.22,0.10",
                    conflicting_flag,
                ]
            )
        except ValueError as error:
            assert "cannot be combined" in str(error)
        else:
            raise AssertionError(f"population mix accepted {conflicting_flag}")


def test_population_mix_rejects_invalid_weights() -> None:
    for value in ("1,2", "1,-1,1", "0,0,0", "nan,1,1"):
        try:
            parse_args(["--device", "cpu", "--population-mix", value])
        except ValueError as error:
            assert "--population-mix" in str(error)
        else:
            raise AssertionError(f"accepted invalid population mix: {value}")

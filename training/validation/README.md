# Validation exporter

`export.py` exports a validation checkpoint as an int6 candidate. Unlike the
other packages, validation calibrates the clause-boundary threshold at export
time (worst-case boundary F1 on the validation split). Everything else comes
from the [shared harness](../shared/README.md): training runs through
`training/shared/train.py --package validation`.

```sh
python training/validation/export.py --checkpoint <run>/best.pt --out <run>/candidate
```

The candidate must be inside `training/validation/runs/`. The script never
promotes; see [validation training](../../packages/validation/training/README.md)
for the full pipeline and gate.

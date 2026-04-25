# Changelog

## 0.2.0 — 2026-04-25

### License change (breaking for commercial users)

Spec2Function is now distributed under the **PolyForm Noncommercial License
1.0.0**. Noncommercial use (research, education, nonprofit / government
organizations, personal study) remains permitted. **Commercial use now
requires a separate license** — contact the author.

Versions 0.1.x and earlier were released under the MIT License and remain MIT.
This change applies only to 0.2.0 and later.

### Added

- `examples/demo_single.json`, `examples/demo_set.csv`, `examples/run_demo.py`:
  bundled demo data and a runnable end-to-end script.
- `single_analysis.SingleSpectrumAnalyzer`, `set_analysis.MetaboliteSetAnalyzer`,
  and the `MS2BioTextWorkflow` / `run_single` / `run_set` entry points are now
  exported from the top-level `Spec2Function` package.
- README rewritten to follow the System requirements / Installation / Demo /
  Instructions for use structure (with pinned dependency versions, tested OS
  list, install/runtime estimates, and expected output).

### Changed

- `setup.py`: pinned minimum versions for runtime dependencies; moved `wandb`
  to `extras_require["train"]`; bumped Python floor to 3.10.

### Removed

- Stale `Spec2Function/.pypi_tmp/` build artifacts.
- `Spec2Function/gpt_inference.py.bak_*` backup file.

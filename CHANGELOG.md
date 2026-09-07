# Changelog

## 0.2.0 — Unreleased

- Introduce the `pynamo_egt` package namespace and a convenient top-level API.
- Replace standalone module imports with package imports throughout the tutorials and tests. Old imports such as `import drawer` are no longer supported.
- Keep the optional notebook widget module lazily loaded.

## 0.1.2 — 2026-09-07

- Preserve Colab’s preinstalled dependencies by installing only the core package without forced reinstallation or notebook extras.

- Fix ESS classification at boundary rest points when unused strategies tie as best responses. The test now includes all feasible tied-mutant invasion directions.
- Correctly flag all-zero games as having non-isolated replicator rest points.
- Clarify that Nash, strict Nash, and ESS classifications annotate detected replicator rest points; they do not enumerate additional static equilibria.
- Add regression tests for boundary and mixed ESS, strategy-order invariance, degenerate games, and preservation of the detected rest-point set.
- Update both tutorials with references and a streamlined ending, and clear saved outputs and execution counts.
- Synchronize package and citation metadata at version 0.1.2.

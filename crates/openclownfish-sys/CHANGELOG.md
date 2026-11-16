# Changelog

Changes to `openclownfish-sys`. The repository `CHANGELOG.md` also covers the unpublished `openclownfish-clc`.

## 0.1.5

First release with content: the crates.io versions before this (0.0.0) only reserved the name. The runtime loader (`OpenCl::load`, 41 entry points each an `Option`), the `opencl-sys` re-export as `types`, and the `icd` module (Windows registry and Linux `*.icd` registration reading, `check_load`). Adds a README stating where it has been tested, `rust-version` 1.88 and this changelog. Earlier 0.0.x to 0.1.4 versions were never published; their history is in the repository `CHANGELOG.md`.

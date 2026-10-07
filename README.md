# HiCARE ARI registry manuscript

**Facility profiles and stratified calibration for mortality risk monitoring**  
*A national acute respiratory infection hospitalisation registry study in Ecuador*

[Read the English main manuscript](main.pdf) · [Read the Supplementary Appendix](appendix.pdf)

This repository contains the English LaTeX manuscript, its figures and numerical tables, and accompanying analysis code. The main manuscript has 30 pages; the appendix contains the detailed methods, complete tables and additional interpretation. The previous cohort-selection flowchart and its citations have been removed. The remaining figures are renumbered as Figures 1–27 and S1–S6. All 14 tables appear as Tables S1–S14 in the appendix.

## Build the paper

Install Python 3 and a LaTeX distribution with **XeLaTeX** (MiKTeX or TeX Live). Required packages include fontspec, amsmath, amssymb, graphicx, booktabs, longtable, tabularx, array, geometry, caption, hyperref, needspace, pdflscape. Times New Roman and Arial are used when installed, with Latin Modern fallbacks.

From the repository root:

```sh
python scripts/compile.py
```

The script compiles both documents twice and writes `main.pdf` and `appendix.pdf`. An explicit executable path can be supplied with `--xelatex PATH`. LaTeX source is directly editable; alternatively, edit the checked English JSON in `source/` and run:

```sh
python scripts/compile.py --regenerate
```

Regeneration overwrites `main.tex` and `appendix.tex` from those JSON files. Figure 1 embeds the supplied vector PDF `figures/risk_monitoring_supplied.pdf` (original `2.pdf`); Figure 3 embeds `figures/hicare_methods_supplied.pdf` (original `1.pdf`). Their byte-level provenance is recorded in `source/supplied_figure_provenance.json`. The earlier TikZ files `x01_architecture.tex` and `x34_hicare_detailed.tex` are legacy versions and are not used by the current manuscript. The optional `--diagrams` flag rebuilds only those legacy assets and requires TikZ/standalone. All other included plots are original English vector PDFs with unchanged plotted numerical results. The paper can be built without downloading any admission records or rerunning model training.

## Repository contents

- `main.tex`, `appendix.tex` and their compiled PDFs.
- `figures/` — 33 figures selected by the manifest, including two supplied vector methodological schematics; earlier schematic assets are retained for reference.
- `source/` — checked English manuscript blocks, table cells and figure manifest.
- `code/` — original cohort, feature, model, evaluation and plotting implementations. Selected legacy explanatory comments were aligned with the manuscript; analysis algorithms were not changed for this translation.
- `results/v2/tables/` and `results/tables/` — archived aggregate numerical outputs, with English analytical variable names. These are source results, not newly estimated results.
- `scripts/` — manuscript generation, compilation and validation.

## Data and analysis

The source study uses the publicly released Ecuador INEC *Registro Estadístico de Camas y Egresos Hospitalarios*, 2015–2023. The final ARI cohort contains 576,195 admissions and 41,048 deaths. Historical facility profiles were derived from 9,463,108 non-ARI discharges.

**Admission-level files, individual prediction files, fitted patient-level caches and third-party article PDFs are not redistributed.** The manuscript and provided aggregate tables are sufficient to inspect reported results. Full model reruns require the INEC files, compatible dependencies and local computational resources. See the download URLs and expected directory layout in `code/download_inec.py`, `code/build_cohort.py` and `code/build_population.py`. Population workbooks and some legacy reference-validation inputs require separate acquisition.

Analysis entry points include:

| Stage | Implementation |
|---|---|
| Source registry acquisition and harmonisation | `download_inec.py`, `build_cohort.py`, `cohort.py` |
| Population denominators | `build_population.py` |
| Historical context and facility profiles | `features.py`, `v2_profiles.py` |
| Base models and evaluation splits | `models.py`, `tabm_model.py`, `run_cv.py` |
| Nested predictions and calibration | `v2_nested.py`, `v2_stack.py` |
| Independent standardisation and demand analyses | `v2_medical.py`, `v2_extra.py` |
| Original result plots | `v2_figures.py`, `make_figures.py` |

The analysis scripts retain their historical filenames, internal model names and output paths to preserve correspondence with archived results. Legacy labels `reference`/`Reference LR` mean LR-7 and `CARE v1` means EW. Geographic “HiCARE” denotes a profile-ensemble configuration, not the full calibration workflow. The old plotting entry points may also generate legacy method schematics or the removed cohort flowchart; the authoritative current manuscript figure set is `source/figure_manifest.json`.

The translation release has been compiled and checked, but the full training pipeline has **not** been rerun in this release. `requirements-analysis.txt` lists dependencies without claiming a recovered, fully pinned original environment. Manuscript building requires no analysis dependencies.

## Interpretation and study status

Prediction and population monitoring use **independent expected values**. Group calibration constraints apply to the fitting data; historical aggregates are precomputed; reporting delays are not strictly simulated. Geographic comparisons change both profile availability and ensemble membership. These conditions and the retrospective scope are described in the manuscript and Appendix S1.

This is a research revision, not a statement of peer-review acceptance. Author names, contributions, ethics approval or exemption, funding and competing-interest declarations must be completed by the study authors. Public repository visibility does not replace source-data terms or grant rights over third-party publications.

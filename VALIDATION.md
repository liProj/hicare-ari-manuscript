# English release validation

- Main manuscript: **30 pages**, 27 figures, six displayed equations and 35 references.
- Supplementary Appendix: **29 pages**, six figures, 14 tables and seven displayed equations.
- The original cohort-selection flowchart and its main/appendix references are absent.
- Figure numbering and all English textual references were shifted consistently after deletion.
- **1,382 non-Chinese table cells, including the numerical result cells, were compared directly with the source inventory and are unchanged.** Translated labels were checked against the original definitions.
- All 31 retained result and nested-fitting plots preserve their original English vector PDFs. Figures 1 and 3 embed the supplied vector PDFs without modification, mapped from original 2.pdf and 1.pdf, respectively. Their captions and corresponding descriptions in Sections 3.2, 3.4 and Appendix S4 have been updated. Source-file SHA-256 values are recorded in source/supplied_figure_provenance.json.
- Text extraction from both compiled documents and every included figure found no Chinese characters.
- Both documents were compiled twice with XeLaTeX. Final logs contain no overfull boxes, missing-character warnings or undefined-reference warnings.
- All 30 accompanying Python analysis files passed syntax parsing. Full model training and source-data downloads were not rerun for the translation release.
- The published files contain aggregate results; no admission-level records, individual prediction tables, model caches or credentials are included.

The checks can be repeated with `python scripts/compile.py` and `python scripts/validate.py` (the latter requires PyMuPDF). The numerical source comparison was performed against the local Chinese manuscript inventory; that working manuscript is not redistributed here.

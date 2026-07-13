# Repository Language Distribution (In-Scope PRs)

Requested by Review 1 ("More information about the dataset would be welcome, in particular
the programming languages and application domains of the repositories included") and Review 2
(better characterization of the 210,902 in-scope PR set, Section 3.2 / 4.2 of the paper).

**Scope:** the 210,902 in-scope PRs across 57,582 repositories (same population as Table 2 /
Section 4.2, "PR Filtering and Commit Retrieval"). Language is the GitHub-detected primary
language of the repository at PR time, as recorded in the AIDev catalogue; a PR's repository
without a detected language is labelled `Unknown`.

**Known limitation — no application-domain field.** AIDev records only the repository's
primary programming language; it does not include an application-domain / vertical label
(e.g., web app, ML pipeline, game, CLI tool). We therefore cannot report the "application
domains" dimension requested by Review 1 from this dataset; doing so would require a separate
classification pass (e.g., topic modelling on repo descriptions/READMEs), which is out of scope
for the camera-ready. Language is reported below as the closest available proxy.

## Top 20 languages (95.8% of in-scope PRs)

| Rank | Language | PRs | Share |
|---|---|---:|---:|
| 1 | Python | 43,624 | 20.68% |
| 2 | TypeScript | 41,497 | 19.68% |
| 3 | JavaScript | 26,974 | 12.79% |
| 4 | Go | 18,037 | 8.55% |
| 5 | Unknown | 14,621 | 6.93% |
| 6 | HTML | 11,844 | 5.62% |
| 7 | Rust | 7,176 | 3.40% |
| 8 | C# | 6,038 | 2.86% |
| 9 | Java | 5,942 | 2.82% |
| 10 | C++ | 5,889 | 2.79% |
| 11 | PHP | 4,711 | 2.23% |
| 12 | Jupyter Notebook | 2,292 | 1.09% |
| 13 | C | 2,154 | 1.02% |
| 14 | Shell | 1,924 | 0.91% |
| 15 | Dart | 1,886 | 0.89% |
| 16 | Kotlin | 1,858 | 0.88% |
| 17 | CSS | 1,729 | 0.82% |
| 18 | Swift | 1,483 | 0.70% |
| 19 | Vue | 1,380 | 0.65% |
| 20 | Ruby | 960 | 0.46% |
| — | Other (185 languages) | 8,883 | 4.21% |

Full breakdown of all 205 detected languages: [`language_distribution.csv`](language_distribution.csv).

## Provenance

Computed from `characterisation.universe.prs_per_language` in
`analysis/results/results_summary.json`, produced by the paper's analysis pipeline
(`analysis/reproduce_paper_statistics.py` / `run_all_analyses.py`) against the dataset snapshot
that reproduces the paper's published population counts (210,902 in-scope PRs, 57,582
repositories). This is the same snapshot referenced in `results_main3.json`
(`n_conflicting_merges: 14960`, matching Table 3).

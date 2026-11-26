# Depot-Augmented Bike-Share Rebalancing — DRO Pipeline (v5)

> **Conditional Value-at-Risk Optimization for Depot-Augmented Bike-Share Rebalancing under Demand Uncertainty**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Solver: HiGHS via PuLP](https://img.shields.io/badge/Solver-HiGHS%20via%20PuLP-green.svg)](https://github.com/coin-or/pulp)
[![Manuscript ID](https://img.shields.io/badge/AIMS%20Math-Manuscript%20Math--32297-orange.svg)](https://aimspress.jams.pub)

Companion code & raw experimental artifacts for the AIMS Mathematics manuscript:

> A. Alfahaid, A. M. Almars, N. Abd El-Mawla, H. M. Balaha, M. Badawy, M. A. Elhosseini,  
> *"Conditional Value-at-Risk Optimization for Depot-Augmented Bike-Share Rebalancing under Demand Uncertainty"*,  
> AIMS Mathematics, Manuscript ID: Math-32297 (Under Review).

---

## Table of Contents

1. [Overview](#1-overview)
2. [The Model Solved](#2-the-model-solved)
3. [Pipeline Architecture](#3-pipeline-architecture)
4. [Repository Layout](#4-repository-layout)
5. [Installation](#5-installation)
6. [Configuration](#6-configuration)
7. [Usage](#7-usage)
8. [Results & Artifacts](#8-results--artifacts)
9. [Runtime Envelope](#9-runtime-envelope)
10. [Data Sources (Open)](#10-data-sources-open)
11. [Citation](#11-citation)
12. [License](#12-license)
13. [Support & Contact](#13-support--contact)

---

## 1. Overview

This repository contains the complete codebase and deposited artifacts for the **Depot-Augmented Distributionally Robust
Optimization (DRO)** pipeline described in the manuscript above. The framework addresses the bike-share rebalancing
problem under demand uncertainty by optimizing Conditional Value-at-Risk (CVaR) across bootstrapped demand scenarios.

The pipeline ingests public trip data, constructs spatial clusters, generates bootstrap demand scenarios, solves the
first-stage CVaR linear program (alongside Expected Value and Sample Average Approximation baselines), and produces a
comprehensive suite of statistical validations and 42+ publication-ready visualizations.

---

## 2. The Model Solved

The first stage chooses inter-station rebalancing flows and depot sourcing before demand is observed; the recourse
charges an unmet-demand penalty of **12.50 USD per vehicle**.

- **Master Problem**: Depot-augmented **Conditional Value-at-Risk (CVaR) linear program** at risk level **α = 0.75**
  over 30 bootstrapped weekday scenarios per cluster (Rockafellar–Uryasev reformulation).
- **Baselines**: The **Expected Value (EV)** and **Sample Average Approximation (SAA)** models use the identical linear
  program at the α = 0 limit.
- **Out-of-Sample Evaluation**: Decisions are evaluated on **100 weekend scenarios** per cluster drawn from a disjoint
  weekend trip pool, embedding a genuine weekday-to-weekend distributional shift.

> **Note on Wasserstein Radius**: For this additive-recourse class, the Wasserstein radius is *not* a tuning device.
> With a 1-Wasserstein ball of radius ε, unbounded support, and Lipschitz growth, the worst-case expectation equals the
> empirical expectation plus the constant offset *p·ε*. Thus, the distributionally robust program is decision-equivalent
> to SAA, and robustness is carried by the CVaR level α alone. `Config.py` retains a `WassersteinRadius` knob solely for
> duality-offset verification; no radius grid search is performed.

---

## 3. Pipeline Architecture

The pipeline is orchestrated via two main entry points (`RunAll.py` for single runs, `RunMulti.py` for multi-run
statistical aggregation) and executes the following phases sequentially:

| Phase       | Module                           | Purpose                                                                                                                                       |
|-------------|----------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------|
| **Phase 0** | `Phase0Download.py`              | Fetches public monthly trip archives directly from cloud buckets (no credentials required).                                                   |
| **Phase 1** | `Phase1RealData.py`              | Filters, cleans, and spatially clusters stations; generates bootstrap training (weekday) and evaluation (weekend) scenarios.                  |
| **Phase 2** | `Phase2DroModel.py`              | Formulates and solves the CVaR, EV, and SAA linear programs via PuLP/HiGHS; outputs first-stage decisions.                                    |
| **Phase 3** | `Phase3BaselinesOutOfSample.py`  | Evaluates all first-stage decisions against the 100 out-of-sample weekend scenarios per cluster.                                              |
| **Phase 4** | `Phase4StatisticalValidation.py` | Computes Wilcoxon signed-rank tests, bootstrap 95% confidence intervals, and scale-invariance metrics.                                        |
| **Phase 5** | `Phase5Visualization.py`         | Generates **42 publication-ready static figures** (PDF/PNG) + **1 interactive Folium map** detailing costs, distributions, and spatial flows. |
| **Phase 6** | `Phase6MultiRunAggregator.py`    | *(Multi-run only)* Aggregates per-run CSVs, computes mean ± std across independent seeds, and renders a multi-run forest plot.                |

---

## 4. Repository Layout

```text
dro-micro-mobility-rebalancing/
├── README.md                     ← this file
├── LICENSE                       ← MIT License
├── CITATION.cff                  ← citation metadata
├── .gitignore
├── pipeline/                     ← final v5 code (11 modules + requirements.txt)
│   ├── Config.py                 ← global knobs, paths, and experiment configuration
│   ├── RunAll.py                 ← single-run orchestrator (Phases 0–5)
│   ├── RunMulti.py               ← multi-run orchestrator (N independent seeds + Phase 6)
│   ├── Phase0Download.py         ← data acquisition
│   ├── Phase1RealData.py         ← clustering and scenario bootstrapping
│   ├── Phase2DroModel.py         ← LP formulation and solving (CVaR, EV, SAA)
│   ├── Phase3BaselinesOutOfSample.py ← out-of-sample cost evaluation
│   ├── Phase4StatisticalValidation.py  ← hypothesis testing and CI computation
│   ├── Phase5Visualization.py    ← 42+ figures and interactive map generation
│   ├── Phase6MultiRunAggregator.py   ← cross-run aggregation and forest plots
│   └── requirements.txt          ← pinned Python dependencies
└── results/                      ← deposited run artifacts (see results/README.md)
    ├── multirun_12clusters/
    ├── multirun_window0066/
    ├── multirun_window00165/
    └── revision_artifacts/       ← pooled statistical outputs backing manuscript tables
```

---

## 5. Installation

Python ≥ 3.10 (the paper runs used Python 3.12). The HiGHS solver is pulled in through `highspy` and called via `PuLP` —
no separate solver installation is needed.

```bash
# Create and activate a clean virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Upgrade pip and install dependencies
python3 -m pip install --upgrade pip
python3 -m pip install -r pipeline/requirements.txt
```

---

## 6. Configuration

All runtime settings, paths, and experiment parameters are centralized in `pipeline/Config.py`. Key configurable blocks
include:

| Parameter                 | Purpose                                            | Default / Paper Value |
|---------------------------|----------------------------------------------------|-----------------------|
| `ACTIVE_CITY_KEY`         | Target city network (`Divvy`, `JC`, `CitibikeNYC`) | `Divvy`               |
| `ACTIVE_START_YEAR_MONTH` | Start of the data window (YYYYMM)                  | `202601`              |
| `ACTIVE_END_YEAR_MONTH`   | End of the data window (YYYYMM)                    | `202612`              |
| `GLOBAL_RANDOM_SEED`      | Base seed for reproducibility                      | `20260801`            |
| `CVaR_ALPHA`              | Risk level for the DRO model                       | `0.75`                |
| `UnmetDemandPenalty`      | Recourse cost per unmet vehicle (USD)              | `12.50`               |
| `DepotSourcingCost`       | Cost per vehicle sourced from depot (USD)          | `4.00`                |
| `BootstrapResamples`      | Number of bootstrap resamples for CIs              | `2000`                |

> ⚠ **Note**: Ensure the `DATA_DIRECTORY` and `FIGURES_DIRECTORY` in `Config.py` point to writable locations. By
> default, outputs are written to `../data/` relative to the pipeline folder.

---

## 7. Usage

### 7.1 Quick Start (Single Run)

Execute the full end-to-end pipeline for a single configuration:

```bash
cd pipeline
python3 RunAll.py --city Divvy --start-year-month 202601 --end-year-month 202612
```

*Outputs (CSVs, 42 figures, `.txt`/`.tex` interpretation reports) are written to the configured `data/` directory.*

### 7.2 Multi-Run Statistical Aggregation

To assess stability, rerun the pipeline *N* times with different random seeds and aggregate the results (mean ± std):

```bash
python3 RunMulti.py --city Divvy --start-year-month 202601 --end-year-month 202612 \
    --n-runs 5 --base-seed 20260801 --skip-phase0
```

*The `--skip-phase0` flag reuses already-downloaded month archives, saving time on subsequent runs.*

### 7.3 Reproducing the Manuscript Runs

| Setting                               | Value                                                          |
|---------------------------------------|----------------------------------------------------------------|
| **Cities**                            | Divvy (Chicago), JC (Jersey City), CitibikeNYC (New York City) |
| **Runs per configuration**            | 5 independent runs                                             |
| **Seeds**                             | 20260801, 20260802, 20260803, 20260804, 20260805               |
| **Clusters**                          | 18 (Divvy, CitibikeNYC); 10 for JC (7 demand-bearing)          |
| **Training scenarios / cluster**      | 30 (weekday pool)                                              |
| **Out-of-sample scenarios / cluster** | 100 (weekend pool)                                             |
| **Bootstrap window fraction**         | 0.033 (baseline); 0.066 and 0.165 (sensitivity runs)           |
| **Risk level α**                      | 0.75                                                           |
| **Unmet-demand penalty *p***          | 12.50 USD per vehicle                                          |
| **Depot sourcing cost**               | 4.00 USD per vehicle                                           |
| **Cluster bootstrap**                 | 2,000 resamples, 95% confidence intervals                      |

---

## 8. Results & Artifacts

The deposited run artifacts backing the manuscript tables and supplementary analyses reside in
`results/revision_artifacts/`. To maintain a clean and focused repository, only the following finalized aggregation and
validation files are retained:

### 8.1 Core Statistics & Hypothesis Testing

| Artifact File                                                                                                      | Purpose                                                                                                |
|--------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------|
| `Revision_Stats_CitibikeNYC.csv` <br> `Revision_Stats_Divvy.csv` <br> `Revision_Stats_JC.csv`                      | Pooled mean operational costs, standard deviations, and 95% bootstrap confidence intervals per method. |
| `Revision_ClusterTests_CitibikeNYC.csv` <br> `Revision_ClusterTests_Divvy.csv` <br> `Revision_ClusterTests_JC.csv` | Cluster-level Wilcoxon signed-rank tests and Cohen’s *r* effect sizes (DRO vs. baselines).             |
| `Revision_Comparative_CitibikeNYC.csv` <br> `Revision_Comparative_Divvy.csv` <br> `Revision_Comparative_JC.csv`    | Mean cost reduction percentages and statistical significance vs. EV and SAA baselines.                 |

### 8.2 Cost Decomposition & Split Analysis

| Artifact File                                                                                                                     | Purpose                                                                                        |
|-----------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------|
| `Revision_CostDecomposition_CitibikeNYC.csv` <br> `Revision_CostDecomposition_Divvy.csv` <br> `Revision_CostDecomposition_JC.csv` | Breakdown of total cost into first-stage rebalancing and out-of-sample unmet-demand penalties. |
| `Revision_CostSplit_CitibikeNYC.csv` <br> `Revision_CostSplit_Divvy.csv` <br> `Revision_CostSplit_JC.csv`                         | Granular per-scenario cost splits for detailed auditing.                                       |
| `Revision_SplitSummary_CitibikeNYC.csv` <br> `Revision_SplitSummary_Divvy.csv` <br> `Revision_SplitSummary_JC.csv`                | Aggregated summaries of the cost splits across all clusters and methods.                       |

### 8.3 Decision Auditing & Distributional Shift

| Artifact File                                                                                                         | Purpose                                                                                                 |
|-----------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------|
| `Revision_DecisionAudit_CitibikeNYC.csv` <br> `Revision_DecisionAudit_Divvy.csv` <br> `Revision_DecisionAudit_JC.csv` | First-stage net positioning decisions and depot sourcing volumes per cluster.                           |
| `Revision_ShiftMeasure.csv`                                                                                           | Quantification of the weekday-to-weekend distributional shift embedded in the out-of-sample evaluation. |

### 8.4 Variant & Sensitivity Analysis

| Artifact File                       | Purpose                                                                                        |
|-------------------------------------|------------------------------------------------------------------------------------------------|
| `Revision_Variants_Comparison.csv`  | Cross-method comparison across different experimental configurations.                          |
| `Revision_Variants_Sensitivity.csv` | Sensitivity analysis results (e.g., varying the bootstrap window fraction to 0.066 and 0.165). |
| `Revision_Variants_Wasserstein.csv` | Duality-offset verification for the Wasserstein radius (confirms decision-equivalence to SAA). |

### 8.5 Metadata & Registry

| Artifact File             | Purpose                                                                                   |
|---------------------------|-------------------------------------------------------------------------------------------|
| `Revision_Numbers.json`   | Machine-readable registry of all per-run means, per-run tests, and aggregated metrics.    |
| `Revision_DataMonths.csv` | Registry of the exact month archives fetched and merged for each city’s experimental run. |

---

> **⚠️ Note on Reproducibility and Numerical Variance**  
> While the pipeline is designed for full reproducibility given a fixed random seed, **exact numerical results in the
> deposited artifacts may differ slightly** if the pipeline is re-executed. This minor variance can arise from:
> 1. **Solver Tolerances**: Floating-point arithmetic and internal branching in the HiGHS solver may vary slightly
>    across different hardware architectures, operating systems, or solver versions.
> 2. **Stochastic Resampling**: Bootstrap confidence intervals rely on random sampling; different random seeds will
>    yield slightly different interval bounds (though the overarching statistical conclusions remain robust).
> 3. **Data Versioning**: Public trip data archives are occasionally updated or corrected by the source providers (e.g.,
>    Divvy, Citi Bike).
>
> These variances are expected in large-scale stochastic optimization pipelines and do not affect the validity of the
> manuscript’s core claims regarding the superiority and stability of the proposed DRO model.

---

## 9. Runtime Envelope

Every per-cluster LP instance in the deposited runs solves in **0.03–0.27 seconds** (HiGHS via PuLP). A full 18-cluster,
three-method city run completes end-to-end in approximately **two minutes per run** on commodity Linux hardware.

---

## 10. Data Sources (Open)

The pipeline automatically fetches data from public buckets (no credentials required):

- **Chicago Divvy**: [Divvy System Data](https://divvybikes.com/system-data)
  ([S3 Index](https://divvy-tripdata.s3.amazonaws.com/index.html))
- **Jersey City / NYC Citi Bike**: [Citi Bike System Data](https://citibikenyc.com/system-data)
  ([S3 Index](https://s3.amazonaws.com/tripdata/index.html))

---

## 11. Citation

If you use this pipeline, code, or deposited artifacts in your own work, please cite the manuscript:

```bibtex
@article{alfahaid2026cvar,
  title={Conditional Value-at-Risk Optimization for Depot-Augmented Bike-Share Rebalancing under Demand Uncertainty},
  author={Alfahaid, Abdullah and Almars, Abdulqader M. and Abd El-Mawla, Nesma and Balaha, Hossam Magdy and Badawy, Mahmoud and Elhosseini, Mostafa A.},
  journal={AIMS Mathematics},
  note={Manuscript ID: Math-32297 (Under Review)},
  year={2026}
}
```

---

## 12. License

The source code in this repository is released under the **MIT License** — see the [`LICENSE`](LICENSE) file for the
full text. Copyright © 2026 the manuscript authors.

The **public trip datasets** referenced retain their original upstream licenses (typically open data / CC0).

---

## 13. Support & Contact

For questions, bug reports, or contributions, please contact the author:

* **Hossam Magdy Balaha** — <h3ossam@gmail.com>
* Or open an issue on the repository's GitHub issue tracker.

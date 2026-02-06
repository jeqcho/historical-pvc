# Historical PVC

Proportional Veto Core (PVC) analysis of historical elections from [PrefLib](https://www.preflib.org/). This project computes PVC size and critical epsilon for election winners across multiple real-world datasets, comparing five voting rules: Plurality, IRV, Borda, Schulze, and Veto.

## Setup

Requires Python 3.11--3.13. Install dependencies with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

Key dependencies include:
- [`preflibtools`](https://pypi.org/project/preflibtools/) -- parsing PrefLib election data
- [`pvc-toolbox`](https://github.com/jeqcho/pvc-toolbox) -- computing PVC and critical epsilon
- [`votekit`](https://github.com/jeqcho/VoteKit) -- computing winners under various voting rules
- `matplotlib`, `seaborn` -- visualization
- `scipy`, `numpy`, `pandas` -- data processing

## Data

Election data is stored in `data/` in [PrefLib](https://www.preflib.org/) format (`.soc` for strict complete orders, `.soi` for strict incomplete orders, `.toc` for complete orders with ties). The `data/` directory is git-ignored; you will need to download the datasets yourself from PrefLib.

**Datasets:**

| Dataset | Format | Description |
|---------|--------|-------------|
| Habermas | `.soc` | Habermas experiments |
| Polish | `.soi` | Polish local elections |
| Eurovision | `.soi` | Eurovision Song Contest |
| ERS | `.soi` | Electoral Reform Society elections |
| Skate | `.soc`, `.toc` | Figure skating competitions |
| Spotify | `.soc` | Spotify country charts |

## Usage

All scripts live in `src/` and should be run from that directory.

### Core analysis

**`compute_voting_results.py`** -- Compute winners and critical epsilon for all voting rules (Plurality, IRV, Borda, Schulze, Veto) across the Polish, Eurovision, ERS, Skate, and Habermas datasets. Saves results to `data/voting_results.csv`.

```bash
cd src
python compute_voting_results.py
```

**`analyze.py`** -- Main PVC analysis script. Loads election datasets, filters by voter/alternative bounds, computes PVC size and effective epsilon for plurality winners, and generates strip plots, bar plots, and scatter plots.

```bash
python analyze.py
```

**`analyze_by_voting_rule.py`** -- Analyzes results broken down by voting rule. Generates per-rule visualizations (strip plots by dataset-regime group, scatter plots) and aggregate comparisons across all rules.

```bash
python analyze_by_voting_rule.py
```

**`analyze_by_regime.py`** -- Analyzes results by dataset-regime combinations, classifying elections into regimes based on voter/alternative ratios. Only includes combinations with more than 10 samples.

```bash
python analyze_by_regime.py
```

### Visualization (from precomputed results)

These scripts generate plots from precomputed CSV data without recomputing winners or epsilon, making them fast to iterate on.

**`plot_voting_results.py`** -- Generate all standard visualizations from `data/voting_results.csv`.

```bash
python plot_voting_results.py
```

**`plot_dataset_cdf.py`** -- CDF plots comparing the cumulative distribution of critical epsilon across voting rules, with a "random" baseline (mean epsilon across all alternatives).

```bash
python plot_dataset_cdf.py
```

**`plot_nm_scatter.py`** -- Scatter plots of n (voters) vs m (alternatives) for each dataset, showing the distribution of elections and regime boundaries.

```bash
python plot_nm_scatter.py
```

**`plot_regime_scatter.py`** -- Scatter plot grid showing each dataset colored by regime classification.

```bash
python plot_regime_scatter.py
```

## Outputs

- **`plots/`** -- All generated visualizations (PNG and PDF), organized by voting rule and dataset
- **`data/voting_results.csv`** -- Precomputed results (winner, epsilon, PVC size per election per voting rule)

## Key Concepts

**Proportional Veto Core (PVC):** The set of alternatives that cannot be "vetoed" by any proportional coalition of voters. A smaller PVC indicates stronger consensus.

**Critical epsilon:** The smallest epsilon value at which a given alternative enters the epsilon-PVC. A winner with lower epsilon has stronger PVC-based legitimacy.

**Regimes:** Elections are classified by the ratio of voters (n) to alternatives (m):
- **n \<\< m** -- many more alternatives than voters (e.g., 3n \<= m)
- **n \approx m** -- similar number of voters and alternatives (e.g., |m-n|/min(m,n) \< 0.25)
- **n \>\> m** -- many more voters than alternatives (e.g., 3m \<= n)

**Voting rules analyzed:**
- **Plurality** -- each voter votes for their top choice; most votes wins
- **IRV (Instant Runoff Voting)** -- iterative elimination of the candidate with fewest first-place votes
- **Borda** -- points assigned by rank position; highest total wins
- **Schulze** -- pairwise comparison method using strongest paths
- **Veto** -- the PVC-based veto rule

## License

MIT License. See [LICENSE](LICENSE) for details.

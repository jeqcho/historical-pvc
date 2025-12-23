#!/usr/bin/env python3
"""
Generate visualizations from precomputed voting results.

Load results from CSV and generate all plots.
This is fast since it doesn't recompute winners or epsilon.
"""

from pathlib import Path
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Voting rules order (for consistent plotting)
VOTING_RULES = ['veto', 'borda', 'schulze', 'irv', 'plurality']

# Color schemes
RULE_COLORS = {
    'veto': '#FF7F00',       # Orange
    'borda': '#2ECC71',      # Green
    'schulze': '#3498DB',    # Blue
    'irv': '#E74C3C',        # Red
    'plurality': '#9B59B6'   # Purple
}

REGIME_COLORS = {
    'n ≪ m': 'blue',
    'n ≈ m': 'green', 
    'n ≫ m': 'red',
    'other': 'gray'
}

DATASET_COLORS = {
    'Polish': '#1f77b4',
    'Eurovision': '#ff7f0e',
    'ERS': '#2ca02c',
    'Skate': '#d62728',
    'Habermas': '#17becf'
}

MIN_SAMPLES = 10

# Input/output paths
RESULTS_FILE = Path(__file__).parent.parent / 'data' / 'voting_results.csv'
PLOTS_DIR = Path(__file__).parent.parent / 'plots'


def load_results() -> pd.DataFrame:
    """Load precomputed results from CSV."""
    if not RESULTS_FILE.exists():
        raise FileNotFoundError(
            f"Results file not found: {RESULTS_FILE}\n"
            "Run compute_voting_results.py first."
        )
    df = pd.read_csv(RESULTS_FILE)
    logger.info(f"Loaded {len(df)} results from {RESULTS_FILE}")
    return df


def create_strip_plot_v2(df: pd.DataFrame, output_dir: Path, voting_rule: str):
    """Create strip plot (by group v2) for epsilon under a voting rule."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Filter for this voting rule and apply 100x100 cap
    df = df[(df['voting_rule'] == voting_rule) & (df['n'] <= 100) & (df['m'] <= 100)].copy()
    
    if df.empty:
        logger.warning(f"No data for {voting_rule}")
        return
    
    # Count samples per group
    group_counts = df['group'].value_counts()
    valid_groups = group_counts[group_counts > MIN_SAMPLES].index.tolist()
    
    if not valid_groups:
        logger.warning(f"No groups with >{MIN_SAMPLES} samples for {voting_rule}")
        return
    
    # Sort groups
    dataset_order = {'Skate': 0, 'Eurovision': 1, 'Polish': 2, 'ERS': 3}
    regime_order = {'n ≪ m': 0, 'n ≈ m': 1, 'n ≫ m': 2, 'other': 3}
    
    def group_sort_key(g):
        parts = g.split(' (')
        dataset = parts[0]
        regime = parts[1].rstrip(')') if len(parts) > 1 else 'other'
        return (dataset_order.get(dataset, 99), regime_order.get(regime, 99))
    
    valid_groups = sorted(valid_groups, key=group_sort_key)
    
    df_filtered = df[df['group'].isin(valid_groups)].copy()
    df_filtered['group_label'] = df_filtered['group'].apply(lambda x: f"{x} (k={group_counts[x]})")
    group_labels = [f"{g} (k={group_counts[g]})" for g in valid_groups]
    
    # Add jitter
    df_filtered['epsilon_jitter'] = df_filtered['epsilon'] + np.random.normal(0, 0.002, len(df_filtered))
    
    # Create plot
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(12, max(6, len(valid_groups) * 0.6)))
    
    sns.stripplot(data=df_filtered, x='epsilon_jitter', y='group_label',
                  hue='regime', order=group_labels,
                  palette=REGIME_COLORS, jitter=0.35, alpha=0.6, size=6, ax=ax)
    
    # Add dataset separator lines
    current_dataset = None
    for i, g in enumerate(valid_groups):
        dataset = g.split(' (')[0]
        if current_dataset is not None and dataset != current_dataset:
            ax.axhline(y=i - 0.5, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
        current_dataset = dataset
    
    # Add mean markers
    for i, group in enumerate(valid_groups):
        group_data = df_filtered[df_filtered['group'] == group]['epsilon']
        if len(group_data) > 0:
            mean_val = group_data.mean()
            ax.scatter([mean_val], [i], color='black', s=100, marker='|', zorder=5, linewidths=2)
    
    ax.set_xlabel('Critical Epsilon')
    ax.set_ylabel('Dataset (Regime)')
    ax.set_title(f'{voting_rule.upper()}: Critical Epsilon by Group (k>10, n,m≤100)')
    ax.legend(title='Regime', bbox_to_anchor=(1.02, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'epsilon_by_group_strip_v2.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"  Saved {output_dir / 'epsilon_by_group_strip_v2.png'}")


def create_scatter_plot(df: pd.DataFrame, output_dir: Path, voting_rule: str, jitter_strength=0.5):
    """Create scatter plot of n vs m colored by epsilon."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Filter for this voting rule and apply 100x100 cap
    df = df[(df['voting_rule'] == voting_rule) & (df['n'] <= 100) & (df['m'] <= 100)].copy()
    
    if df.empty:
        logger.warning(f"No data for scatter plot ({voting_rule})")
        return
    
    # Add jitter
    n_jitter = df['n'].values + np.random.normal(0, jitter_strength, len(df))
    m_jitter = df['m'].values + np.random.normal(0, jitter_strength, len(df))
    
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(n_jitter, m_jitter, c=df['epsilon'], cmap='viridis',
                        alpha=0.7, s=60, edgecolors='white', linewidth=0.5)
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Critical Epsilon')
    
    ax.set_xlabel('n (voters)')
    ax.set_ylabel('m (alternatives)')
    ax.set_title(f'{voting_rule.upper()}: Critical Epsilon (n,m ≤ 100, k={len(df)})')
    ax.set_xlim(0, 105)
    ax.set_ylim(0, 105)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'scatter_epsilon.png', dpi=150)
    plt.close()
    logger.info(f"  Saved {output_dir / 'scatter_epsilon.png'}")


def create_aggregate_dataset_plots(df: pd.DataFrame, output_dir: Path):
    """Create aggregate plots per dataset."""
    # Apply 100x100 cap
    df = df[(df['n'] <= 100) & (df['m'] <= 100)].copy()
    
    datasets = df['dataset_label'].unique()
    
    for dataset in datasets:
        dataset_dir = output_dir / dataset.lower()
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        df_dataset = df[df['dataset_label'] == dataset].copy()
        
        if df_dataset.empty:
            continue
        
        # Count per rule
        rule_counts = df_dataset['voting_rule'].value_counts()
        rule_labels = [f"{r.upper()} (k={rule_counts.get(r, 0)})" for r in VOTING_RULES if r in rule_counts]
        
        sns.set_style("whitegrid")
        
        # Strip plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        df_dataset['epsilon_jitter'] = df_dataset['epsilon'] + np.random.normal(0, 0.002, len(df_dataset))
        df_dataset['rule_label'] = df_dataset['voting_rule'].apply(
            lambda x: f"{x.upper()} (k={rule_counts.get(x, 0)})"
        )
        
        sns.stripplot(data=df_dataset, x='epsilon_jitter', y='rule_label',
                      hue='voting_rule', order=rule_labels,
                      palette=RULE_COLORS, jitter=0.35, alpha=0.6, size=6, ax=ax,
                      legend=False)
        
        # Add mean markers
        for i, rule in enumerate(VOTING_RULES):
            if rule in rule_counts:
                rule_data = df_dataset[df_dataset['voting_rule'] == rule]['epsilon']
                if len(rule_data) > 0:
                    mean_val = rule_data.mean()
                    ax.scatter([mean_val], [i], color='black', s=100, marker='|', zorder=5, linewidths=2)
        
        ax.set_xlabel('Critical Epsilon')
        ax.set_ylabel('Voting Rule')
        ax.set_title(f'{dataset}: Critical Epsilon by Voting Rule (n,m ≤ 100)')
        
        plt.tight_layout()
        plt.savefig(dataset_dir / 'epsilon_by_rule_strip.png', dpi=150)
        plt.close()
        
        # Bar plot
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(data=df_dataset, x='voting_rule', y='epsilon',
                    order=VOTING_RULES, palette=RULE_COLORS,
                    errorbar=('ci', 95), capsize=0.1, ax=ax)
        
        # Add value labels on bars
        for i, rule in enumerate(VOTING_RULES):
            rule_data = df_dataset[df_dataset['voting_rule'] == rule]['epsilon']
            if len(rule_data) > 0:
                mean_val = rule_data.mean()
                ax.text(i, mean_val, f'{mean_val:.3f}', ha='center', va='bottom', 
                        fontsize=10, fontweight='bold')
        
        ax.set_xlabel('Voting Rule')
        ax.set_ylabel('Mean Critical Epsilon')
        ax.set_title(f'{dataset}: Critical Epsilon by Voting Rule (95% CI)')
        ax.set_xticklabels([r.upper() for r in VOTING_RULES])
        plt.tight_layout()
        plt.savefig(dataset_dir / 'epsilon_by_rule_bar.png', dpi=150)
        plt.close()
        
        logger.info(f"  Saved plots for {dataset}")


def create_aggregate_all_plots(df: pd.DataFrame, output_dir: Path):
    """Create aggregate plots for all datasets combined."""
    all_dir = output_dir / 'all'
    all_dir.mkdir(parents=True, exist_ok=True)
    
    # Apply 100x100 cap
    df = df[(df['n'] <= 100) & (df['m'] <= 100)].copy()
    
    if df.empty:
        logger.warning("No data for aggregate plots")
        return
    
    # Count per rule
    rule_counts = df['voting_rule'].value_counts()
    rule_labels = [f"{r.upper()} (k={rule_counts.get(r, 0)})" for r in VOTING_RULES if r in rule_counts]
    
    sns.set_style("whitegrid")
    
    # Strip plot v1 - colored by dataset
    fig, ax = plt.subplots(figsize=(12, 6))
    
    df['epsilon_jitter'] = df['epsilon'] + np.random.normal(0, 0.002, len(df))
    df['rule_label'] = df['voting_rule'].apply(lambda x: f"{x.upper()} (k={rule_counts.get(x, 0)})")
    
    sns.stripplot(data=df, x='epsilon_jitter', y='rule_label',
                  hue='dataset_label', order=rule_labels,
                  palette=DATASET_COLORS, jitter=0.35, alpha=0.5, size=5, ax=ax)
    
    # Add mean markers
    for i, rule in enumerate(VOTING_RULES):
        if rule in rule_counts:
            rule_data = df[df['voting_rule'] == rule]['epsilon']
            if len(rule_data) > 0:
                mean_val = rule_data.mean()
                ax.scatter([mean_val], [i], color='black', s=100, marker='|', zorder=5, linewidths=2)
    
    ax.set_xlabel('Critical Epsilon')
    ax.set_ylabel('Voting Rule')
    ax.set_title('All Datasets: Critical Epsilon by Voting Rule (n,m ≤ 100)')
    ax.legend(title='Dataset', bbox_to_anchor=(1.02, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(all_dir / 'epsilon_by_rule_strip.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Strip plot v2 - colored by voting rule
    fig, ax = plt.subplots(figsize=(12, 6))
    
    sns.stripplot(data=df, x='epsilon_jitter', y='rule_label',
                  hue='voting_rule', order=rule_labels,
                  palette=RULE_COLORS, jitter=0.35, alpha=0.5, size=5, ax=ax,
                  legend=False)
    
    for i, rule in enumerate(VOTING_RULES):
        if rule in rule_counts:
            rule_data = df[df['voting_rule'] == rule]['epsilon']
            if len(rule_data) > 0:
                mean_val = rule_data.mean()
                ax.scatter([mean_val], [i], color='black', s=100, marker='|', zorder=5, linewidths=2)
    
    ax.set_xlabel('Critical Epsilon')
    ax.set_ylabel('Voting Rule')
    ax.set_title('All Datasets: Critical Epsilon by Voting Rule (n,m ≤ 100)')
    
    plt.tight_layout()
    plt.savefig(all_dir / 'epsilon_by_rule_strip_v2.png', dpi=150)
    plt.close()
    
    # Bar plot
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=df, x='voting_rule', y='epsilon',
                order=VOTING_RULES, palette=RULE_COLORS,
                errorbar=('ci', 95), capsize=0.1, ax=ax)
    
    # Add value labels on bars
    for i, rule in enumerate(VOTING_RULES):
        rule_data = df[df['voting_rule'] == rule]['epsilon']
        if len(rule_data) > 0:
            mean_val = rule_data.mean()
            ax.text(i, mean_val, f'{mean_val:.3f}', ha='center', va='bottom',
                    fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Voting Rule')
    ax.set_ylabel('Mean Critical Epsilon')
    ax.set_title('All Datasets: Critical Epsilon by Voting Rule (95% CI)')
    ax.set_xticklabels([r.upper() for r in VOTING_RULES])
    plt.tight_layout()
    plt.savefig(all_dir / 'epsilon_by_rule_bar.png', dpi=150)
    plt.close()
    
    logger.info(f"  Saved aggregate plots to {all_dir}")


def main():
    logger.info("=" * 60)
    logger.info("Generating Voting Results Visualizations")
    logger.info("=" * 60)
    
    # Load results
    df = load_results()
    
    # Create per-rule plots
    for voting_rule in VOTING_RULES:
        rule_dir = PLOTS_DIR / voting_rule
        rule_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"\nCreating plots for {voting_rule.upper()}...")
        
        create_strip_plot_v2(df, rule_dir, voting_rule)
        create_scatter_plot(df, rule_dir, voting_rule)
    
    # Create aggregate plots
    logger.info("\nCreating aggregate plots...")
    aggregate_dir = PLOTS_DIR / 'aggregate'
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    
    create_aggregate_dataset_plots(df, aggregate_dir)
    create_aggregate_all_plots(df, aggregate_dir)
    
    logger.info("\n" + "=" * 60)
    logger.info("Visualization complete!")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()


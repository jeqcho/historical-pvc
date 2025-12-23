#!/usr/bin/env python3
"""
PVC Analysis by Dataset-Regime combinations.

For each dataset, classify elections into regimes using:
- n < m: 3n <= m
- n > m: 3m <= n  
- n = m: |m-n|/min(m,n) < 0.25

Only include combinations with > 10 samples.
"""

import os
import glob
from pathlib import Path
from typing import List, Dict, Any
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from preflibtools.instances import OrdinalInstance
from pvc_toolbox import compute_pvc, compute_critical_epsilon

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Dataset configurations
DATASETS = {
    'spotify': {
        'path': '../data/spotify',
        'pattern': '*.soc',
        'label': 'Spotify'
    },
    'polish': {
        'path': '../data/polish',
        'pattern': '*.soi',
        'label': 'Polish'
    },
    'eurovision': {
        'path': '../data/eurovision',
        'pattern': '*.soi',
        'label': 'Eurovision'
    },
    'ers': {
        'path': '../data/ers',
        'pattern': '*.soi',
        'label': 'ERS'
    },
    'skate': {
        'path': '../data/skate',
        'patterns': ['*.soc', '*.toc'],
        'label': 'Skate'
    }
}

MIN_SAMPLES = 10


def classify_regime(n, m):
    """
    Classify into regime using 2x multiplier.
    """
    min_val = min(n, m)
    if min_val == 0:
        return None
    
    if 2 * n <= m:
        return 'n ≪ m'
    if 2 * m <= n:
        return 'n ≫ m'
    if abs(m - n) / min_val < 0.25:
        return 'n ≈ m'
    return None


def extend_incomplete_order(order: tuple, all_alternatives: set) -> tuple:
    """Extend incomplete order by appending missing alternatives."""
    ranked_alts = set(alt for rank in order for alt in rank)
    missing = sorted(all_alternatives - ranked_alts)
    
    if not missing:
        return order
    
    extended = list(order)
    for alt in missing:
        extended.append((alt,))
    return tuple(extended)


def instance_to_preferences(instance: OrdinalInstance):
    """Convert preflibtools instance to pvc_toolbox format."""
    all_alternatives = set(instance.alternatives_name.keys())
    full_profile = instance.full_profile()
    
    extended_profile = []
    for order in full_profile:
        extended = extend_incomplete_order(order, all_alternatives)
        extended_profile.append(extended)
    
    num_ranks = len(all_alternatives)
    alternatives = [str(a) for a in sorted(all_alternatives)]
    
    preferences = []
    for rank in range(num_ranks):
        row = []
        for voter_pref in extended_profile:
            alt = voter_pref[rank][0]
            row.append(str(alt))
        preferences.append(row)
    
    return preferences, alternatives


def load_all_elections():
    """Load all elections from all datasets with regime classification."""
    all_elections = []
    
    for name, config in DATASETS.items():
        base_path = Path(__file__).parent / config['path']
        # Handle both 'pattern' (single) and 'patterns' (list)
        patterns = config.get('patterns', [config.get('pattern', '*.soi')])
        files = []
        for pattern in patterns:
            files.extend(sorted(glob.glob(str(base_path / pattern))))
        logger.info(f"Loading {len(files)} files from {name}...")
        
        for filepath in files:
            try:
                instance = OrdinalInstance()
                instance.parse_file(filepath)
                
                n = instance.num_voters
                m = instance.num_alternatives
                
                # Cap at 100x100
                if n > 100 or m > 100:
                    continue
                
                regime = classify_regime(n, m)
                if regime is None:
                    continue
                
                all_elections.append({
                    'file': os.path.basename(filepath),
                    'dataset': name,
                    'dataset_label': config['label'],
                    'instance': instance,
                    'n': n,
                    'm': m,
                    'regime': regime,
                    'group': f"{config['label']} ({regime})"
                })
            except Exception as e:
                logger.warning(f"Error loading {filepath}: {e}")
    
    return all_elections


def compute_metrics(election: Dict[str, Any]) -> Dict[str, Any]:
    """Compute PVC and epsilon for an election."""
    instance = election['instance']
    
    try:
        preferences, alternatives = instance_to_preferences(instance)
        
        pvc = compute_pvc(preferences, alternatives)
        pvc_size = len(pvc)
        pvc_proportion = pvc_size / len(alternatives)
        
        # Find plurality winner
        first_place_counts = {}
        for voter_pref in preferences[0]:
            first_place_counts[voter_pref] = first_place_counts.get(voter_pref, 0) + 1
        winner = max(first_place_counts, key=first_place_counts.get)
        
        epsilon = compute_critical_epsilon(preferences, alternatives, winner)
        
        return {
            'file': election['file'],
            'dataset': election['dataset'],
            'dataset_label': election['dataset_label'],
            'n': election['n'],
            'm': election['m'],
            'regime': election['regime'],
            'group': election['group'],
            'pvc_size': pvc_size,
            'pvc_proportion': pvc_proportion,
            'winner': winner,
            'epsilon': epsilon
        }
    except Exception as e:
        logger.error(f"Error computing metrics for {election['file']}: {e}")
        return None


def create_strip_plots(df: pd.DataFrame, output_dir: Path):
    """Create strip plots for groups with >10 samples."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Count samples per group
    group_counts = df['group'].value_counts()
    valid_groups = group_counts[group_counts > MIN_SAMPLES].index.tolist()
    
    # Sort groups: by regime first, then by dataset
    regime_order = {'n ≪ m': 0, 'n ≈ m': 1, 'n ≫ m': 2}
    valid_groups = sorted(valid_groups, 
                         key=lambda x: (regime_order.get(x.split('(')[1].rstrip(')'), 3), x))
    
    logger.info(f"Groups with >{MIN_SAMPLES} samples: {valid_groups}")
    
    # Filter dataframe
    df_filtered = df[df['group'].isin(valid_groups)].copy()
    
    if df_filtered.empty:
        logger.error("No groups with sufficient samples!")
        return
    
    # Add count to group labels
    df_filtered['group_label'] = df_filtered['group'].apply(
        lambda x: f"{x} (k={group_counts[x]})"
    )
    group_labels = [f"{g} (k={group_counts[g]})" for g in valid_groups]
    
    # Dataset color palette
    dataset_colors = {'Spotify': '#1DB954', 'Polish': '#DC143C', 'Eurovision': '#0066CC', 'ERS': '#FF8C00', 'Skate': '#9B59B6'}
    
    # Find regime boundaries for separator lines
    regime_boundaries = []
    current_regime = None
    for i, group in enumerate(valid_groups):
        regime = group.split('(')[1].rstrip(')')
        if current_regime is not None and regime != current_regime:
            regime_boundaries.append(i - 0.5)
        current_regime = regime
    
    sns.set_style("whitegrid")
    
    # Determine figure height based on number of groups
    fig_height = max(6, len(valid_groups) * 1.2)
    
    # Add x-axis jitter to the data
    np.random.seed(42)
    df_filtered['pvc_proportion_jitter'] = df_filtered['pvc_proportion'] + np.random.normal(0, 0.005, len(df_filtered))
    df_filtered['epsilon_jitter'] = df_filtered['epsilon'] + np.random.normal(0, 0.002, len(df_filtered))
    
    # 1. PVC Proportion Strip Plot
    fig, ax = plt.subplots(figsize=(12, fig_height))
    sns.stripplot(data=df_filtered, x='pvc_proportion_jitter', y='group_label', 
                  hue='dataset_label', order=group_labels, 
                  palette=dataset_colors, jitter=0.35, alpha=0.6, size=6, ax=ax)
    
    # Add regime separator lines
    for boundary in regime_boundaries:
        ax.axhline(y=boundary, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
    
    # Add mean markers
    for i, group in enumerate(valid_groups):
        group_data = df_filtered[df_filtered['group'] == group]['pvc_proportion']
        if len(group_data) > 0:
            mean_val = group_data.mean()
            ax.scatter([mean_val], [i], marker='D', s=80, c='black', 
                      edgecolors='white', linewidth=1.5, zorder=10)
    
    # Add mean to legend
    ax.scatter([], [], marker='D', s=80, c='black', edgecolors='white', linewidth=1.5, label='Mean')
    ax.legend(loc='upper right', title='Dataset')
    ax.set_xlabel('Proportion of Candidates in PVC')
    ax.set_ylabel('Dataset (Regime)')
    ax.set_title('PVC Proportion by Dataset-Regime')
    plt.tight_layout()
    plt.savefig(output_dir / 'pvc_proportion_by_group_strip.png', dpi=150)
    plt.close()
    logger.info("Created pvc_proportion_by_group_strip.png")
    
    # 2. Epsilon Strip Plot
    fig, ax = plt.subplots(figsize=(12, fig_height))
    sns.stripplot(data=df_filtered, x='epsilon_jitter', y='group_label',
                  hue='dataset_label', order=group_labels,
                  palette=dataset_colors, jitter=0.35, alpha=0.6, size=6, ax=ax)
    
    # Add regime separator lines
    for boundary in regime_boundaries:
        ax.axhline(y=boundary, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
    
    # Add mean markers
    for i, group in enumerate(valid_groups):
        group_data = df_filtered[df_filtered['group'] == group]['epsilon']
        if len(group_data) > 0:
            mean_val = group_data.mean()
            ax.scatter([mean_val], [i], marker='D', s=80, c='black', 
                      edgecolors='white', linewidth=1.5, zorder=10)
    
    # Add mean to legend
    ax.scatter([], [], marker='D', s=80, c='black', edgecolors='white', linewidth=1.5, label='Mean')
    ax.legend(loc='upper right', title='Dataset')
    ax.set_xlabel('Effective Epsilon of Winner')
    ax.set_ylabel('Dataset (Regime)')
    ax.set_title('Effective Epsilon by Dataset-Regime')
    plt.tight_layout()
    plt.savefig(output_dir / 'epsilon_by_group_strip.png', dpi=150)
    plt.close()
    logger.info("Created epsilon_by_group_strip.png")
    
    # 3. Bar plots with 95% CI - colored by dataset
    fig, ax = plt.subplots(figsize=(12, fig_height))
    sns.barplot(data=df_filtered, x='pvc_proportion', y='group_label',
                hue='dataset_label', order=group_labels, palette=dataset_colors,
                errorbar=('ci', 95), capsize=0.1, ax=ax, dodge=False)
    
    # Add regime separator lines
    for boundary in regime_boundaries:
        ax.axhline(y=boundary, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
    
    ax.legend(loc='upper right', title='Dataset')
    ax.set_xlabel('Mean Proportion of Candidates in PVC')
    ax.set_ylabel('Dataset (Regime)')
    ax.set_title('PVC Proportion by Dataset-Regime (with 95% CI)')
    plt.tight_layout()
    plt.savefig(output_dir / 'pvc_proportion_by_group_bar.png', dpi=150)
    plt.close()
    logger.info("Created pvc_proportion_by_group_bar.png")
    
    fig, ax = plt.subplots(figsize=(12, fig_height))
    sns.barplot(data=df_filtered, x='epsilon', y='group_label',
                hue='dataset_label', order=group_labels, palette=dataset_colors,
                errorbar=('ci', 95), capsize=0.1, ax=ax, dodge=False)
    
    # Add regime separator lines
    for boundary in regime_boundaries:
        ax.axhline(y=boundary, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
    
    ax.legend(loc='upper right', title='Dataset')
    ax.set_xlabel('Mean Effective Epsilon of Winner')
    ax.set_ylabel('Dataset (Regime)')
    ax.set_title('Effective Epsilon by Dataset-Regime (with 95% CI)')
    plt.tight_layout()
    plt.savefig(output_dir / 'epsilon_by_group_bar.png', dpi=150)
    plt.close()
    logger.info("Created epsilon_by_group_bar.png")


def create_strip_plots_by_regime(df: pd.DataFrame, output_dir: Path):
    """Create strip plots colored by regime with lines between datasets."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Count samples per group
    group_counts = df['group'].value_counts()
    valid_groups = group_counts[group_counts > MIN_SAMPLES].index.tolist()
    
    # Sort groups: by dataset first, then by regime
    dataset_order = {'Skate': 0, 'Spotify': 1, 'Eurovision': 2, 'Polish': 3, 'ERS': 4}
    regime_order = {'n ≪ m': 0, 'n ≈ m': 1, 'n ≫ m': 2}
    valid_groups = sorted(valid_groups, 
                         key=lambda x: (dataset_order.get(x.split(' (')[0], 9), 
                                       regime_order.get(x.split('(')[1].rstrip(')'), 3)))
    
    # Filter dataframe
    df_filtered = df[df['group'].isin(valid_groups)].copy()
    
    if df_filtered.empty:
        return
    
    # Add count to group labels
    df_filtered['group_label'] = df_filtered['group'].apply(
        lambda x: f"{x} (k={group_counts[x]})"
    )
    group_labels = [f"{g} (k={group_counts[g]})" for g in valid_groups]
    
    # Regime color palette (same as nm_scatter_regimes)
    regime_colors = {'n ≪ m': 'blue', 'n ≈ m': 'green', 'n ≫ m': 'red'}
    
    # Find dataset boundaries for separator lines
    dataset_boundaries = []
    current_dataset = None
    for i, group in enumerate(valid_groups):
        dataset = group.split(' (')[0]
        if current_dataset is not None and dataset != current_dataset:
            dataset_boundaries.append(i - 0.5)
        current_dataset = dataset
    
    sns.set_style("whitegrid")
    fig_height = max(6, len(valid_groups) * 1.2)
    
    # Add x-axis jitter to the data
    np.random.seed(42)
    df_filtered['pvc_proportion_jitter'] = df_filtered['pvc_proportion'] + np.random.normal(0, 0.005, len(df_filtered))
    df_filtered['epsilon_jitter'] = df_filtered['epsilon'] + np.random.normal(0, 0.002, len(df_filtered))
    
    # 1. PVC Proportion Strip Plot - colored by regime
    fig, ax = plt.subplots(figsize=(12, fig_height))
    sns.stripplot(data=df_filtered, x='pvc_proportion_jitter', y='group_label', 
                  hue='regime', order=group_labels, 
                  palette=regime_colors, jitter=0.35, alpha=0.6, size=6, ax=ax)
    
    # Add dataset separator lines
    for boundary in dataset_boundaries:
        ax.axhline(y=boundary, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
    
    # Add mean markers
    for i, group in enumerate(valid_groups):
        group_data = df_filtered[df_filtered['group'] == group]['pvc_proportion']
        if len(group_data) > 0:
            mean_val = group_data.mean()
            ax.scatter([mean_val], [i], marker='D', s=80, c='black', 
                      edgecolors='white', linewidth=1.5, zorder=10)
    
    ax.scatter([], [], marker='D', s=80, c='black', edgecolors='white', linewidth=1.5, label='Mean')
    ax.legend(loc='upper right', title='Regime')
    ax.set_xlabel('Proportion of Candidates in PVC')
    ax.set_ylabel('Dataset (Regime)')
    ax.set_title('PVC Proportion by Dataset-Regime (colored by regime)')
    plt.tight_layout()
    plt.savefig(output_dir / 'pvc_proportion_by_group_strip_v2.png', dpi=150)
    plt.close()
    logger.info("Created pvc_proportion_by_group_strip_v2.png")
    
    # 2. Epsilon Strip Plot - colored by regime
    fig, ax = plt.subplots(figsize=(12, fig_height))
    sns.stripplot(data=df_filtered, x='epsilon_jitter', y='group_label',
                  hue='regime', order=group_labels,
                  palette=regime_colors, jitter=0.35, alpha=0.6, size=6, ax=ax)
    
    # Add dataset separator lines
    for boundary in dataset_boundaries:
        ax.axhline(y=boundary, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
    
    # Add mean markers
    for i, group in enumerate(valid_groups):
        group_data = df_filtered[df_filtered['group'] == group]['epsilon']
        if len(group_data) > 0:
            mean_val = group_data.mean()
            ax.scatter([mean_val], [i], marker='D', s=80, c='black', 
                      edgecolors='white', linewidth=1.5, zorder=10)
    
    ax.scatter([], [], marker='D', s=80, c='black', edgecolors='white', linewidth=1.5, label='Mean')
    ax.legend(loc='upper right', title='Regime')
    ax.set_xlabel('Effective Epsilon of Winner')
    ax.set_ylabel('Dataset (Regime)')
    ax.set_title('Effective Epsilon by Dataset-Regime (colored by regime)')
    plt.tight_layout()
    plt.savefig(output_dir / 'epsilon_by_group_strip_v2.png', dpi=150)
    plt.close()
    logger.info("Created epsilon_by_group_strip_v2.png")
    
    # 3. Bar plots with 95% CI - colored by regime
    fig, ax = plt.subplots(figsize=(12, fig_height))
    sns.barplot(data=df_filtered, x='pvc_proportion', y='group_label',
                hue='regime', order=group_labels, palette=regime_colors,
                errorbar=('ci', 95), capsize=0.1, ax=ax, dodge=False)
    
    for boundary in dataset_boundaries:
        ax.axhline(y=boundary, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
    
    ax.legend(loc='upper right', title='Regime')
    ax.set_xlabel('Mean Proportion of Candidates in PVC')
    ax.set_ylabel('Dataset (Regime)')
    ax.set_title('PVC Proportion by Dataset-Regime (with 95% CI, colored by regime)')
    plt.tight_layout()
    plt.savefig(output_dir / 'pvc_proportion_by_group_bar_v2.png', dpi=150)
    plt.close()
    logger.info("Created pvc_proportion_by_group_bar_v2.png")
    
    fig, ax = plt.subplots(figsize=(12, fig_height))
    sns.barplot(data=df_filtered, x='epsilon', y='group_label',
                hue='regime', order=group_labels, palette=regime_colors,
                errorbar=('ci', 95), capsize=0.1, ax=ax, dodge=False)
    
    for boundary in dataset_boundaries:
        ax.axhline(y=boundary, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
    
    ax.legend(loc='upper right', title='Regime')
    ax.set_xlabel('Mean Effective Epsilon of Winner')
    ax.set_ylabel('Dataset (Regime)')
    ax.set_title('Effective Epsilon by Dataset-Regime (with 95% CI, colored by regime)')
    plt.tight_layout()
    plt.savefig(output_dir / 'epsilon_by_group_bar_v2.png', dpi=150)
    plt.close()
    logger.info("Created epsilon_by_group_bar_v2.png")


def create_scatter_plots_per_dataset(df: pd.DataFrame, output_dir: Path, jitter_strength=0.5):
    """Create scatter plots (n vs m colored by pvc/epsilon) for each dataset."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    np.random.seed(42)  # For reproducible jitter
    
    # First create combined scatter plots for all datasets
    n_jitter_all = df['n'].values + np.random.normal(0, jitter_strength, len(df))
    m_jitter_all = df['m'].values + np.random.normal(0, jitter_strength, len(df))
    
    # Combined scatter plot colored by PVC proportion
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(n_jitter_all, m_jitter_all, 
                        c=df['pvc_proportion'], 
                        cmap='viridis', alpha=0.7, s=60, 
                        edgecolors='white', linewidth=0.5)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('PVC Proportion')
    ax.set_xlabel('Number of Voters (n)')
    ax.set_ylabel('Number of Alternatives (m)')
    ax.set_title('All Datasets: PVC Proportion')
    plt.tight_layout()
    plt.savefig(output_dir / 'scatter_pvc.png', dpi=150)
    plt.close()
    logger.info("Created scatter_pvc.png")
    
    # Combined scatter plot colored by epsilon
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(n_jitter_all, m_jitter_all, 
                        c=df['epsilon'], 
                        cmap='plasma', alpha=0.7, s=60, 
                        edgecolors='white', linewidth=0.5)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Effective Epsilon')
    ax.set_xlabel('Number of Voters (n)')
    ax.set_ylabel('Number of Alternatives (m)')
    ax.set_title('All Datasets: Effective Epsilon')
    plt.tight_layout()
    plt.savefig(output_dir / 'scatter_epsilon.png', dpi=150)
    plt.close()
    logger.info("Created scatter_epsilon.png")
    
    # Now create per-dataset scatter plots
    datasets = df['dataset_label'].unique()
    
    for dataset in datasets:
        df_dataset = df[df['dataset_label'] == dataset].copy()
        
        if len(df_dataset) < 5:
            continue
        
        # Add jitter
        n_jitter = df_dataset['n'].values + np.random.normal(0, jitter_strength, len(df_dataset))
        m_jitter = df_dataset['m'].values + np.random.normal(0, jitter_strength, len(df_dataset))
        
        # Scatter plot colored by PVC proportion
        fig, ax = plt.subplots(figsize=(8, 6))
        scatter = ax.scatter(n_jitter, m_jitter, 
                            c=df_dataset['pvc_proportion'], 
                            cmap='viridis', alpha=0.7, s=60, 
                            edgecolors='white', linewidth=0.5)
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('PVC Proportion')
        ax.set_xlabel('Number of Voters (n)')
        ax.set_ylabel('Number of Alternatives (m)')
        ax.set_title(f'{dataset}: PVC Proportion')
        plt.tight_layout()
        plt.savefig(output_dir / f'scatter_pvc_{dataset.lower()}.png', dpi=150)
        plt.close()
        logger.info(f"Created scatter_pvc_{dataset.lower()}.png")
        
        # Scatter plot colored by epsilon
        fig, ax = plt.subplots(figsize=(8, 6))
        scatter = ax.scatter(n_jitter, m_jitter, 
                            c=df_dataset['epsilon'], 
                            cmap='plasma', alpha=0.7, s=60, 
                            edgecolors='white', linewidth=0.5)
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Effective Epsilon')
        ax.set_xlabel('Number of Voters (n)')
        ax.set_ylabel('Number of Alternatives (m)')
        ax.set_title(f'{dataset}: Effective Epsilon')
        plt.tight_layout()
        plt.savefig(output_dir / f'scatter_epsilon_{dataset.lower()}.png', dpi=150)
        plt.close()
        logger.info(f"Created scatter_epsilon_{dataset.lower()}.png")


def main():
    logger.info("=" * 60)
    logger.info("PVC Analysis by Dataset-Regime")
    logger.info("=" * 60)
    
    # Load all elections
    elections = load_all_elections()
    logger.info(f"Loaded {len(elections)} elections total")
    
    # Count by group
    group_counts = {}
    for e in elections:
        group = e['group']
        group_counts[group] = group_counts.get(group, 0) + 1
    
    logger.info("\nGroup counts:")
    for group, count in sorted(group_counts.items()):
        marker = " *" if count > MIN_SAMPLES else ""
        logger.info(f"  {group}: {count}{marker}")
    
    # Filter to groups with enough samples
    valid_groups = {g for g, c in group_counts.items() if c > MIN_SAMPLES}
    elections = [e for e in elections if e['group'] in valid_groups]
    logger.info(f"\nFiltered to {len(elections)} elections in valid groups")
    
    # Compute metrics
    results = []
    for i, election in enumerate(elections):
        if (i + 1) % 50 == 0:
            logger.info(f"Processing {i+1}/{len(elections)}...")
        result = compute_metrics(election)
        if result:
            results.append(result)
    
    logger.info(f"Computed metrics for {len(results)} elections")
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Save results
    output_dir = Path(__file__).parent.parent / 'plots'
    df.to_csv(output_dir / 'results_by_group.csv', index=False)
    logger.info(f"Results saved to {output_dir / 'results_by_group.csv'}")
    
    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    
    for group in sorted(df['group'].unique()):
        gdf = df[df['group'] == group]
        logger.info(f"\n{group}:")
        logger.info(f"  Elections: {len(gdf)}")
        logger.info(f"  PVC Proportion: {gdf['pvc_proportion'].mean():.2%} ± {gdf['pvc_proportion'].std():.2%}")
        logger.info(f"  Epsilon: {gdf['epsilon'].mean():.4f} ± {gdf['epsilon'].std():.4f}")
    
    # Create visualizations
    logger.info("\nCreating visualizations...")
    create_strip_plots(df, output_dir)
    create_strip_plots_by_regime(df, output_dir)
    create_scatter_plots_per_dataset(df, output_dir)
    
    logger.info("\n" + "=" * 60)
    logger.info("Analysis complete!")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()


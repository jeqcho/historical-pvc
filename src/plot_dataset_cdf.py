#!/usr/bin/env python3
"""
Create CDF plots for Habermas and Polish datasets showing cumulative distribution
of critical epsilon values across voting rules, with a "random" baseline.

The random baseline is computed by taking the mean epsilon across ALL alternatives
for each election (representing expected epsilon if winner chosen uniformly at random).
"""

import os
import glob
from pathlib import Path
from typing import List, Dict, Any, Tuple
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from preflibtools.instances import OrdinalInstance
from pvc_toolbox import compute_critical_epsilon

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Dataset configurations for Habermas, Polish, ERS, and Eurovision
DATASETS = {
    'habermas': {
        'path': '../data/habermas',
        'patterns': ['*.soc'],
        'label': 'Habermas'
    },
    'polish': {
        'path': '../data/polish',
        'patterns': ['*.soi'],
        'label': 'Polish'
    },
    'ers': {
        'path': '../data/ers',
        'patterns': ['*.soi'],
        'label': 'ERS'
    },
    'eurovision': {
        'path': '../data/eurovision',
        'patterns': ['*.soi'],
        'label': 'Eurovision'
    }
}

# Voting rules with their colors
VOTING_RULES = ['veto', 'borda', 'schulze', 'irv', 'plurality']
RULE_COLORS = {
    'veto': '#FF7F00',
    'borda': '#2ECC71',
    'schulze': '#3498DB',
    'irv': '#E74C3C',
    'plurality': '#9B59B6'
}

# Paths
VOTING_RESULTS_FILE = Path(__file__).parent.parent / 'data' / 'voting_results.csv'
PLOTS_DIR = Path(__file__).parent.parent / 'plots' / 'aggregate'


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


def instance_to_preferences(instance: OrdinalInstance) -> Tuple[List[List[str]], List[str]]:
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


def compute_random_baseline_for_election(filepath: str) -> Dict[str, Any]:
    """
    Compute random baseline epsilon for a single election.
    
    Returns mean epsilon across all alternatives.
    """
    try:
        instance = OrdinalInstance()
        instance.parse_file(filepath)
        
        preferences, alternatives = instance_to_preferences(instance)
        
        epsilons = []
        for alt in alternatives:
            try:
                eps = compute_critical_epsilon(preferences, alternatives, alt)
                epsilons.append(eps)
            except Exception:
                pass
        
        if not epsilons:
            return None
        
        return {
            'file': os.path.basename(filepath),
            'n_alternatives': len(alternatives),
            'mean_epsilon': np.mean(epsilons),
            'min_epsilon': np.min(epsilons),
            'max_epsilon': np.max(epsilons),
            'all_epsilons': epsilons
        }
    except Exception as e:
        logger.debug(f"Error processing {filepath}: {e}")
        return None


def compute_random_baselines(dataset_name: str, config: Dict) -> List[Dict[str, Any]]:
    """Compute random baseline for all elections in a dataset."""
    base_path = Path(__file__).parent / config['path']
    patterns = config.get('patterns', ['*.soi'])
    
    files = []
    for pattern in patterns:
        files.extend(sorted(glob.glob(str(base_path / pattern))))
    
    logger.info(f"Computing random baseline for {len(files)} elections in {dataset_name}...")
    
    results = []
    for i, filepath in enumerate(files):
        if (i + 1) % 50 == 0:
            logger.info(f"  Processing {i+1}/{len(files)}...")
        result = compute_random_baseline_for_election(filepath)
        if result:
            result['dataset'] = dataset_name
            results.append(result)
    
    logger.info(f"  Computed random baseline for {len(results)} elections")
    return results


def create_cdf_plot(df: pd.DataFrame, random_results: List[Dict[str, Any]], dataset_label: str,
                    output_path: Path, xlim: Tuple[float, float] = (0, 1),
                    ylim: Tuple[float, float] = (0, 1)):
    """Create a CDF plot for epsilon values across voting rules."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for rule in VOTING_RULES:
        rule_data = df[df['voting_rule'] == rule]['epsilon'].sort_values()
        if len(rule_data) == 0:
            continue
        
        n = len(rule_data)
        cdf_y = np.arange(1, n + 1) / n
        
        ax.plot(rule_data.values, cdf_y, 
                color=RULE_COLORS[rule], 
                linewidth=2.5,
                label=rule.upper())
    
    # Plot random baseline as CDF curve (per-election mean epsilons)
    random_epsilons = np.array([r['mean_epsilon'] for r in random_results])
    random_epsilons_sorted = np.sort(random_epsilons)
    n_random = len(random_epsilons_sorted)
    cdf_y_random = np.arange(1, n_random + 1) / n_random
    
    ax.plot(random_epsilons_sorted, cdf_y_random, 
            color='black', linestyle='-', linewidth=2.5,
            label='Random', zorder=5)
    
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel('Critical Epsilon', fontsize=14)
    ax.set_ylabel('Cumulative Probability', fontsize=14)
    ax.set_title(f'{dataset_label}: CDF of Critical Epsilon by Voting Rule', fontsize=16)
    ax.legend(loc='lower right', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', which='major', labelsize=12)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    logger.info(f"Created {output_path}")


def create_bar_plot_with_random(df: pd.DataFrame, random_baseline: float, 
                                 dataset_label: str, output_path: Path):
    """Create bar plot with 95% CI and random baseline reference line."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sns.barplot(data=df, x='voting_rule', y='epsilon',
                order=VOTING_RULES, hue='voting_rule', hue_order=VOTING_RULES,
                palette=RULE_COLORS, legend=False,
                errorbar=('ci', 95), capsize=0.1, ax=ax)
    
    ax.axhline(y=random_baseline, color='red', linestyle='--', linewidth=2.5,
               label='Random', zorder=5)
    
    ax.set_xlabel('Voting Rule', fontsize=14)
    ax.set_ylabel('Mean Critical Epsilon', fontsize=14)
    ax.set_title(f'{dataset_label}: Critical Epsilon by Voting Rule (95% CI)', fontsize=16)
    ax.set_xticks(range(len(VOTING_RULES)))
    ax.set_xticklabels([r.upper() for r in VOTING_RULES], fontsize=12)
    ax.tick_params(axis='y', which='major', labelsize=12)
    ax.legend(loc='upper right', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    logger.info(f"Created {output_path}")


def create_bar_plot_with_random_annotation(df: pd.DataFrame, random_baseline: float, 
                                            dataset_label: str, output_path: Path):
    """Create bar plot with std error and random baseline shown as annotation in legend."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sns.barplot(data=df, x='voting_rule', y='epsilon',
                order=VOTING_RULES, hue='voting_rule', hue_order=VOTING_RULES,
                palette=RULE_COLORS, legend=False,
                errorbar='se', capsize=0.1, ax=ax)
    
    ax.set_xlabel('Voting Rule', fontsize=14)
    ax.set_ylabel('Mean Critical Epsilon', fontsize=14)
    ax.set_title(f'{dataset_label}: Critical Epsilon by Voting Rule (Std Error)', fontsize=16)
    ax.set_xticks(range(len(VOTING_RULES)))
    ax.set_xticklabels([r.upper() for r in VOTING_RULES], fontsize=12)
    ax.tick_params(axis='y', which='major', labelsize=12)
    
    # Add random baseline as text annotation in legend area
    ax.annotate(f'Random baseline: {random_baseline:.4f}', 
                xy=(0.02, 0.95), xycoords='axes fraction',
                ha='left', va='top', fontsize=12,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.9))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    logger.info(f"Created {output_path}")


def main():
    logger.info("=" * 60)
    logger.info("Creating CDF and Bar Plots for Habermas and Polish")
    logger.info("=" * 60)
    
    logger.info(f"Loading voting results from {VOTING_RESULTS_FILE}...")
    voting_results = pd.read_csv(VOTING_RESULTS_FILE)
    logger.info(f"Loaded {len(voting_results)} records")
    
    for dataset_name, config in DATASETS.items():
        logger.info(f"\n{'=' * 40}")
        logger.info(f"Processing {config['label']}")
        logger.info(f"{'=' * 40}")
        
        output_dir = PLOTS_DIR / dataset_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        df = voting_results[voting_results['dataset'] == dataset_name].copy()
        
        if df.empty:
            logger.warning(f"No voting results found for {dataset_name}")
            continue
        
        logger.info(f"Found {len(df)} voting results for {dataset_name}")
        
        random_results = compute_random_baselines(dataset_name, config)
        
        if not random_results:
            logger.warning(f"Could not compute random baseline for {dataset_name}")
            continue
        
        random_baseline = np.mean([r['mean_epsilon'] for r in random_results])
        logger.info(f"Random baseline for {config['label']}: {random_baseline:.4f}")
        
        create_cdf_plot(
            df, random_results, config['label'],
            output_dir / 'epsilon_cdf_full.png',
            xlim=(0, 1), ylim=(0, 1)
        )
        
        create_cdf_plot(
            df, random_results, config['label'],
            output_dir / 'epsilon_cdf_zoomed.png',
            xlim=(0, 0.5), ylim=(0.5, 1)
        )
        
        create_bar_plot_with_random(
            df, random_baseline, config['label'],
            output_dir / 'epsilon_by_rule_bar.png'
        )
        
        create_bar_plot_with_random_annotation(
            df, random_baseline, config['label'],
            output_dir / 'epsilon_by_rule_bar_v2.png'
        )
        
        logger.info(f"\nSummary for {config['label']}:")
        logger.info(f"  Random baseline: {random_baseline:.4f}")
        for rule in VOTING_RULES:
            rule_data = df[df['voting_rule'] == rule]['epsilon']
            if len(rule_data) > 0:
                logger.info(f"  {rule.upper()}: mean={rule_data.mean():.4f}, "
                          f"median={rule_data.median():.4f}, n={len(rule_data)}")
    
    logger.info("\n" + "=" * 60)
    logger.info("Done!")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()

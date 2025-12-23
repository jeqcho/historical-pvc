#!/usr/bin/env python3
"""
PVC Analysis of Historical Elections from PrefLib

This script:
1. Loads election datasets from PrefLib (Spotify, Polish Elections, Eurovision)
2. Filters elections by voter/alternative bounds
3. Computes PVC size and effective epsilon
4. Generates visualizations (strip plots, bar plots, scatter plots)
"""

import os
import glob
from pathlib import Path
from typing import List, Tuple, Dict, Any
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

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
        'path': 'data/spotify',
        'pattern': '*.soc',  # Use complete orders
        'n_min': 20, 'n_max': 40,
        'm_min': 80, 'm_max': 100,
        'regime': 'n < m'
    },
    'polish': {
        'path': 'data/polish',
        'pattern': '*.soi',  # Only incomplete orders available
        'n_min': 10, 'n_max': 20,
        'm_min': 10, 'm_max': 20,
        'regime': 'n = m'
    },
    'eurovision': {
        'path': 'data/eurovision',
        'pattern': '*.soi',  # Only incomplete orders available
        'n_min': 80, 'n_max': 100,
        'm_min': 20, 'm_max': 40,
        'regime': 'n > m'
    }
}


def extend_incomplete_order(order: tuple, all_alternatives: set) -> tuple:
    """
    Extend an incomplete order by appending missing alternatives at the end.
    
    Args:
        order: Tuple of tuples representing the ranking
        all_alternatives: Set of all alternatives
    
    Returns:
        Complete order with all alternatives
    """
    ranked_alts = set(alt for rank in order for alt in rank)
    missing = sorted(all_alternatives - ranked_alts)
    
    if not missing:
        return order
    
    # Append missing alternatives as individual ranks at the end
    extended = list(order)
    for alt in missing:
        extended.append((alt,))
    
    return tuple(extended)


def load_and_filter_dataset(name: str, config: dict) -> List[Dict[str, Any]]:
    """
    Load a dataset and filter by bounds.
    
    Args:
        name: Dataset name
        config: Dataset configuration
    
    Returns:
        List of election data dictionaries
    """
    base_path = Path(__file__).parent / config['path']
    files = sorted(glob.glob(str(base_path / config['pattern'])))
    
    logger.info(f"Found {len(files)} files for {name}")
    
    results = []
    for filepath in files:
        try:
            instance = OrdinalInstance()
            instance.parse_file(filepath)
            
            n = instance.num_voters
            m = instance.num_alternatives
            
            # Filter by bounds
            if not (config['n_min'] <= n <= config['n_max'] and 
                    config['m_min'] <= m <= config['m_max']):
                continue
            
            results.append({
                'file': os.path.basename(filepath),
                'instance': instance,
                'n': n,
                'm': m,
                'regime': config['regime']
            })
            
        except Exception as e:
            logger.warning(f"Error loading {filepath}: {e}")
    
    logger.info(f"Filtered to {len(results)} elections for {name}")
    return results


def instance_to_preferences(instance: OrdinalInstance) -> Tuple[List[List[str]], List[str]]:
    """
    Convert a preflibtools instance to pvc_toolbox format.
    
    Args:
        instance: OrdinalInstance from preflibtools
    
    Returns:
        (preferences, alternatives) tuple
    """
    all_alternatives = set(instance.alternatives_name.keys())
    full_profile = instance.full_profile()
    
    # Extend incomplete orders if necessary
    extended_profile = []
    for order in full_profile:
        extended = extend_incomplete_order(order, all_alternatives)
        extended_profile.append(extended)
    
    # Get number of ranks (should equal number of alternatives now)
    num_ranks = len(all_alternatives)
    alternatives = [str(a) for a in sorted(all_alternatives)]
    
    # Build preferences matrix: preferences[rank][voter]
    preferences = []
    for rank in range(num_ranks):
        row = []
        for voter_pref in extended_profile:
            # Each voter_pref is like ((6,), (9,), ...)
            # voter_pref[rank] is a tuple, take first element
            alt = voter_pref[rank][0]
            row.append(str(alt))
        preferences.append(row)
    
    return preferences, alternatives


def compute_election_metrics(election: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute PVC size and effective epsilon for an election.
    
    Args:
        election: Election data dictionary
    
    Returns:
        Dictionary with computed metrics
    """
    instance = election['instance']
    
    try:
        preferences, alternatives = instance_to_preferences(instance)
        
        # Compute PVC
        pvc = compute_pvc(preferences, alternatives)
        pvc_size = len(pvc)
        pvc_proportion = pvc_size / len(alternatives)
        
        # Find the winner (plurality winner - most first-place votes)
        first_place_counts = {}
        for voter_pref in preferences[0]:
            first_place_counts[voter_pref] = first_place_counts.get(voter_pref, 0) + 1
        winner = max(first_place_counts, key=first_place_counts.get)
        
        # Compute effective epsilon of the winner
        epsilon = compute_critical_epsilon(preferences, alternatives, winner)
        
        return {
            'file': election['file'],
            'n': election['n'],
            'm': election['m'],
            'regime': election['regime'],
            'pvc_size': pvc_size,
            'pvc_proportion': pvc_proportion,
            'winner': winner,
            'epsilon': epsilon
        }
        
    except Exception as e:
        logger.error(f"Error computing metrics for {election['file']}: {e}")
        return None


def create_visualizations(df: pd.DataFrame, output_dir: Path):
    """
    Create all visualizations.
    
    Args:
        df: DataFrame with election results
        output_dir: Directory to save plots
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['font.size'] = 12
    
    regime_order = ['n < m', 'n = m', 'n > m']
    
    # 1. Strip plot for PVC proportion
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.stripplot(data=df, x='pvc_proportion', y='regime', order=regime_order,
                  jitter=True, alpha=0.6, size=8, ax=ax)
    ax.set_xlabel('Proportion of Candidates in PVC')
    ax.set_ylabel('Regime')
    ax.set_title('PVC Proportion by Regime')
    plt.tight_layout()
    plt.savefig(output_dir / 'pvc_proportion_strip.png', dpi=150)
    plt.close()
    logger.info("Created pvc_proportion_strip.png")
    
    # 2. Strip plot for epsilon
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.stripplot(data=df, x='epsilon', y='regime', order=regime_order,
                  jitter=True, alpha=0.6, size=8, ax=ax)
    ax.set_xlabel('Effective Epsilon of Winner')
    ax.set_ylabel('Regime')
    ax.set_title('Effective Epsilon by Regime')
    plt.tight_layout()
    plt.savefig(output_dir / 'epsilon_strip.png', dpi=150)
    plt.close()
    logger.info("Created epsilon_strip.png")
    
    # 3. Bar plot with 95% CI for PVC proportion
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=df, x='regime', y='pvc_proportion', order=regime_order,
                errorbar=('ci', 95), capsize=0.1, ax=ax)
    ax.set_xlabel('Regime')
    ax.set_ylabel('Mean Proportion of Candidates in PVC')
    ax.set_title('PVC Proportion by Regime (with 95% CI)')
    plt.tight_layout()
    plt.savefig(output_dir / 'pvc_proportion_bar.png', dpi=150)
    plt.close()
    logger.info("Created pvc_proportion_bar.png")
    
    # 4. Bar plot with 95% CI for epsilon
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=df, x='regime', y='epsilon', order=regime_order,
                errorbar=('ci', 95), capsize=0.1, ax=ax)
    ax.set_xlabel('Regime')
    ax.set_ylabel('Mean Effective Epsilon of Winner')
    ax.set_title('Effective Epsilon by Regime (with 95% CI)')
    plt.tight_layout()
    plt.savefig(output_dir / 'epsilon_bar.png', dpi=150)
    plt.close()
    logger.info("Created epsilon_bar.png")
    
    # 5. Scatter plot: voters vs alternatives, colored by PVC proportion
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(df['n'], df['m'], c=df['pvc_proportion'], 
                         cmap='viridis', alpha=0.7, s=60, edgecolors='white', linewidth=0.5)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('PVC Proportion')
    ax.set_xlabel('Number of Voters (n)')
    ax.set_ylabel('Number of Alternatives (m)')
    ax.set_title('Elections: PVC Proportion')
    plt.tight_layout()
    plt.savefig(output_dir / 'scatter_pvc.png', dpi=150)
    plt.close()
    logger.info("Created scatter_pvc.png")
    
    # 6. Scatter plot: voters vs alternatives, colored by epsilon
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(df['n'], df['m'], c=df['epsilon'], 
                         cmap='plasma', alpha=0.7, s=60, edgecolors='white', linewidth=0.5)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Effective Epsilon')
    ax.set_xlabel('Number of Voters (n)')
    ax.set_ylabel('Number of Alternatives (m)')
    ax.set_title('Elections: Effective Epsilon')
    plt.tight_layout()
    plt.savefig(output_dir / 'scatter_epsilon.png', dpi=150)
    plt.close()
    logger.info("Created scatter_epsilon.png")


def main():
    """Main entry point."""
    logger.info("=" * 60)
    logger.info("PVC Analysis of Historical Elections")
    logger.info("=" * 60)
    
    all_results = []
    
    # Process each dataset
    for name, config in DATASETS.items():
        logger.info(f"\nProcessing {name} dataset...")
        elections = load_and_filter_dataset(name, config)
        
        for i, election in enumerate(elections):
            logger.info(f"  [{i+1}/{len(elections)}] {election['file']} (n={election['n']}, m={election['m']})")
            result = compute_election_metrics(election)
            if result:
                all_results.append(result)
                logger.info(f"    PVC: {result['pvc_size']}/{result['m']} ({result['pvc_proportion']:.2%}), "
                           f"ε*={result['epsilon']:.4f}")
    
    if not all_results:
        logger.error("No valid elections found!")
        return
    
    # Create DataFrame
    df = pd.DataFrame(all_results)
    
    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    
    for regime in ['n < m', 'n = m', 'n > m']:
        regime_df = df[df['regime'] == regime]
        if len(regime_df) > 0:
            logger.info(f"\n{regime}:")
            logger.info(f"  Elections: {len(regime_df)}")
            logger.info(f"  PVC Proportion: {regime_df['pvc_proportion'].mean():.2%} ± {regime_df['pvc_proportion'].std():.2%}")
            logger.info(f"  Epsilon: {regime_df['epsilon'].mean():.4f} ± {regime_df['epsilon'].std():.4f}")
    
    # Save results to CSV
    output_dir = Path(__file__).parent / 'plots'
    df.to_csv(output_dir / 'results.csv', index=False)
    logger.info(f"\nResults saved to {output_dir / 'results.csv'}")
    
    # Create visualizations
    logger.info("\nCreating visualizations...")
    create_visualizations(df, output_dir)
    
    logger.info("\n" + "=" * 60)
    logger.info("Analysis complete!")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()


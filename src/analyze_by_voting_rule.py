#!/usr/bin/env python3
"""
PVC Analysis by Voting Rule.

For Polish, Eurovision, and ERS datasets:
- Compute winners using 4 voting rules: Plurality, IRV, Borda, Schulze
- Calculate critical epsilon for each winner
- Generate strip plots (by group v2) and scatter plots per voting rule

Uses VoteKit for winner computation and pvc_toolbox for epsilon.
"""

import os
import glob
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from preflibtools.instances import OrdinalInstance
from pvc_toolbox import compute_critical_epsilon, compute_pvc

# VoteKit imports
from votekit import PreferenceProfile, Ballot
from votekit.elections import STV, Plurality, Borda

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Dataset configurations (excluding Spotify)
DATASETS = {
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

# Voting rules to analyze
VOTING_RULES = ['veto', 'borda', 'schulze', 'irv', 'plurality']

MIN_SAMPLES = 10


def classify_regime(n, m):
    """Classify into regime using 2x multiplier."""
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


def instance_to_votekit_profile(instance: OrdinalInstance) -> PreferenceProfile:
    """Convert preflibtools instance to VoteKit PreferenceProfile."""
    all_alternatives = set(instance.alternatives_name.keys())
    full_profile = instance.full_profile()
    
    ballots = []
    for order in full_profile:
        extended = extend_incomplete_order(order, all_alternatives)
        # Convert to ranking format: list of sets
        ranking = []
        for rank_tuple in extended:
            # Each rank is a frozenset of candidates at that rank
            rank_set = frozenset(str(c) for c in rank_tuple)
            ranking.append(rank_set)
        
        ballot = Ballot(ranking=tuple(ranking), weight=1)
        ballots.append(ballot)
    
    candidates = frozenset(str(a) for a in all_alternatives)
    return PreferenceProfile(ballots=ballots, candidates=candidates)


def compute_plurality_winner(profile: PreferenceProfile) -> Optional[str]:
    """Compute plurality winner using VoteKit."""
    try:
        election = Plurality(profile, m=1)
        elected = election.get_elected()
        if elected and len(elected) > 0:
            # get_elected() returns tuple of frozensets
            first_winners = elected[0]
            if first_winners:
                return list(first_winners)[0]
        return None
    except Exception as e:
        logger.debug(f"Plurality error: {e}")
        return None


def compute_irv_winner(profile: PreferenceProfile) -> Optional[str]:
    """Compute IRV winner using VoteKit STV with 1 seat."""
    try:
        election = STV(profile, m=1)
        elected = election.get_elected()
        if elected and len(elected) > 0:
            first_winners = elected[0]
            if first_winners:
                return list(first_winners)[0]
        return None
    except Exception as e:
        logger.debug(f"IRV error: {e}")
        return None


def compute_borda_winner(profile: PreferenceProfile) -> Optional[str]:
    """Compute Borda winner using VoteKit."""
    try:
        election = Borda(profile, m=1)
        elected = election.get_elected()
        if elected and len(elected) > 0:
            first_winners = elected[0]
            if first_winners:
                return list(first_winners)[0]
        return None
    except Exception as e:
        logger.debug(f"Borda error: {e}")
        return None


def compute_schulze_winner_manual(preferences: List[List[str]], alternatives: List[str]) -> Optional[str]:
    """
    Compute Schulze winner using manual implementation.
    
    The Schulze method finds the winner based on beatpath strengths.
    """
    try:
        n_alts = len(alternatives)
        alt_to_idx = {alt: i for i, alt in enumerate(alternatives)}
        
        # Build pairwise preference matrix
        # pairwise[i][j] = number of voters preferring i over j
        pairwise = [[0] * n_alts for _ in range(n_alts)]
        
        n_voters = len(preferences[0])
        for voter in range(n_voters):
            # Get this voter's ranking
            voter_ranking = [preferences[rank][voter] for rank in range(len(preferences))]
            
            # For each pair of alternatives
            for i, alt_i in enumerate(voter_ranking):
                for j in range(i + 1, len(voter_ranking)):
                    alt_j = voter_ranking[j]
                    # alt_i is ranked higher (preferred) over alt_j
                    idx_i = alt_to_idx[alt_i]
                    idx_j = alt_to_idx[alt_j]
                    pairwise[idx_i][idx_j] += 1
        
        # Compute strongest path strengths using Floyd-Warshall variant
        strength = [[0] * n_alts for _ in range(n_alts)]
        
        for i in range(n_alts):
            for j in range(n_alts):
                if i != j:
                    if pairwise[i][j] > pairwise[j][i]:
                        strength[i][j] = pairwise[i][j]
        
        # Floyd-Warshall to find strongest paths
        for k in range(n_alts):
            for i in range(n_alts):
                if i != k:
                    for j in range(n_alts):
                        if j != i and j != k:
                            strength[i][j] = max(
                                strength[i][j],
                                min(strength[i][k], strength[k][j])
                            )
        
        # Find Schulze winner: candidate who beats all others via beatpath
        for i in range(n_alts):
            is_winner = True
            for j in range(n_alts):
                if i != j and strength[j][i] > strength[i][j]:
                    is_winner = False
                    break
            if is_winner:
                return alternatives[i]
        
        return alternatives[0]  # Fallback
    except Exception as e:
        logger.debug(f"Schulze error: {e}")
        return None


def compute_veto_winner(preferences: List[List[str]], alternatives: List[str]) -> Optional[str]:
    """
    Compute winner using Veto by Consumption (PVC).
    
    Selects a winner from the Proportional Veto Core. If PVC has multiple
    candidates, selects the one with highest plurality score (most first-place votes).
    """
    try:
        # Compute the PVC
        pvc = compute_pvc(preferences, alternatives)
        
        if not pvc:
            return None
        
        if len(pvc) == 1:
            return pvc[0]
        
        # If multiple candidates in PVC, use plurality as tiebreaker
        first_place_counts = {}
        for voter_pref in preferences[0]:
            if voter_pref in pvc:
                first_place_counts[voter_pref] = first_place_counts.get(voter_pref, 0) + 1
        
        # If no PVC member got first place votes, just return first PVC member
        if not first_place_counts:
            return pvc[0]
        
        winner = max(first_place_counts, key=first_place_counts.get)
        return winner
    except Exception as e:
        logger.debug(f"Veto error: {e}")
        return None


def get_winner(profile: PreferenceProfile, preferences: List[List[str]], 
               alternatives: List[str], method: str) -> Optional[str]:
    """Get winner using specified voting method."""
    if method == 'plurality':
        return compute_plurality_winner(profile)
    elif method == 'irv':
        return compute_irv_winner(profile)
    elif method == 'borda':
        return compute_borda_winner(profile)
    elif method == 'schulze':
        return compute_schulze_winner_manual(preferences, alternatives)
    elif method == 'veto':
        return compute_veto_winner(preferences, alternatives)
    return None


def load_all_elections():
    """Load all elections from all datasets."""
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
                
                # Classify regime (may be None for elections not in any regime)
                regime = classify_regime(n, m)
                regime_label = regime if regime else 'other'
                
                all_elections.append({
                    'file': os.path.basename(filepath),
                    'dataset': name,
                    'dataset_label': config['label'],
                    'instance': instance,
                    'n': n,
                    'm': m,
                    'regime': regime_label,
                    'group': f"{config['label']} ({regime_label})"
                })
            except Exception as e:
                logger.warning(f"Error loading {filepath}: {e}")
    
    return all_elections


def compute_metrics_for_rule(election: Dict[str, Any], voting_rule: str) -> Optional[Dict[str, Any]]:
    """Compute epsilon for winner under specified voting rule."""
    instance = election['instance']
    
    try:
        preferences, alternatives = instance_to_preferences(instance)
        profile = instance_to_votekit_profile(instance)
        
        # Get winner under this voting rule
        winner = get_winner(profile, preferences, alternatives, voting_rule)
        
        if winner is None:
            return None
        
        # Compute critical epsilon for this winner
        epsilon = compute_critical_epsilon(preferences, alternatives, winner)
        
        return {
            'file': election['file'],
            'dataset': election['dataset'],
            'dataset_label': election['dataset_label'],
            'n': election['n'],
            'm': election['m'],
            'regime': election['regime'],
            'group': election['group'],
            'voting_rule': voting_rule,
            'winner': winner,
            'epsilon': epsilon
        }
    except Exception as e:
        logger.debug(f"Error computing metrics for {election['file']} ({voting_rule}): {e}")
        return None


def create_strip_plot_v2(df: pd.DataFrame, output_dir: Path, voting_rule: str):
    """Create strip plot (by group v2) for epsilon under a voting rule."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Apply 100x100 cap for plots
    df = df[(df['n'] <= 100) & (df['m'] <= 100)].copy()
    
    if df.empty:
        logger.warning(f"No elections with n,m <= 100 for {voting_rule}")
        return
    
    # Count samples per group
    group_counts = df['group'].value_counts()
    valid_groups = group_counts[group_counts > MIN_SAMPLES].index.tolist()
    
    if not valid_groups:
        logger.warning(f"No groups with >{MIN_SAMPLES} samples for {voting_rule}")
        return
    
    # Sort groups: by dataset first, then by regime
    dataset_order = {'Eurovision': 0, 'Polish': 1, 'ERS': 2}
    regime_order = {'n ≪ m': 0, 'n ≈ m': 1, 'n ≫ m': 2, 'other': 3}
    valid_groups = sorted(valid_groups, 
                         key=lambda x: (dataset_order.get(x.split(' (')[0], 9), 
                                       regime_order.get(x.split('(')[1].rstrip(')'), 4)))
    
    # Filter dataframe
    df_filtered = df[df['group'].isin(valid_groups)].copy()
    
    if df_filtered.empty:
        return
    
    # Add count to group labels
    df_filtered['group_label'] = df_filtered['group'].apply(
        lambda x: f"{x} (k={group_counts[x]})"
    )
    group_labels = [f"{g} (k={group_counts[g]})" for g in valid_groups]
    
    # Regime color palette
    regime_colors = {'n ≪ m': 'blue', 'n ≈ m': 'green', 'n ≫ m': 'red', 'other': 'gray'}
    
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
    
    # Add x-axis jitter
    np.random.seed(42)
    df_filtered['epsilon_jitter'] = df_filtered['epsilon'] + np.random.normal(0, 0.002, len(df_filtered))
    
    # Create strip plot
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
    ax.set_xlabel('Critical Epsilon of Winner')
    ax.set_ylabel('Dataset (Regime)')
    ax.set_title(f'Critical Epsilon by Dataset-Regime ({voting_rule.upper()} Winner)')
    plt.tight_layout()
    plt.savefig(output_dir / 'epsilon_by_group_strip_v2.png', dpi=150)
    plt.close()
    logger.info(f"Created {voting_rule}/epsilon_by_group_strip_v2.png")


def create_scatter_plot(df: pd.DataFrame, output_dir: Path, voting_rule: str, jitter_strength=0.5):
    """Create scatter plot (n vs m colored by epsilon) for a voting rule."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Apply 100x100 cap for plots
    df = df[(df['n'] <= 100) & (df['m'] <= 100)].copy()
    
    if df.empty:
        return
    
    np.random.seed(42)
    n_jitter = df['n'].values + np.random.normal(0, jitter_strength, len(df))
    m_jitter = df['m'].values + np.random.normal(0, jitter_strength, len(df))
    
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(n_jitter, m_jitter, 
                        c=df['epsilon'], 
                        cmap='plasma', alpha=0.7, s=60, 
                        edgecolors='white', linewidth=0.5)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Critical Epsilon')
    ax.set_xlabel('Number of Voters (n)')
    ax.set_ylabel('Number of Alternatives (m)')
    ax.set_title(f'Critical Epsilon ({voting_rule.upper()} Winner)')
    plt.tight_layout()
    plt.savefig(output_dir / 'scatter_epsilon.png', dpi=150)
    plt.close()
    logger.info(f"Created {voting_rule}/scatter_epsilon.png")


def create_aggregate_dataset_plots(all_results_df: pd.DataFrame, output_dir: Path):
    """Create aggregate plots per dataset showing each voting method."""
    # Apply 100x100 cap for plots
    all_results_df = all_results_df[(all_results_df['n'] <= 100) & (all_results_df['m'] <= 100)].copy()
    
    datasets = all_results_df['dataset_label'].unique()
    
    rule_colors = {
        'veto': '#FF7F00',       # Orange
        'borda': '#2ECC71',      # Green
        'schulze': '#3498DB',    # Blue
        'irv': '#E74C3C',        # Red
        'plurality': '#9B59B6'   # Purple
    }
    
    for dataset in datasets:
        dataset_dir = output_dir / dataset.lower()
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        df_dataset = all_results_df[all_results_df['dataset_label'] == dataset].copy()
        
        if df_dataset.empty:
            continue
        
        # Count per voting rule
        rule_counts = df_dataset['voting_rule'].value_counts()
        
        # Add count to labels
        df_dataset['rule_label'] = df_dataset['voting_rule'].apply(
            lambda x: f"{x.upper()} (k={rule_counts.get(x, 0)})"
        )
        rule_labels = [f"{r.upper()} (k={rule_counts.get(r, 0)})" for r in VOTING_RULES if r in rule_counts]
        
        sns.set_style("whitegrid")
        
        # Add jitter
        np.random.seed(42)
        df_dataset['epsilon_jitter'] = df_dataset['epsilon'] + np.random.normal(0, 0.002, len(df_dataset))
        
        # Strip plot
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.stripplot(data=df_dataset, x='epsilon_jitter', y='rule_label',
                      hue='voting_rule', order=rule_labels,
                      palette=rule_colors, jitter=0.35, alpha=0.6, size=6, ax=ax,
                      legend=False)
        
        # Add mean markers
        for i, rule in enumerate(VOTING_RULES):
            if rule in rule_counts:
                rule_data = df_dataset[df_dataset['voting_rule'] == rule]['epsilon']
                if len(rule_data) > 0:
                    mean_val = rule_data.mean()
                    ax.scatter([mean_val], [i], marker='D', s=100, c='black', 
                              edgecolors='white', linewidth=1.5, zorder=10)
        
        ax.scatter([], [], marker='D', s=100, c='black', edgecolors='white', linewidth=1.5, label='Mean')
        ax.legend(loc='upper right')
        ax.set_xlabel('Critical Epsilon')
        ax.set_ylabel('Voting Rule')
        ax.set_title(f'{dataset}: Critical Epsilon by Voting Rule')
        plt.tight_layout()
        plt.savefig(dataset_dir / 'epsilon_by_rule_strip.png', dpi=150)
        plt.close()
        logger.info(f"Created aggregate/{dataset.lower()}/epsilon_by_rule_strip.png")
        
        # Bar plot with 95% CI
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(data=df_dataset, x='voting_rule', y='epsilon',
                    order=VOTING_RULES, palette=rule_colors,
                    errorbar=('ci', 95), capsize=0.1, ax=ax)
        ax.set_xlabel('Voting Rule')
        ax.set_ylabel('Mean Critical Epsilon')
        ax.set_title(f'{dataset}: Critical Epsilon by Voting Rule (95% CI)')
        ax.set_xticklabels([r.upper() for r in VOTING_RULES])
        plt.tight_layout()
        plt.savefig(dataset_dir / 'epsilon_by_rule_bar.png', dpi=150)
        plt.close()
        logger.info(f"Created aggregate/{dataset.lower()}/epsilon_by_rule_bar.png")


def create_aggregate_all_plots(all_results_df: pd.DataFrame, output_dir: Path):
    """Create aggregate plots across all datasets showing each voting method."""
    all_dir = output_dir / 'all'
    all_dir.mkdir(parents=True, exist_ok=True)
    
    # Apply 100x100 cap for plots
    df = all_results_df[(all_results_df['n'] <= 100) & (all_results_df['m'] <= 100)].copy()
    
    if df.empty:
        return
    
    rule_colors = {
        'veto': '#FF7F00',       # Orange
        'borda': '#2ECC71',      # Green
        'schulze': '#3498DB',    # Blue
        'irv': '#E74C3C',        # Red
        'plurality': '#9B59B6'   # Purple
    }
    
    # Count per voting rule
    rule_counts = df['voting_rule'].value_counts()
    
    # Add count to labels
    df['rule_label'] = df['voting_rule'].apply(
        lambda x: f"{x.upper()} (k={rule_counts.get(x, 0)})"
    )
    rule_labels = [f"{r.upper()} (k={rule_counts.get(r, 0)})" for r in VOTING_RULES if r in rule_counts]
    
    sns.set_style("whitegrid")
    
    # Add jitter
    np.random.seed(42)
    df['epsilon_jitter'] = df['epsilon'] + np.random.normal(0, 0.002, len(df))
    
    # Strip plot - colored by dataset
    dataset_colors = {'Polish': '#DC143C', 'Eurovision': '#0066CC', 'ERS': '#FF8C00'}
    
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.stripplot(data=df, x='epsilon_jitter', y='rule_label',
                  hue='dataset_label', order=rule_labels,
                  palette=dataset_colors, jitter=0.35, alpha=0.5, size=5, ax=ax)
    
    # Add mean markers
    for i, rule in enumerate(VOTING_RULES):
        if rule in rule_counts:
            rule_data = df[df['voting_rule'] == rule]['epsilon']
            if len(rule_data) > 0:
                mean_val = rule_data.mean()
                ax.scatter([mean_val], [i], marker='D', s=100, c='black', 
                          edgecolors='white', linewidth=1.5, zorder=10)
    
    ax.scatter([], [], marker='D', s=100, c='black', edgecolors='white', linewidth=1.5, label='Mean')
    ax.legend(loc='upper right', title='Dataset')
    ax.set_xlabel('Critical Epsilon')
    ax.set_ylabel('Voting Rule')
    ax.set_title('All Datasets: Critical Epsilon by Voting Rule')
    plt.tight_layout()
    plt.savefig(all_dir / 'epsilon_by_rule_strip.png', dpi=150)
    plt.close()
    logger.info("Created aggregate/all/epsilon_by_rule_strip.png")
    
    # Strip plot - colored by voting rule
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.stripplot(data=df, x='epsilon_jitter', y='rule_label',
                  hue='voting_rule', order=rule_labels,
                  palette=rule_colors, jitter=0.35, alpha=0.5, size=5, ax=ax,
                  legend=False)
    
    # Add mean markers
    for i, rule in enumerate(VOTING_RULES):
        if rule in rule_counts:
            rule_data = df[df['voting_rule'] == rule]['epsilon']
            if len(rule_data) > 0:
                mean_val = rule_data.mean()
                ax.scatter([mean_val], [i], marker='D', s=100, c='black', 
                          edgecolors='white', linewidth=1.5, zorder=10)
    
    ax.scatter([], [], marker='D', s=100, c='black', edgecolors='white', linewidth=1.5, label='Mean')
    ax.legend(loc='upper right')
    ax.set_xlabel('Critical Epsilon')
    ax.set_ylabel('Voting Rule')
    ax.set_title('All Datasets: Critical Epsilon by Voting Rule')
    plt.tight_layout()
    plt.savefig(all_dir / 'epsilon_by_rule_strip_v2.png', dpi=150)
    plt.close()
    logger.info("Created aggregate/all/epsilon_by_rule_strip_v2.png")
    
    # Bar plot with 95% CI
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=df, x='voting_rule', y='epsilon',
                order=VOTING_RULES, palette=rule_colors,
                errorbar=('ci', 95), capsize=0.1, ax=ax)
    ax.set_xlabel('Voting Rule')
    ax.set_ylabel('Mean Critical Epsilon')
    ax.set_title('All Datasets: Critical Epsilon by Voting Rule (95% CI)')
    ax.set_xticklabels([r.upper() for r in VOTING_RULES])
    plt.tight_layout()
    plt.savefig(all_dir / 'epsilon_by_rule_bar.png', dpi=150)
    plt.close()
    logger.info("Created aggregate/all/epsilon_by_rule_bar.png")
    
    # Save combined results
    df.to_csv(all_dir / 'all_results.csv', index=False)
    logger.info(f"Saved aggregate/all/all_results.csv")


def main():
    logger.info("=" * 60)
    logger.info("PVC Analysis by Voting Rule")
    logger.info("=" * 60)
    
    # Load all elections
    elections = load_all_elections()
    logger.info(f"Loaded {len(elections)} elections total")
    
    # Create output directories
    plots_dir = Path(__file__).parent.parent / 'plots'
    
    # Collect all results across voting rules for aggregate plots
    all_results = []
    
    for voting_rule in VOTING_RULES:
        rule_dir = plots_dir / voting_rule
        rule_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"\n{'=' * 40}")
        logger.info(f"Processing voting rule: {voting_rule.upper()}")
        logger.info(f"{'=' * 40}")
        
        # Compute metrics for this voting rule
        results = []
        for i, election in enumerate(elections):
            if (i + 1) % 100 == 0:
                logger.info(f"  Processing {i+1}/{len(elections)}...")
            result = compute_metrics_for_rule(election, voting_rule)
            if result:
                results.append(result)
                all_results.append(result)  # Also add to aggregate
        
        logger.info(f"  Computed metrics for {len(results)} elections")
        
        if not results:
            logger.warning(f"  No results for {voting_rule}")
            continue
        
        # Create DataFrame
        df = pd.DataFrame(results)
        
        # Save results
        df.to_csv(rule_dir / 'results.csv', index=False)
        logger.info(f"  Saved results to {rule_dir / 'results.csv'}")
        
        # Print summary
        logger.info(f"\n  Summary for {voting_rule.upper()}:")
        for group in sorted(df['group'].unique()):
            gdf = df[df['group'] == group]
            logger.info(f"    {group}: {len(gdf)} elections, ε = {gdf['epsilon'].mean():.4f} ± {gdf['epsilon'].std():.4f}")
        
        # Create visualizations
        create_strip_plot_v2(df, rule_dir, voting_rule)
        create_scatter_plot(df, rule_dir, voting_rule)
    
    # Create aggregate plots
    logger.info(f"\n{'=' * 40}")
    logger.info("Creating aggregate plots")
    logger.info(f"{'=' * 40}")
    
    if all_results:
        all_results_df = pd.DataFrame(all_results)
        aggregate_dir = plots_dir / 'aggregate'
        aggregate_dir.mkdir(parents=True, exist_ok=True)
        
        # Per-dataset aggregate plots
        create_aggregate_dataset_plots(all_results_df, aggregate_dir)
        
        # All-aggregate plots
        create_aggregate_all_plots(all_results_df, aggregate_dir)
    
    logger.info("\n" + "=" * 60)
    logger.info("Analysis complete!")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()


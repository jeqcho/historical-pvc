#!/usr/bin/env python3
"""
Compute voting method results and save to CSV.

This script computes winners and critical epsilon for all voting rules
across all datasets, saving results to a single CSV for fast visualization.
"""

import os
import glob
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

import pandas as pd

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

# Dataset configurations
DATASETS = {
    'polish': {
        'path': '../data/polish',
        'patterns': ['*.soi'],
        'label': 'Polish'
    },
    'eurovision': {
        'path': '../data/eurovision',
        'patterns': ['*.soi'],
        'label': 'Eurovision'
    },
    'ers': {
        'path': '../data/ers',
        'patterns': ['*.soi'],
        'label': 'ERS'
    },
    'skate': {
        'path': '../data/skate',
        'patterns': ['*.soc', '*.toc'],
        'label': 'Skate'
    },
    'habermas': {
        'path': '../data/habermas',
        'patterns': ['*.soc'],
        'label': 'Habermas'
    }
}

# Voting rules to analyze
VOTING_RULES = ['veto', 'borda', 'schulze', 'irv', 'plurality']

# Output file
RESULTS_FILE = Path(__file__).parent.parent / 'data' / 'voting_results.csv'


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
    """Convert preflibtools instance to pvc_toolbox format.
    
    Returns preferences as preferences[rank][voter] format required by pvc_toolbox.
    """
    all_alternatives = set(instance.alternatives_name.keys())
    full_profile = instance.full_profile()
    
    extended_profile = []
    for order in full_profile:
        extended = extend_incomplete_order(order, all_alternatives)
        extended_profile.append(extended)
    
    num_ranks = len(all_alternatives)
    alternatives = [str(a) for a in sorted(all_alternatives)]
    
    # Build preferences as preferences[rank][voter]
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
        ranking = []
        for rank_tuple in extended:
            ranking.append(frozenset([str(alt) for alt in rank_tuple]))
        ballots.append(Ballot(ranking=tuple(ranking), weight=1))
    
    return PreferenceProfile(ballots=ballots)


def compute_plurality_winner(profile: PreferenceProfile) -> Optional[str]:
    """Compute winner using Plurality voting."""
    try:
        plurality = Plurality(profile, m=1)
        elected = plurality.get_elected()
        if elected and len(elected) > 0:
            first_round = elected[0]
            if first_round:
                return list(first_round)[0]
        return None
    except Exception as e:
        logger.debug(f"Plurality error: {e}")
        return None


def compute_irv_winner(profile: PreferenceProfile) -> Optional[str]:
    """Compute winner using Instant Runoff Voting (STV with 1 seat)."""
    try:
        irv = STV(profile, m=1)
        elected = irv.get_elected()
        if elected and len(elected) > 0:
            first_round = elected[0]
            if first_round:
                return list(first_round)[0]
        return None
    except Exception as e:
        logger.debug(f"IRV error: {e}")
        return None


def compute_borda_winner(profile: PreferenceProfile) -> Optional[str]:
    """Compute winner using Borda count."""
    try:
        borda = Borda(profile, m=1)
        elected = borda.get_elected()
        if elected and len(elected) > 0:
            first_round = elected[0]
            if first_round:
                return list(first_round)[0]
        return None
    except Exception as e:
        logger.debug(f"Borda error: {e}")
        return None


def compute_schulze_winner_manual(preferences: List[List[str]], alternatives: List[str]) -> Optional[str]:
    """
    Compute Schulze winner using manual implementation.
    
    Expects preferences in preferences[rank][voter] format.
    """
    try:
        n_alts = len(alternatives)
        alt_to_idx = {alt: i for i, alt in enumerate(alternatives)}
        
        # Build pairwise preference matrix
        pairwise = [[0] * n_alts for _ in range(n_alts)]
        
        n_voters = len(preferences[0])
        for voter in range(n_voters):
            # Get this voter's ranking
            voter_ranking = [preferences[rank][voter] for rank in range(len(preferences))]
            
            # For each pair of alternatives
            for i, alt_i in enumerate(voter_ranking):
                idx_i = alt_to_idx.get(alt_i)
                if idx_i is None:
                    continue
                for j, alt_j in enumerate(voter_ranking[i+1:], i+1):
                    idx_j = alt_to_idx.get(alt_j)
                    if idx_j is None:
                        continue
                    # Voter prefers alt_i over alt_j
                    pairwise[idx_i][idx_j] += 1
        
        # Compute strongest path strengths (Floyd-Warshall)
        strength = [[0] * n_alts for _ in range(n_alts)]
        for i in range(n_alts):
            for j in range(n_alts):
                if i != j:
                    if pairwise[i][j] > pairwise[j][i]:
                        strength[i][j] = pairwise[i][j]
        
        for k in range(n_alts):
            for i in range(n_alts):
                if i != k:
                    for j in range(n_alts):
                        if j != i and j != k:
                            strength[i][j] = max(strength[i][j], min(strength[i][k], strength[k][j]))
        
        # Find winner (beats all others in strongest path comparison)
        for i in range(n_alts):
            is_winner = True
            for j in range(n_alts):
                if i != j and strength[j][i] > strength[i][j]:
                    is_winner = False
                    break
            if is_winner:
                return alternatives[i]
        
        return None
    except Exception as e:
        logger.debug(f"Schulze error: {e}")
        return None


def compute_veto_winner(preferences: List[List[str]], alternatives: List[str]) -> Optional[str]:
    """
    Compute winner using Veto by Consumption (PVC).
    
    Selects a winner from the Proportional Veto Core. If PVC has multiple
    candidates, selects the one with highest plurality score (most first-place votes).
    
    Expects preferences in preferences[rank][voter] format.
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
        for voter_pref in preferences[0]:  # preferences[0] = first-place votes
            if voter_pref in pvc:
                first_place_counts[voter_pref] = first_place_counts.get(voter_pref, 0) + 1
        
        if not first_place_counts:
            return pvc[0]  # fallback
        
        # Return candidate with most first-place votes
        return max(first_place_counts, key=first_place_counts.get)
    except Exception as e:
        logger.debug(f"Veto winner computation error: {e}")
        return None


def get_winner(profile: PreferenceProfile, preferences: List[List[str]], 
               alternatives: List[str], method: str) -> Optional[str]:
    """Get election winner under specified method."""
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
    else:
        raise ValueError(f"Unknown voting method: {method}")


def load_all_elections():
    """Load all elections from all datasets."""
    all_elections = []
    
    for name, config in DATASETS.items():
        base_path = Path(__file__).parent / config['path']
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
        
        winner = get_winner(profile, preferences, alternatives, voting_rule)
        
        if winner is None:
            return None
        
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


def main():
    logger.info("=" * 60)
    logger.info("Computing Voting Method Results")
    logger.info("=" * 60)
    
    # Load all elections
    elections = load_all_elections()
    logger.info(f"Loaded {len(elections)} elections total")
    
    # Compute all results
    all_results = []
    
    for voting_rule in VOTING_RULES:
        logger.info(f"\n{'=' * 40}")
        logger.info(f"Processing voting rule: {voting_rule.upper()}")
        logger.info(f"{'=' * 40}")
        
        for i, election in enumerate(elections):
            if (i + 1) % 100 == 0:
                logger.info(f"  Processing {i+1}/{len(elections)}...")
            result = compute_metrics_for_rule(election, voting_rule)
            if result:
                all_results.append(result)
        
        rule_count = sum(1 for r in all_results if r['voting_rule'] == voting_rule)
        logger.info(f"  Computed metrics for {rule_count} elections")
    
    # Save to CSV
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(all_results)
    df.to_csv(RESULTS_FILE, index=False)
    
    logger.info(f"\n{'=' * 60}")
    logger.info(f"Saved {len(all_results)} results to {RESULTS_FILE}")
    logger.info("=" * 60)
    
    # Print summary
    logger.info("\nSummary by voting rule:")
    for rule in VOTING_RULES:
        rule_df = df[df['voting_rule'] == rule]
        logger.info(f"  {rule.upper()}: {len(rule_df)} results, mean ε = {rule_df['epsilon'].mean():.4f}")


if __name__ == '__main__':
    main()


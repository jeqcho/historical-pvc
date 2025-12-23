#!/usr/bin/env python3
"""
Plot n (voters) vs m (alternatives) scatter plots for each dataset.
This helps visualize the distribution and adjust bounds if needed.
"""

import os
import glob
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from preflibtools.instances import OrdinalInstance

# Dataset configurations - using all files (no filtering)
DATASETS = {
    'spotify': {
        'path': '../data/spotify',
        'patterns': ['*.soc'],
        'title': 'Spotify Countries (n ≪ m)',
        'bounds': {'n_min': 20, 'n_max': 40, 'm_min': 80, 'm_max': 100},
        'target_regime': 'n < m'
    },
    'polish': {
        'path': '../data/polish',
        'patterns': ['*.soi'],
        'title': 'Polish Elections (n ≈ m)',
        'bounds': {'n_min': 10, 'n_max': 20, 'm_min': 10, 'm_max': 20},
        'target_regime': 'n = m'
    },
    'eurovision': {
        'path': '../data/eurovision',
        'patterns': ['*.soi'],
        'title': 'Eurovision (n ≫ m)',
        'bounds': {'n_min': 80, 'n_max': 100, 'm_min': 20, 'm_max': 40},
        'target_regime': 'n > m'
    },
    'ers': {
        'path': '../data/ers',
        'patterns': ['*.soi'],
        'title': 'ERS (n ≫ m)',
        'bounds': None,
        'target_regime': 'n > m'
    },
    'skate': {
        'path': '../data/skate',
        'patterns': ['*.soc', '*.toc'],
        'title': 'Figure Skating (n ≪ m)',
        'bounds': None,
        'target_regime': 'n < m'
    }
}


def load_dataset_stats(name: str, config: dict) -> list:
    """Load n, m values for all files in a dataset."""
    base_path = Path(__file__).parent / config['path']
    # Handle both 'pattern' (single) and 'patterns' (list)
    patterns = config.get('patterns', [config.get('pattern', '*.soi')])
    files = []
    for pattern in patterns:
        files.extend(sorted(glob.glob(str(base_path / pattern))))
    
    results = []
    for filepath in files:
        try:
            instance = OrdinalInstance()
            instance.parse_file(filepath)
            results.append({
                'n': instance.num_voters,
                'm': instance.num_alternatives,
                'file': os.path.basename(filepath),
                'dataset': name
            })
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
    
    return results


def classify_regime(n, m):
    """
    Classify a point into a regime based on n and m.
    
    - n < m: 2n <= m (m is at least 2x larger than n)
    - n > m: 2m <= n (n is at least 2x larger than m)
    - n = m: |m-n|/min(m,n) < 0.25 (within 25% tolerance)
    
    Returns: 'n < m', 'n = m', 'n > m', or None if doesn't fit any
    """
    min_val = min(n, m)
    if min_val == 0:
        return None
    
    # Check n << m: 2n <= m
    if 2 * n <= m:
        return 'n ≪ m'
    
    # Check n >> m: 2m <= n
    if 2 * m <= n:
        return 'n ≫ m'
    
    # Check n ≈ m: |m-n|/min(m,n) < 0.25
    if abs(m - n) / min_val < 0.25:
        return 'n ≈ m'
    
    return None


def plot_dataset(name: str, config: dict, ax, jitter_strength=0.3):
    """Plot n vs m scatter for a dataset with jitter."""
    data = load_dataset_stats(name, config)
    
    if not data:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(config['title'])
        return
    
    n_vals = np.array([d['n'] for d in data])
    m_vals = np.array([d['m'] for d in data])
    
    # Add jitter
    n_jitter = n_vals + np.random.normal(0, jitter_strength * np.std(n_vals) * 0.1 + 0.5, len(n_vals))
    m_jitter = m_vals + np.random.normal(0, jitter_strength * np.std(m_vals) * 0.1 + 0.5, len(m_vals))
    
    # Plot points
    ax.scatter(n_jitter, m_jitter, c='black', alpha=0.5, s=20, edgecolors='none')
    
    # Draw bounds rectangle if specified
    bounds = config.get('bounds')
    if bounds:
        from matplotlib.patches import Rectangle
        rect = Rectangle(
            (bounds['n_min'], bounds['m_min']),
            bounds['n_max'] - bounds['n_min'],
            bounds['m_max'] - bounds['m_min'],
            linewidth=2, edgecolor='red', facecolor='red', alpha=0.1
        )
        ax.add_patch(rect)
        
        # Count points in bounds
        in_bounds = sum(1 for d in data 
                       if bounds['n_min'] <= d['n'] <= bounds['n_max'] 
                       and bounds['m_min'] <= d['m'] <= bounds['m_max'])
        ax.text(0.98, 0.02, f'In bounds: {in_bounds}/{len(data)}',
               transform=ax.transAxes, ha='right', va='bottom',
               fontsize=9, color='red')
    
    ax.set_xlabel('n (voters)')
    ax.set_ylabel('m (alternatives)')
    ax.set_title(f"{config['title']} (k={len(data)})")
    
    # Add stats
    stats_text = f"n: [{min(n_vals)}, {max(n_vals)}]\nm: [{min(m_vals)}, {max(m_vals)}]"
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
            ha='left', va='top', fontsize=8, 
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))


def plot_dataset_regime_colored(name: str, config: dict, ax, jitter_strength=0.5):
    """
    Plot a single dataset with points colored by regime classification.
    Capped at 100x100.
    """
    data = load_dataset_stats(name, config)
    
    if not data:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(config['title'])
        return
    
    colors = {'n ≪ m': 'blue', 'n ≈ m': 'green', 'n ≫ m': 'red', None: 'gray'}
    
    # Separate by regime
    regime_data = {'n ≪ m': [], 'n ≈ m': [], 'n ≫ m': [], None: []}
    
    for d in data:
        n, m = d['n'], d['m']
        # Only include points within 100x100
        if n <= 100 and m <= 100:
            regime = classify_regime(n, m)
            regime_data[regime].append((n, m))
    
    # Plot each regime
    for regime, points in regime_data.items():
        if not points:
            continue
        n_vals = np.array([p[0] for p in points])
        m_vals = np.array([p[1] for p in points])
        
        # Add jitter
        n_jitter = n_vals + np.random.normal(0, jitter_strength, len(n_vals))
        m_jitter = m_vals + np.random.normal(0, jitter_strength, len(m_vals))
        
        label = regime if regime else 'Unclassified'
        ax.scatter(n_jitter, m_jitter, c=colors[regime], alpha=0.6, s=30, 
                  edgecolors='white', linewidth=0.3, label=f'{label} ({len(points)})')
    
    # Draw regime boundaries
    x = np.linspace(0, 100, 100)
    
    # n < m boundary: m = 2n
    ax.plot(x, 2*x, 'b--', alpha=0.5, linewidth=1.5)
    
    # n > m boundary: n = 2m, so m = n/2
    ax.plot(x, x/2, 'r--', alpha=0.5, linewidth=1.5)
    
    # n = m boundaries: |m-n|/min(m,n) = 0.25
    ax.plot(x, 1.25*x, 'g--', alpha=0.5, linewidth=1)
    ax.plot(x, 0.8*x, 'g--', alpha=0.5, linewidth=1)
    
    # Diagonal m = n
    ax.plot(x, x, 'k:', alpha=0.3, linewidth=1)
    
    ax.set_xlim(0, 105)
    ax.set_ylim(0, 105)
    ax.set_xlabel('n (voters)')
    ax.set_ylabel('m (alternatives)')
    ax.set_title(f"{config['title']}")
    ax.legend(loc='upper left', fontsize=7)
    ax.set_aspect('equal')


def main():
    np.random.seed(42)  # For reproducible jitter
    
    num_datasets = len(DATASETS)
    
    # Plot 1: Individual dataset subplots (black dots with bounds)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, (name, config) in enumerate(DATASETS.items()):
        print(f"Processing {name}...")
        plot_dataset(name, config, axes[idx])
    
    # Hide unused axes
    for idx in range(num_datasets, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    output_path = Path(__file__).parent.parent / 'plots' / 'nm_scatter_all.png'
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved to {output_path}")
    
    # Plot 2: 2x3 Regime-colored plots (one per dataset, capped at 100x100)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, (name, config) in enumerate(DATASETS.items()):
        print(f"Processing {name} (regime colored)...")
        plot_dataset_regime_colored(name, config, axes[idx])
    
    # Hide unused axes
    for idx in range(num_datasets, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    output_path = Path(__file__).parent.parent / 'plots' / 'nm_scatter_regimes.png'
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved to {output_path}")


if __name__ == '__main__':
    main()

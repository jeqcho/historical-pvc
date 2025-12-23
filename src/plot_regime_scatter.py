#!/usr/bin/env python3
"""
Standalone script to plot n vs m scatter with regime classification.
Creates a grid showing each dataset colored by regime (n≪m, n≈m, n≫m).
"""

import os
import glob
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from preflibtools.instances import OrdinalInstance

# Dataset configurations
DATASETS = {
    'spotify': {
        'path': '../data/spotify',
        'patterns': ['*.soc'],
        'title': 'Spotify Countries',
    },
    'polish': {
        'path': '../data/polish',
        'patterns': ['*.soi'],
        'title': 'Polish Elections',
    },
    'eurovision': {
        'path': '../data/eurovision',
        'patterns': ['*.soi'],
        'title': 'Eurovision',
    },
    'ers': {
        'path': '../data/ers',
        'patterns': ['*.soi'],
        'title': 'ERS',
    },
    'skate': {
        'path': '../data/skate',
        'patterns': ['*.soc', '*.toc'],
        'title': 'Figure Skating',
    },
}

# Regime classification colors
REGIME_COLORS = {
    'n ≪ m': '#3498db',  # Blue
    'n ≈ m': '#27ae60',  # Green
    'n ≫ m': '#e74c3c',  # Red
    None: '#95a5a6',     # Gray (unclassified)
}


def classify_regime(n: int, m: int) -> str:
    """
    Classify a point into a regime based on n (voters) and m (alternatives).
    
    Regimes:
    - n ≪ m: m ≥ 2n (alternatives at least 2x voters)
    - n ≫ m: n ≥ 2m (voters at least 2x alternatives)  
    - n ≈ m: |m-n|/min(m,n) < 0.25 (within 25% of each other)
    - None: doesn't fit any regime
    """
    if n == 0 or m == 0:
        return None
    
    min_val = min(n, m)
    
    # n ≪ m: m is at least 2x larger than n
    if 2 * n <= m:
        return 'n ≪ m'
    
    # n ≫ m: n is at least 2x larger than m
    if 2 * m <= n:
        return 'n ≫ m'
    
    # n ≈ m: within 25% tolerance
    if abs(m - n) / min_val < 0.25:
        return 'n ≈ m'
    
    return None


def load_dataset_stats(name: str, config: dict) -> list:
    """Load n, m values for all files in a dataset."""
    base_path = Path(__file__).parent / config['path']
    
    results = []
    for pattern in config['patterns']:
        files = sorted(glob.glob(str(base_path / pattern)))
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


def plot_dataset_regime(name: str, config: dict, ax, jitter_strength: float = 0.5, 
                        max_nm: int = 100):
    """
    Plot a single dataset with points colored by regime classification.
    
    Args:
        name: Dataset name
        config: Dataset configuration
        ax: Matplotlib axis
        jitter_strength: Amount of jitter to add
        max_nm: Maximum n and m values to include
    """
    data = load_dataset_stats(name, config)
    
    if not data:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(config['title'])
        return
    
    # Group by regime
    regime_data = {'n ≪ m': [], 'n ≈ m': [], 'n ≫ m': [], None: []}
    
    for d in data:
        n, m = d['n'], d['m']
        if n <= max_nm and m <= max_nm:
            regime = classify_regime(n, m)
            regime_data[regime].append((n, m))
    
    total_plotted = sum(len(pts) for pts in regime_data.values())
    
    # Plot each regime
    for regime, points in regime_data.items():
        if not points:
            continue
        n_vals = np.array([p[0] for p in points])
        m_vals = np.array([p[1] for p in points])
        
        # Add jitter
        n_jitter = n_vals + np.random.normal(0, jitter_strength, len(n_vals))
        m_jitter = m_vals + np.random.normal(0, jitter_strength, len(m_vals))
        
        label = regime if regime else 'other'
        ax.scatter(n_jitter, m_jitter, c=REGIME_COLORS[regime], alpha=0.6, s=35, 
                   edgecolors='white', linewidth=0.3, label=f'{label} (k={len(points)})')
    
    # Draw regime boundaries
    x = np.linspace(0, max_nm, 100)
    
    # n ≪ m boundary: m = 2n (blue dashed)
    ax.plot(x, 2*x, color='#3498db', linestyle='--', alpha=0.6, linewidth=1.5)
    
    # n ≫ m boundary: m = n/2 (red dashed)
    ax.plot(x, x/2, color='#e74c3c', linestyle='--', alpha=0.6, linewidth=1.5)
    
    # n ≈ m boundaries: ±25% (green dashed)
    ax.plot(x, 1.25*x, color='#27ae60', linestyle='--', alpha=0.5, linewidth=1)
    ax.plot(x, 0.8*x, color='#27ae60', linestyle='--', alpha=0.5, linewidth=1)
    
    # Diagonal m = n (gray dotted)
    ax.plot(x, x, 'k:', alpha=0.3, linewidth=1)
    
    ax.set_xlim(0, max_nm + 5)
    ax.set_ylim(0, max_nm + 5)
    ax.set_xlabel('n (voters)', fontsize=10)
    ax.set_ylabel('m (alternatives)', fontsize=10)
    ax.set_title(f"{config['title']} (k={total_plotted})", fontsize=11, fontweight='bold')
    ax.legend(loc='upper left', fontsize=7, framealpha=0.9)
    ax.set_aspect('equal')


def add_regime_explanation(fig, ax):
    """Add a text box explaining the regime classifications."""
    explanation = (
        "Regime Classification\n"
        "─────────────────────────────────\n\n"
        "n ≪ m (blue):  m ≥ 2n\n"
        "  More alternatives than voters\n"
        "  (e.g., few judges ranking many items)\n\n"
        "n ≈ m (green): |m−n|/min(m,n) < 0.25\n"
        "  Similar number of voters & alternatives\n"
        "  (within 25% tolerance)\n\n"
        "n ≫ m (red):   n ≥ 2m\n"
        "  More voters than alternatives\n"
        "  (e.g., many voters ranking few candidates)\n\n"
        "other (gray):  Does not fit any regime"
    )
    
    ax.text(0.5, 0.5, explanation, transform=ax.transAxes,
            fontsize=10, verticalalignment='center', horizontalalignment='center',
            bbox=dict(boxstyle='round,pad=0.8', facecolor='#f8f9fa', 
                      edgecolor='#dee2e6', linewidth=1.5))
    ax.axis('off')


def main():
    np.random.seed(42)  # For reproducible jitter
    
    # Create 2x3 grid (5 datasets + 1 explanation panel)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    # Plot each dataset
    dataset_names = list(DATASETS.keys())
    for idx, name in enumerate(dataset_names):
        config = DATASETS[name]
        print(f"Processing {name}...")
        plot_dataset_regime(name, config, axes[idx], jitter_strength=0.5, max_nm=100)
    
    # Add explanation in the last panel
    add_regime_explanation(fig, axes[5])
    
    plt.suptitle('Dataset Distribution by Regime (n vs m, capped at 100×100)', 
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    output_path = Path(__file__).parent.parent / 'plots' / 'nm_scatter_regimes.png'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved to {output_path}")


if __name__ == '__main__':
    main()


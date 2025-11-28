#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import json
import sys
from pathlib import Path

def load_results(filename='benchmark_results.json'):
    """Load benchmark results from JSON file"""
    with open(filename, 'r') as f:
        return json.load(f)

def plot_comparison(results, output_dir='plots'):
    """Create comparison plots for all datasets"""
    Path(output_dir).mkdir(exist_ok=True)
    
    for dataset_name, data in results.items():
        configs = [r['name'] for r in data]
        build_times = [r['build_time_ms'] for r in data]
        lookup_times = [r['avg_lookup_ns'] for r in data]
        sizes = [r['size_mb'] for r in data]
        errors = [r['avg_error'] for r in data]
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'{dataset_name} - Performance Comparison', fontsize=16, fontweight='bold')
        
        # Colors: B-Trees in blue, Learned in green/orange
        colors = ['#3498db' if 'B-Tree' in c else '#2ecc71' if 'linear' in c else '#e74c3c' for c in configs]
        
        # Plot 1: Build Time
        ax1 = axes[0, 0]
        bars1 = ax1.bar(range(len(configs)), build_times, color=colors, alpha=0.7, edgecolor='black')
        ax1.set_xlabel('Configuration', fontweight='bold')
        ax1.set_ylabel('Build Time (ms)', fontweight='bold')
        ax1.set_title('Build Time Comparison', fontweight='bold')
        ax1.set_xticks(range(len(configs)))
        ax1.set_xticklabels(configs, rotation=45, ha='right', fontsize=8)
        ax1.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, val in zip(bars1, build_times):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.0f}', ha='center', va='bottom', fontsize=8)
        
        # Plot 2: Lookup Time
        ax2 = axes[0, 1]
        bars2 = ax2.bar(range(len(configs)), lookup_times, color=colors, alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Configuration', fontweight='bold')
        ax2.set_ylabel('Avg Lookup Time (ns)', fontweight='bold')
        ax2.set_title('Lookup Speed Comparison', fontweight='bold')
        ax2.set_xticks(range(len(configs)))
        ax2.set_xticklabels(configs, rotation=45, ha='right', fontsize=8)
        ax2.grid(axis='y', alpha=0.3)
        
        for bar, val in zip(bars2, lookup_times):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.0f}', ha='center', va='bottom', fontsize=8)
        
        # Plot 3: Memory Usage
        ax3 = axes[1, 0]
        bars3 = ax3.bar(range(len(configs)), sizes, color=colors, alpha=0.7, edgecolor='black')
        ax3.set_xlabel('Configuration', fontweight='bold')
        ax3.set_ylabel('Memory Size (MB)', fontweight='bold')
        ax3.set_title('Memory Usage Comparison', fontweight='bold')
        ax3.set_xticks(range(len(configs)))
        ax3.set_xticklabels(configs, rotation=45, ha='right', fontsize=8)
        ax3.grid(axis='y', alpha=0.3)
        
        for bar, val in zip(bars3, sizes):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=8)
        
        # Plot 4: Speedup vs Best B-Tree
        ax4 = axes[1, 1]
        btree_times = [t for c, t in zip(configs, lookup_times) if 'B-Tree' in c]
        best_btree_time = min(btree_times) if btree_times else lookup_times[0]
        speedups = [best_btree_time / t for t in lookup_times]
        
        bars4 = ax4.bar(range(len(configs)), speedups, color=colors, alpha=0.7, edgecolor='black')
        ax4.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='Baseline (Best B-Tree)')
        ax4.set_xlabel('Configuration', fontweight='bold')
        ax4.set_ylabel('Speedup Factor', fontweight='bold')
        ax4.set_title('Speedup vs Best B-Tree', fontweight='bold')
        ax4.set_xticks(range(len(configs)))
        ax4.set_xticklabels(configs, rotation=45, ha='right', fontsize=8)
        ax4.grid(axis='y', alpha=0.3)
        ax4.legend()
        
        for bar, val in zip(bars4, speedups):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.1f}×', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        
        # Save plot
        safe_name = dataset_name.replace(' ', '_').replace('(', '').replace(')', '')
        output_file = f'{output_dir}/{safe_name}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f'✓ Saved plot: {output_file}')
        plt.close()

def plot_summary(results, output_dir='plots'):
    """Create a summary plot comparing all datasets"""
    Path(output_dir).mkdir(exist_ok=True)
    
    # Extract best learned index for each dataset
    dataset_names = []
    learned_speedups = []
    btree_baseline = []
    
    for dataset_name, data in results.items():
        configs = [r['name'] for r in data]
        lookup_times = [r['avg_lookup_ns'] for r in data]
        
        # Get best B-Tree and best learned index
        btree_times = [(c, t) for c, t in zip(configs, lookup_times) if 'B-Tree' in c]
        learned_times = [(c, t) for c, t in zip(configs, lookup_times) if 'Learned' in c]
        
        if btree_times and learned_times:
            best_btree = min(btree_times, key=lambda x: x[1])
            best_learned = min(learned_times, key=lambda x: x[1])
            
            dataset_names.append(dataset_name.split('(')[0].strip())
            btree_baseline.append(best_btree[1])
            learned_speedups.append(best_btree[1] / best_learned[1])
    
    # Create summary plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Overall Performance Summary', fontsize=16, fontweight='bold')
    
    x = np.arange(len(dataset_names))
    width = 0.35
    
    # Plot 1: Absolute lookup times
    bars1_btree = ax1.bar(x - width/2, btree_baseline, width, label='Best B-Tree', 
                          color='#3498db', alpha=0.7, edgecolor='black')
    bars1_learned = ax1.bar(x + width/2, [b/s for b, s in zip(btree_baseline, learned_speedups)], 
                           width, label='Best Learned', color='#2ecc71', alpha=0.7, edgecolor='black')
    
    ax1.set_xlabel('Dataset', fontweight='bold')
    ax1.set_ylabel('Lookup Time (ns)', fontweight='bold')
    ax1.set_title('Best Lookup Times', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(dataset_names, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # Plot 2: Speedup factors
    bars2 = ax2.bar(x, learned_speedups, color='#2ecc71', alpha=0.7, edgecolor='black')
    ax2.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='Baseline')
    ax2.set_xlabel('Dataset', fontweight='bold')
    ax2.set_ylabel('Speedup Factor', fontweight='bold')
    ax2.set_title('Learned Index Speedup', fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(dataset_names, rotation=45, ha='right')
    ax2.grid(axis='y', alpha=0.3)
    ax2.legend()
    
    for bar, val in zip(bars2, learned_speedups):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}×', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    output_file = f'{output_dir}/summary.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f'✓ Saved summary plot: {output_file}')
    plt.close()

def main():
    input_file = 'benchmark_results.json' if len(sys.argv) < 2 else sys.argv[1]
    output_dir = 'plots' if len(sys.argv) < 3 else sys.argv[2]
    
    print(f'Loading results from {input_file}...')
    results = load_results(input_file)
    
    print(f'Generating plots in {output_dir}/...')
    plot_comparison(results, output_dir)
    plot_summary(results, output_dir)
    
    print(f'\n✓ All plots generated successfully!')
    print(f'  View plots in: {output_dir}/')

if __name__ == '__main__':
    main()

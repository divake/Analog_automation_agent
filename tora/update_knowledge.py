import pandas as pd
import json
import numpy as np
import os
import matplotlib.pyplot as plt
from pathlib import Path

def analyze_dataset(csv_path):
    """
    Analyze the circuit dataset to extract parameter ranges, correlations,
    and identify parameter combinations that achieve target specifications.
    """
    print(f"Loading dataset from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"Dataset loaded with {len(df)} samples.")
    
    # Calculate statistics for each parameter
    param_stats = {}
    for param in ['VDD', 'Vgate', 'Wn', 'Rd']:
        param_stats[param] = {
            'min': float(df[param].min()),
            'max': float(df[param].max()),
            'median': float(df[param].median()),
            'p05': float(df[param].quantile(0.05)),
            'p95': float(df[param].quantile(0.95))
        }
    
    # Calculate stats for performance metrics
    perf_stats = {}
    for metric in ['Bandwidth', 'PowerConsumption', 'VoltageGain']:
        perf_stats[metric] = {
            'min': float(df[metric].min()),
            'max': float(df[metric].max()),
            'median': float(df[metric].median()),
            'p05': float(df[metric].quantile(0.05)),
            'p95': float(df[metric].quantile(0.95))
        }
    
    # Find correlations between parameters and metrics
    correlations = {}
    for param in ['VDD', 'Vgate', 'Wn', 'Rd']:
        correlations[param] = {}
        for metric in ['Bandwidth', 'PowerConsumption', 'VoltageGain']:
            correlations[param][metric] = float(df[param].corr(df[metric]))
    
    # Find combinations close to target specs
    # Example target: gain≈4.8, bandwidth≈577MHz, power≈0.00279W
    target_specs = {
        'VoltageGain': 4.8,
        'Bandwidth': 577000000.0,
        'PowerConsumption': 0.00279
    }
    
    # Calculate normalized distance from target for each row
    # Using weighted normalization to prioritize meeting all specs equally
    df['gain_diff'] = abs(df['VoltageGain'] - target_specs['VoltageGain']) / max(abs(df['VoltageGain'] - target_specs['VoltageGain']))
    df['bw_diff'] = abs(df['Bandwidth'] - target_specs['Bandwidth']) / target_specs['Bandwidth']
    df['power_diff'] = abs(df['PowerConsumption'] - target_specs['PowerConsumption']) / target_specs['PowerConsumption']
    df['total_diff'] = df['gain_diff'] + df['bw_diff'] + df['power_diff']
    
    # Get top 10 closest matches
    best_matches = df.sort_values('total_diff').head(10)
    
    # Create parameter sensitivity analysis
    sensitivity = {}
    for param in ['VDD', 'Vgate', 'Wn', 'Rd']:
        sensitivity[param] = {}
        for metric in ['Bandwidth', 'PowerConsumption', 'VoltageGain']:
            # Calculate typical change in metric for a small change in parameter
            param_range = param_stats[param]['max'] - param_stats[param]['min']
            metric_range = perf_stats[metric]['max'] - perf_stats[metric]['min']
            param_step = param_range / 20  # 5% of parameter range
            
            # Estimated change in metric for this parameter step
            sensitivity[param][metric] = {
                'param_step': param_step,
                'metric_change': correlations[param][metric] * (metric_range / param_range) * param_step
            }
    
    return {
        'param_stats': param_stats,
        'perf_stats': perf_stats,
        'correlations': correlations,
        'best_matches': best_matches.to_dict('records'),
        'sensitivity': sensitivity
    }

def generate_knowledge_json(analysis, input_path, output_path):
    """
    Update the knowledge.json file with insights from dataset analysis.
    """
    print(f"Updating knowledge file at {output_path}...")
    
    # Load existing knowledge.json
    with open(input_path, 'r') as f:
        knowledge = json.load(f)
    
    # Update parameter ranges based on dataset analysis
    for param, details in knowledge['parameters'].items():
        if param in analysis['param_stats']:
            details['typical_range'] = [
                analysis['param_stats'][param]['p05'],
                analysis['param_stats'][param]['p95']
            ]
            
            # Add empirical observations for each parameter
            if 'empirical_observations' not in details:
                details['empirical_observations'] = []
                
            observations = []
            for metric in ['Bandwidth', 'PowerConsumption', 'VoltageGain']:
                corr = analysis['correlations'][param][metric]
                if abs(corr) > 0.3:  # Only include significant correlations
                    direction = "increases" if corr > 0 else "decreases"
                    strength = "strongly" if abs(corr) > 0.7 else "moderately"
                    observations.append(f"{metric} {direction} {strength} as {param} increases")
            
            details['empirical_observations'] = observations
    
    # Update performance metric ranges
    for metric, details in knowledge['performance_metrics'].items():
        if metric in analysis['perf_stats']:
            details['typical_range'] = [
                analysis['perf_stats'][metric]['p05'],
                analysis['perf_stats'][metric]['p95']
            ]
    
    # Add design strategies for target specifications
    avg_vdd = sum(match['VDD'] for match in analysis['best_matches'][:3]) / 3
    avg_vgate = sum(match['Vgate'] for match in analysis['best_matches'][:3]) / 3
    avg_wn = sum(match['Wn'] for match in analysis['best_matches'][:3]) / 3
    avg_rd = sum(match['Rd'] for match in analysis['best_matches'][:3]) / 3
    
    target_strategy = [
        f"Set VDD close to {avg_vdd:.2f}V (range: {analysis['best_matches'][0]['VDD']:.2f}V - {analysis['best_matches'][2]['VDD']:.2f}V)",
        f"Set Vgate close to {avg_vgate:.2f}V (range: {analysis['best_matches'][0]['Vgate']:.2f}V - {analysis['best_matches'][2]['Vgate']:.2f}V)",
        f"Set Wn close to {avg_wn:.2e} (range: {analysis['best_matches'][0]['Wn']:.2e} - {analysis['best_matches'][2]['Wn']:.2e})",
        f"Set Rd close to {avg_rd:.0f}Ω (range: {analysis['best_matches'][0]['Rd']:.0f}Ω - {analysis['best_matches'][2]['Rd']:.0f}Ω)"
    ]
    knowledge['design_strategies']['target_specs'] = target_strategy
    
    # Write updated knowledge.json
    with open(output_path, 'w') as f:
        json.dump(knowledge, f, indent=2)
    
    print(f"Knowledge file updated successfully!")
    return knowledge

def update_prompt_templates(analysis, templates_dir):
    """
    Update the prompt templates with dataset-specific guidance.
    """
    print(f"Updating prompt templates in {templates_dir}...")
    
    # Update explanation template
    explanation_template_path = os.path.join(templates_dir, 'explanation.txt')
    with open(explanation_template_path, 'r') as f:
        explanation_template = f.read()
    
    # Update refinement template
    refinement_template_path = os.path.join(templates_dir, 'refinement.txt')
    with open(refinement_template_path, 'r') as f:
        refinement_template = f.read()
    
    # Add dataset-specific guidance to explanation template
    explanation_addendum = f"""
Your explanation should consider these empirical insights from the dataset:
1. Parameter ranges: 
   - VDD: {analysis['param_stats']['VDD']['p05']:.2f}V - {analysis['param_stats']['VDD']['p95']:.2f}V
   - Vgate: {analysis['param_stats']['Vgate']['p05']:.2f}V - {analysis['param_stats']['Vgate']['p95']:.2f}V
   - Wn: {analysis['param_stats']['Wn']['p05']:.2e} - {analysis['param_stats']['Wn']['p95']:.2e}
   - Rd: {analysis['param_stats']['Rd']['p05']:.0f}Ω - {analysis['param_stats']['Rd']['p95']:.0f}Ω

2. Key correlations:
   - VDD most strongly affects: PowerConsumption (correlation: {analysis['correlations']['VDD']['PowerConsumption']:.2f})
   - Vgate most strongly affects: PowerConsumption (correlation: {analysis['correlations']['Vgate']['PowerConsumption']:.2f})
   - Wn most strongly affects: Bandwidth (correlation: {analysis['correlations']['Wn']['Bandwidth']:.2f})
   - Rd most strongly affects: VoltageGain (correlation: {analysis['correlations']['Rd']['VoltageGain']:.2f})

3. For target specifications (gain≈4.8, bandwidth≈577MHz, power≈0.00279W):
   Top 3 parameter combinations from dataset:
   - VDD={analysis['best_matches'][0]['VDD']:.2f}V, Vgate={analysis['best_matches'][0]['Vgate']:.2f}V, Wn={analysis['best_matches'][0]['Wn']:.2e}, Rd={analysis['best_matches'][0]['Rd']:.0f}Ω
   - VDD={analysis['best_matches'][1]['VDD']:.2f}V, Vgate={analysis['best_matches'][1]['Vgate']:.2f}V, Wn={analysis['best_matches'][1]['Wn']:.2e}, Rd={analysis['best_matches'][1]['Rd']:.0f}Ω
   - VDD={analysis['best_matches'][2]['VDD']:.2f}V, Vgate={analysis['best_matches'][2]['Vgate']:.2f}V, Wn={analysis['best_matches'][2]['Wn']:.2e}, Rd={analysis['best_matches'][2]['Rd']:.0f}Ω
"""
    updated_explanation = explanation_template + explanation_addendum
    
    # Add dataset-specific guidance to refinement template
    refinement_addendum = f"""
Your parameter adjustments should consider these empirical insights from the dataset:

1. Parameter sensitivity (typical impact):
   - VDD: A change of {analysis['sensitivity']['VDD']['VoltageGain']['param_step']:.2f}V typically changes gain by {abs(analysis['sensitivity']['VDD']['VoltageGain']['metric_change']):.2f}
   - Vgate: A change of {analysis['sensitivity']['Vgate']['Bandwidth']['param_step']:.2f}V typically changes bandwidth by {abs(analysis['sensitivity']['Vgate']['Bandwidth']['metric_change']/1e6):.2f}MHz
   - Wn: A change of {analysis['sensitivity']['Wn']['PowerConsumption']['param_step']:.2e} typically changes power by {abs(analysis['sensitivity']['Wn']['PowerConsumption']['metric_change']):.5f}W
   - Rd: A change of {analysis['sensitivity']['Rd']['VoltageGain']['param_step']:.0f}Ω typically changes gain by {abs(analysis['sensitivity']['Rd']['VoltageGain']['metric_change']):.2f}

2. For target specifications (gain≈4.8, bandwidth≈577MHz, power≈0.00279W):
   Parameter combinations from dataset that achieve similar specifications:
   - VDD={analysis['best_matches'][0]['VDD']:.2f}V, Vgate={analysis['best_matches'][0]['Vgate']:.2f}V, Wn={analysis['best_matches'][0]['Wn']:.2e}, Rd={analysis['best_matches'][0]['Rd']:.0f}Ω
     Results: Gain={analysis['best_matches'][0]['VoltageGain']:.2f}, BW={analysis['best_matches'][0]['Bandwidth']/1e6:.2f}MHz, Power={analysis['best_matches'][0]['PowerConsumption']:.6f}W
     
   - VDD={analysis['best_matches'][1]['VDD']:.2f}V, Vgate={analysis['best_matches'][1]['Vgate']:.2f}V, Wn={analysis['best_matches'][1]['Wn']:.2e}, Rd={analysis['best_matches'][1]['Rd']:.0f}Ω
     Results: Gain={analysis['best_matches'][1]['VoltageGain']:.2f}, BW={analysis['best_matches'][1]['Bandwidth']/1e6:.2f}MHz, Power={analysis['best_matches'][1]['PowerConsumption']:.6f}W

3. Suggested parameter adjustment strategies:
   - To increase gain: Increase Rd (strongest positive correlation: {analysis['correlations']['Rd']['VoltageGain']:.2f})
   - To increase bandwidth: Decrease Rd (correlation: {analysis['correlations']['Rd']['Bandwidth']:.2f}) or increase Wn (correlation: {analysis['correlations']['Wn']['Bandwidth']:.2f})
   - To decrease power: Decrease VDD (correlation: {analysis['correlations']['VDD']['PowerConsumption']:.2f}) or decrease Wn (correlation: {analysis['correlations']['Wn']['PowerConsumption']:.2f})
"""
    updated_refinement = refinement_template + refinement_addendum
    
    # Write updated templates
    explanation_backup = explanation_template_path + '.bak'
    refinement_backup = refinement_template_path + '.bak'
    
    # Create backups first
    with open(explanation_backup, 'w') as f:
        f.write(explanation_template)
    with open(refinement_backup, 'w') as f:
        f.write(refinement_template)
    
    # Write updated templates
    with open(explanation_template_path, 'w') as f:
        f.write(updated_explanation)
    
    with open(refinement_template_path, 'w') as f:
        f.write(updated_refinement)
    
    print(f"Prompt templates updated successfully! Backups created at {explanation_backup} and {refinement_backup}")
    return {
        'explanation': updated_explanation,
        'refinement': updated_refinement
    }

def generate_visualizations(analysis, output_dir):
    """
    Generate visualizations of the dataset analysis to help understand parameter relationships.
    """
    print(f"Generating visualizations in {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot parameter correlations with performance metrics
    plt.figure(figsize=(12, 8))
    params = ['VDD', 'Vgate', 'Wn', 'Rd']
    metrics = ['Bandwidth', 'PowerConsumption', 'VoltageGain']
    corr_data = []
    
    for param in params:
        row = []
        for metric in metrics:
            row.append(analysis['correlations'][param][metric])
        corr_data.append(row)
    
    plt.imshow(corr_data, cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar(label='Correlation Coefficient')
    plt.xticks(range(len(metrics)), metrics, rotation=45)
    plt.yticks(range(len(params)), params)
    plt.title('Parameter-Metric Correlations')
    
    for i in range(len(params)):
        for j in range(len(metrics)):
            plt.text(j, i, f"{corr_data[i][j]:.2f}", ha='center', va='center', 
                     color='white' if abs(corr_data[i][j]) > 0.5 else 'black')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'parameter_correlations.png'))
    
    # Plot parameter ranges
    plt.figure(figsize=(10, 6))
    for i, param in enumerate(params):
        plt.subplot(2, 2, i+1)
        p05 = analysis['param_stats'][param]['p05']
        p95 = analysis['param_stats'][param]['p95']
        median = analysis['param_stats'][param]['median']
        
        plt.axvspan(p05, p95, alpha=0.3, color='blue')
        plt.axvline(median, color='red', linestyle='--')
        
        if param == 'Wn':
            # Format scientific notation for Wn
            plt.text(p05, 0.5, f"{p05:.2e}", verticalalignment='center')
            plt.text(p95, 0.5, f"{p95:.2e}", verticalalignment='center')
            plt.text(median, 0.5, f"{median:.2e}", verticalalignment='center', 
                     horizontalalignment='right', color='red')
        else:
            plt.text(p05, 0.5, f"{p05:.2f}", verticalalignment='center')
            plt.text(p95, 0.5, f"{p95:.2f}", verticalalignment='center')
            plt.text(median, 0.5, f"{median:.2f}", verticalalignment='center', 
                     horizontalalignment='right', color='red')
        
        plt.yticks([])
        plt.title(f"{param} Range (5th-95th percentile)")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'parameter_ranges.png'))
    
    print(f"Visualizations generated successfully!")

def main():
    # Define paths relative to the tora directory
    base_dir = Path('/ssd_4TB/divake/AICircuit')
    tora_dir = base_dir / 'tora'
    csv_path = base_dir / 'Dataset/CSVA/CSVA.csv'
    knowledge_input_path = tora_dir / 'CSVA/knowledge.json'
    knowledge_output_path = tora_dir / 'CSVA/knowledge.json'
    templates_dir = tora_dir / 'prompt_templates'
    vis_output_dir = tora_dir / 'knowledge_analysis'
    
    # Ensure all required files exist
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {csv_path}")
    if not knowledge_input_path.exists():
        raise FileNotFoundError(f"Knowledge file not found: {knowledge_input_path}")
    if not templates_dir.exists():
        raise FileNotFoundError(f"Templates directory not found: {templates_dir}")
    
    # Analyze dataset
    print(f"Starting dataset analysis...")
    analysis = analyze_dataset(csv_path)
    
    # Generate updated files
    knowledge = generate_knowledge_json(analysis, knowledge_input_path, knowledge_output_path)
    templates = update_prompt_templates(analysis, templates_dir)
    
    # Generate visualizations
    generate_visualizations(analysis, vis_output_dir)
    
    print("\nKnowledge update complete!")
    print(f"Files updated:")
    print(f"  - Knowledge file: {knowledge_output_path}")
    print(f"  - Explanation template: {templates_dir / 'explanation.txt'}")
    print(f"  - Refinement template: {templates_dir / 'refinement.txt'}")
    print(f"  - Visualizations: {vis_output_dir}")
    print("\nThese updates should improve the performance of your circuit optimization process.")
    print("Run your TORA pipeline as usual to utilize the updated knowledge.")

if __name__ == "__main__":
    main() 
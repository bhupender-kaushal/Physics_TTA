import os
import re
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import defaultdict

def parse_dice_file(filepath):
    """Parses the perturbation_dices.txt file into a dictionary of lists."""
    results = defaultdict(list)
    current_aug = None
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith("---") and "VOL DICES" in line:
                # Extract clean augmentation name
                aug_raw = line.replace("---", "").replace("VOL DICES", "").strip()
                current_aug = aug_raw.replace("TTA:", "").strip()
            elif line.startswith("Pass") and current_aug is not None:
                match = re.search(r"Mean Slice Dice = ([\d\.]+)", line)
                if match:
                    results[current_aug].append(float(match.group(1)))
    return results

def main():
    # --- CONFIGURATION ---
    # Update this to the parent directory containing your 'baseline' and 'dropout_X' folders
    results_dir = '/bhupender/Physics_Uncert/MRI_CT/nnUNet/uncert_analysis/mc_results/improvements'
    
    # We will aggregate all data here
    data = []
    
    # --- DATA GATHERING ---
    if not os.path.exists(results_dir):
        print(f"Directory not found: {results_dir}. Please update the path.")
        return

    for model_name in os.listdir(results_dir):
        print(f"Processing model folder: {model_name}")
        model_path = os.path.join(results_dir, model_name)
        if not os.path.isdir(model_path): continue
        
        # Locate the dice txt file in the uncertainties folder
        # uncert_dir = os.path.join(model_path, 'uncertainties')

        uncert_dir = model_path
        if not os.path.exists(uncert_dir): continue
            
        txt_files = [f for f in os.listdir(uncert_dir) if f.endswith('perturbation_dices.txt')]
        if not txt_files: continue
            
        dice_file = os.path.join(uncert_dir, txt_files[0])
        parsed_data = parse_dice_file(dice_file)
        
        for aug, dices in parsed_data.items():
            for pass_idx, dice in enumerate(dices):
                data.append({
                    'Model': 'Baseline' if 'baseline' in model_name.lower() else model_name,
                    'Augmentation': aug,
                    'Pass': pass_idx,
                    'Dice': dice
                })

    df = pd.DataFrame(data)
    if df.empty:
        print("No data found to plot!")
        return

    # --- PLOTTING ---
    plt.figure(figsize=(18, 8))
    sns.set_theme(style="whitegrid")

    # Create grouped barplot to show ONLY the mean across passes (errorbar=None hides the variance)
    ax = sns.barplot(
        data=df, x='Augmentation', y='Dice', hue='Model', 
        palette='Set2', errorbar=None
    )
    plt.title("Impact of MC Dropout on Model Robustness Under Physics Perturbations", fontsize=18, weight='bold')
    plt.ylabel("Mean Slice Dice Score (Averaged Across Passes)", fontsize=14)
    plt.xlabel("Test-Time Augmentation (Perturbation)", fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.legend(title='Model Variant', bbox_to_anchor=(1.01, 1), loc='upper left')

    # Compute and display differences (Increment/Decrement) inside the bars
    augs = df['Augmentation'].unique()
    models = [t.get_text() for t in ax.get_legend().texts] if ax.get_legend() else df['Model'].unique()
    baseline_col = [m for m in models if 'Baseline' in m]

    if baseline_col and len(models) >= 2:
        baseline_name = baseline_col[0]
        # Get standard offset for boxes in seaborn
        num_models = len(models)
        
        for i, aug in enumerate(augs):
            base_data = df[(df['Augmentation'] == aug) & (df['Model'] == baseline_name)]
            if base_data.empty: continue
            base_mean = base_data['Dice'].mean()
            
            for j, model in enumerate(models):
                drop_data = df[(df['Augmentation'] == aug) & (df['Model'] == model)]
                if drop_data.empty: continue
                drop_mean = drop_data['Dice'].mean()
                
                # Calculate x-offset based on hue grouping
                x_pos = i - 0.4 + (j + 0.5) * (0.8 / num_models)
                # Y position inside the bar (subtracted to sit below the top edge)
                y_pos = drop_mean - 0.02 
                
                if model == baseline_name:
                    # Print absolute baseline dice
                    ax.text(
                        x_pos, y_pos, f"Base: {base_mean:.3f}",
                        color='black', ha='center', va='top',
                        fontsize=9, weight='bold', rotation=90, fontfamily='monospace'
                    )
                else:
                    # Print increment/decrement
                    diff = drop_mean - base_mean
                    color = 'darkgreen' if diff > 0 else 'darkred'
                    sign = '+' if diff > 0 else ''
                    
                    if not pd.isna(diff):
                        ax.text(
                            x_pos, y_pos, f"{sign}{diff:.3f}",
                            color=color, ha='center', va='top',
                            fontsize=9, weight='bold', rotation=90, fontfamily='monospace'
                        )

    plt.ylim(0.5, 1)  # Assuming Dice scores are between 0 and 1
    
    plt.tight_layout()
    output_path = os.path.join(results_dir, "dropout_improvement_plot.png")
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"\nPlot successfully saved to: {output_path}")
    plt.show()

if __name__ == "__main__":
    main()
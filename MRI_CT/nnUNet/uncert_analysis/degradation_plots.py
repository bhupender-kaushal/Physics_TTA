import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    # Update this path if plotting a different case or the combined directory
    csv_path = '/bhupender/Physics_Uncert/MRI_CT/nnUNet/uncert_analysis/mc_results/degradation_comparison/prostate_10/prostate_10_degradation_log.csv'
    
    if not os.path.exists(csv_path):
        print(f"CSV file not found: {csv_path}")
        return
        
    # Read the data
    df = pd.read_csv(csv_path)
    
    # Restructure from Wide to Long format for Seaborn plotting
    df_melted = df.melt(
        id_vars=['Case', 'Augmentation', 'Intensity'], 
        value_vars=['Baseline_Dice', 'Pretrained_Dice'],
        var_name='Model', 
        value_name='Dice'
    )
    
    # Clean up labels for a prettier plot
    df_melted['Model'] = df_melted['Model'].str.replace('_Dice', '')
    df_melted['Augmentation'] = df_melted['Augmentation'].str.replace('_', ' ').str.title()
    
    # --- PLOTTING ---
    sns.set_theme(style="whitegrid")
    
    # Create a FacetGrid with 1 subplot per augmentation type
    g = sns.FacetGrid(
        df_melted, col="Augmentation", hue="Model", 
        height=5, aspect=1.2, palette="Set1", sharey=False
    )
    
    # Map the lineplot to the grid
    g.map(sns.lineplot, "Intensity", "Dice", marker="o", linewidth=2.5, markersize=8)
    
    # Format the grid wrappers
    g.set_axis_labels("Perturbation Intensity", "Mean Dice Score")
    g.set_titles(col_template="{col_name}", size=14, weight='bold')
    g.add_legend(title="Model", title_fontsize='12', fontsize='11')
    
    # Add explicit data values above the points & adjust axes
    for ax in g.axes.flat:
        ax.set_ylim(bottom=0.0, top=1.05)
        ax.set_xticks(sorted(df['Intensity'].unique()))
        
        # Annotate points
        for line in ax.lines:
            for x, y in zip(line.get_xdata(), line.get_ydata()):
                ax.text(x, y + 0.02, f'{y:.2f}', color=line.get_color(), 
                        fontsize=9, ha='center', va='bottom', weight='bold')

    # Add a main title
    case_name = df['Case'].iloc[0] if len(df['Case'].unique()) == 1 else "All Cases"
    plt.subplots_adjust(top=0.85)
    g.fig.suptitle(f"Dice Degradation Under Increasing Perturbations ({case_name})", fontsize=18, weight='bold')
    
    # Save the output next to the CSV
    output_dir = os.path.dirname(csv_path)
    output_file = os.path.join(output_dir, f"{case_name}_degradation_plot.png")
    
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    print(f"Plot successfully saved to: {output_file}")
    plt.show()

if __name__ == '__main__':
    main()
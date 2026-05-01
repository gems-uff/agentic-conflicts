import pandas as pd
import matplotlib.pyplot as plt
from analysis_utils import load_classified_chunks, save_plot, save_table

def plot_stacked_bar(df, index_col, title, filename, directory="results/rq2"):
    # Create crosstab
    ct = pd.crosstab(df[index_col], df['strategy'], normalize='index') * 100
    
    # Sort for better visualization (e.g. by index count or alphabetically)
    if index_col != 'strata':
        # only keep top categories if there are too many
        top_cats = df[index_col].value_counts().nlargest(10).index
        ct = ct.loc[top_cats]
        # Re-normalize just in case
        ct = ct.div(ct.sum(axis=1), axis=0) * 100

    ax = ct.plot(kind='barh', stacked=True, figsize=(10, 6), colormap='Set2')
    plt.title(title)
    plt.xlabel('Percentage (%)')
    plt.ylabel(index_col.capitalize())
    
    # Place legend outside
    plt.legend(title='Strategy', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    save_plot(plt.gcf(), filename, directory)
    plt.close()

def generate_rq2_stats(df, suffix=""):
    print(f"Computing global strategy frequencies{suffix}...")
    global_freq = df['strategy'].value_counts(normalize=True).reset_index()
    global_freq.columns = ['strategy', 'percentage']
    global_freq['percentage'] *= 100
    save_table(global_freq, f"strategy_frequencies_global{suffix}", "results/rq2")

    for strata in ['agent', 'language', 'pr_task_type', 'resolver_type']:
        print(f"Computing strategies by {strata}{suffix}...")
        
        # Crosstab to get table
        ct = pd.crosstab(df[strata], df['strategy'], normalize='index') * 100
        ct_flat = ct.reset_index()
        save_table(ct_flat, f"strategy_frequencies_by_{strata}{suffix}", "results/rq2")
        
        # Plot stacked bar chart
        plot_stacked_bar(df, strata, f'Resolution Strategies by {strata.capitalize()}{" (Precise Only)" if suffix else ""}', f'strategies_stacked_{strata}{suffix}')

def main():
    print("Loading chunks for RQ2...")
    df = load_classified_chunks()

    if df.empty:
        print("No classified chunks found.")
        return

    # Ensure resolver_type has no NaNs
    df['resolver_type'] = df['resolver_type'].fillna('Unknown').astype(str)

    generate_rq2_stats(df, "")

    print("Filtering out Imprecise chunks...")
    df_precise = df[df['strategy'] != 'Imprecise']
    generate_rq2_stats(df_precise, "_precise")

    print("RQ2 analysis complete.")

if __name__ == "__main__":
    main()

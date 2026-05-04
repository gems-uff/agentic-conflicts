import pandas as pd
import matplotlib.pyplot as plt
from analysis_utils import load_resolver_labels, save_plot, save_table

def plot_stacked_bar(df, index_col, title, filename, directory="results/rq3"):
    ct = pd.crosstab(df[index_col], df['resolver_type'], normalize='index') * 100
    
    if index_col != 'strata':
        top_cats = df[index_col].value_counts().nlargest(10).index
        ct = ct.loc[top_cats]
        ct = ct.div(ct.sum(axis=1), axis=0) * 100

    ax = ct.plot(kind='barh', stacked=True, figsize=(10, 6), color=['#1f77b4', '#ff7f0e'])
    plt.title(title)
    plt.xlabel('Percentage (%)')
    plt.ylabel(index_col.capitalize())
    
    plt.legend(title='Resolver Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    save_plot(plt.gcf(), filename, directory)
    plt.close()

def main():
    print("Loading resolver labels for RQ3...")
    df = load_resolver_labels()

    if df.empty:
        print("No resolver labels found.")
        return

    print("Computing global resolver frequencies...")
    # Resolver is at the merge-commit level
    global_freq = df['resolver_type'].value_counts(normalize=True).reset_index()
    global_freq.columns = ['resolver_type', 'percentage']
    global_freq['percentage'] *= 100
    save_table(global_freq, "resolver_frequencies_global", "results/rq3")

    # For stratifications, compute relative percentage
    for strata in ['agent', 'language', 'pr_task_type']:
        print(f"Computing resolver distribution by {strata}...")
        
        ct = pd.crosstab(df[strata], df['resolver_type'], normalize='index') * 100
        ct_flat = ct.rename_axis('strata_value').reset_index()
        save_table(ct_flat, f"resolver_frequencies_by_{strata}", "results/rq3")
        
        plot_stacked_bar(df, strata, f'Resolver Identity by {strata.capitalize()}', f'resolver_stacked_{strata}')

    print("RQ3 analysis complete.")

if __name__ == "__main__":
    main()

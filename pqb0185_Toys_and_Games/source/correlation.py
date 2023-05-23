import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def main():
    features_training = pd.read_json("features/features_training.json")

    cols = ['avg_compound_text', 'avg_compound_summ', 'std_text', 'std_summ', 'pct_verif', 'amt_reviews',
                    'amt_stars', 'awesomeness']
    X = features_training[cols]

    # Correlation Heatmap
    correlations = X.corr()
    plt.figure()
    plt.title("Heatmap of Correlations (adapted features)")
    _ = sns.heatmap(correlations, annot = True, vmin=-1, vmax=1)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
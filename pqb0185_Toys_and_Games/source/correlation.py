import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA


def main():
    features_training = pd.read_json("features/features_training.json")

    cols = ['avg_compound_text', 'avg_compound_summ', 'pos_text', 'avg_pos_text', 'pos_summ', 'avg_pos_summ', 'neg_text', 'avg_neg_text',
            'neg_summ', 'avg_neg_summ', 'std_text', 'std_summ', 'pct_verif', 'amt_reviews', 'amt_stars', 'starsabove4', 'product_age', 'awesomeness']
    X = features_training[cols]

    # Correlation Heatmap
    correlations = X.corr()
    matrix = np.triu(correlations)
    np.fill_diagonal(matrix, False)
    plt.figure()
    plt.title("Heatmap of Correlations (adapted features)")
    _ = sns.heatmap(correlations, annot = True, vmin=-1, vmax=1, mask=matrix)
    plt.tight_layout()
    plt.savefig('heatmap.png')
    plt.show()


if __name__ == '__main__':
    main()
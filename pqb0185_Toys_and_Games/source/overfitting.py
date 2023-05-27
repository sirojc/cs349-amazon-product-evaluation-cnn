import pandas as pd
import numpy as np

from sklearn.metrics import f1_score

def main():
    print("Loading data")
    predictions_test2 = pd.read_json("predictions_test2.json")
    ground_truth_test2 = pd.read_json("Toys_and_Games/test2/product_test.json")

    ground_truth_test2.sort_index(axis=0, inplace=True)
    print(predictions_test2.head(10))
    print(ground_truth_test2.head(10))

    # TODO: f1_score once files are in same order





if __name__ == "__main__":
    main()
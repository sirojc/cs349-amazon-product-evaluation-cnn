import pandas as pd
import numpy as np

from sklearn.metrics import f1_score

def main():
    print("Loading data")
    predictions_test2 = pd.read_json("predictions_test2.json")
    ground_truth_test2 = pd.read_json("ground_truth/Toys_and_Games_test2_labels.json")
    predictions_test1 = pd.read_json("predictions_test1.json")
    ground_truth_test1 = pd.read_json("ground_truth/Toys_and_Games_test1_labels.json")

    ground_truth_test2.sort_values(by='asin', axis=0, inplace=True)
    predictions_test2.sort_values(by='asin', axis=0, inplace=True)
    # print("test2 asin match: " + str(predictions_test2['asin'].equals(ground_truth_test2['asin'])))
    ground_truth_test1.sort_values(by='asin', axis=0, inplace=True)
    predictions_test1.sort_values(by='asin', axis=0, inplace=True)
    # print("test1 asin match: " + str(predictions_test1['asin'].equals(ground_truth_test1['asin'])))

    # print(ground_truth_test1['asin'].head(10))
    # print(predictions_test1['asin'].head(10))

    # TODO: f1_score once files are in same order
    test2_f1 = f1_score(ground_truth_test2['awesomeness'], predictions_test2['awesomeness'])
    test1_f1 = f1_score(ground_truth_test1['awesomeness'], predictions_test1['awesomeness'])

    print("test2 f1: " + str(test2_f1))
    print("test1 f1: " + str(test1_f1))




if __name__ == "__main__":
    main()
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

from sklearn.metrics import f1_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA

from sklearn_genetic import GASearchCV
from sklearn_genetic.space import Integer, Categorical, Continuous

import seaborn as sns

### Helper Functions ###

def generate_bar_plot(models, model_names, scoring_method, scores_avg, w=0.2, bottom=0.6, top=1):
    x = []
    for k in range(len(models)):
        x.append(model_names[k].replace("model_", ""))

    x_axis = np.arange(len(x))
    offset = -(((len(scoring_method) - 1) * w) / 2)

    method_scores = {}

    for method in scoring_method:
        method_scores[method] = []

    for model in scores_avg:
        for method in scoring_method:
            method_scores[method].append(scores_avg[model][method])

    count = 0
    colors = ['lightseagreen', 'mediumorchid', 'steelblue']
    for method in scoring_method:
        plt.bar(x_axis + offset + (count * w), method_scores[method], w, color=colors[count], label=method)
        count += 1
    plt.ylim(bottom, top)
    plt.xticks(x_axis, x)
    plt.xlabel("Models")
    plt.ylabel("Performance")
    plt.title("Scores for each model")
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig("./plots/bar_plot.png")
    # plt.show()

### Hyperparameter Optimization ###

def grid_search(X, Y):
    print("Performing Grid Search")
    # model_ab = AdaBoostClassifier(RandomForestClassifier(max_depth=8, n_estimators=100, n_jobs=-1, criterion="log_loss", class_weight=None))
    model_nn = MLPClassifier(max_iter=2000)

    # ab_params = {'n_estimators': [13],
    #             'estimator__max_depth': [5],
    #             'learning_rate': [2.0725, 2.075, 2.0775]
    #             }
    nn_params = {'hidden_layer_sizes': [(100, 100, 100, 100), (100, 100, 100, 100, 100)],
                'activation': ['relu', 'tanh', 'logistic'],
                'solver': ['adam', 'sgd'],
                'learning_rate': ['constant', 'adaptive'],
                'learning_rate_init': [0.01, 0.025, 0.05]
                }

    # ab_grid = GridSearchCV(model_ab, ab_params, cv=10, scoring="f1", verbose=2, n_jobs=-1)
    nn_grid = GridSearchCV(model_nn, nn_params, cv=10, scoring="f1", verbose=2, n_jobs=-1)
    
    #ab_grid.fit(X, Y)
    nn_grid.fit(X, Y)

    # ab_best_params = ab_grid.best_params_
    nn_best_params = nn_grid.best_params_

    # print("AdaBoost best parameters: ", ab_best_params)
    print("Neural Network best parameters: ", nn_best_params)

    return nn_best_params

def evolutionary_search(X, Y):
    print("Performing Evolutionary Search")
    model_nn = MLPClassifier(max_iter=2000)

    nn_params = {'tol': Continuous(1e-4, 1e-1, distribution='log-uniform'),
            'hidden_layer_sizes': Categorical([ (100, 100), (10, 10, 10), (100, 100, 100), (10, 10, 10, 10), (100, 100, 100, 100)]),
            'alpha': Continuous(1e-5, 3e-5),
            'activation': Categorical(['relu', 'tanh', 'logistic']),
            'solver': Categorical(['adam', 'sgd']),
            'learning_rate': Categorical(['constant', 'adaptive']),
            'learning_rate_init': Continuous(1e-3, 1e-1),
            'shuffle': Categorical([True, False])
            #batch_size, beta_1, beta_2,...
            }
    
    nn_grid = GASearchCV(estimator=model_nn, param_grid=nn_params, cv=10, scoring="f1", verbose=True, n_jobs=-1, population_size=20, generations=20)
    nn_grid.fit(X, Y)
    nn_best_params = nn_grid.best_params_
    print("Evolutionary Search Neural Network best parameters: ", nn_best_params)

    return nn_best_params


### Main ###

def main():

    ### Training ###
    # Read feature vectors from file
    print("Reading feature vectors from file")
    features_training = pd.read_json("features/features_training.json")
    features_test = pd.read_json("features/features_test3.json")

    feature_cols = ['avg_compound_text', 'avg_compound_summ', 'pos_text', 'avg_pos_text', 'pos_summ', 'avg_pos_summ', 'neg_text', 'avg_neg_text',
            'neg_summ', 'avg_neg_summ', 'std_text', 'std_summ', 'pct_verif', 'amt_reviews', 'amt_stars', 'starsabove4', 'product_age']
    X = features_training[feature_cols]
    Y = features_training.awesomeness

    # Scaling
    scaler = MinMaxScaler(feature_range=(-1, 1)).fit(X)
    X = scaler.transform(X)
    
    # PCA
    pca = PCA(n_components=12) # usually: improved model performance at cost of accuracy
    X = pca.fit_transform(X)

    # Perform Hyperparameter Optimization
    # nn_best_params = grid_search(X, Y)
    nn_best_params = evolutionary_search(X, Y)

    # nn_best_params = {'activation': 'tanh', 'hidden_layer_sizes': (10, 10, 10, 10), 'learning_rate': 'constant', 'learning_rate_init': 0.05, 'solver': 'sgd'}
    # nn_best_params = {'tol': 0.008330546988037554, 'hidden_layer_sizes': (100, 100, 100, 100), 'alpha': 1.176332001397507e-05, 'activation': 'relu', 'solver': 'sgd', 'learning_rate': 'constant', 'learning_rate_init': 0.1187653569983233, 'shuffle': True}
    
    ### Boosting ###
    model_ab = AdaBoostClassifier(
        RandomForestClassifier(max_depth=5, n_estimators=100, n_jobs=-1, criterion="log_loss", class_weight=None),
        n_estimators=13, learning_rate=2.075)
    model_nn = MLPClassifier(**nn_best_params)

    models = [value for name, value in locals().items() if name.startswith('model_')]
    model_names = [name for name, value in locals().items() if name.startswith('model_')]

    scores = {}
    scoring_methods = ['recall', 'precision', 'f1', 'roc_auc', 'accuracy']
    scores_avg = {}

    # Train models using 10-fold cross validation
    for i in range(len(models)):
        model_name = model_names[i]
        print("Running cross validation on " + model_name)
        models[i].fit(X, Y)
        scores[model_name] = \
            cross_validate(models[i], X, Y, cv=10, scoring=scoring_methods)
        scores_avg[model_name] = {}
        for method in scoring_methods:
            scores_avg[model_name][method] = np.mean(scores[model_name]["test_" + method])
        pickle.dump(models[i], open("./models/" + model_name + ".pkl", "wb"))

    ### Visualize Scores ###
    print("Generating bar plot")
    generate_bar_plot(models, model_names, scoring_methods[0:3], scores_avg)

    print("Printing scores to scores_avg.csv")
    scores_avg_df = pd.DataFrame(scores_avg)
    scores_avg_df = scores_avg_df.round(4)
    scores_avg_df.to_csv("./scores/scores_avg.csv")

    ### Read models if dumped ###
    # print("Reading dumped model from file")
    # # model_ab = pickle.load(open("./models/model_ab.pkl", "rb"))
    # model_nn = pickle.load(open("./models/model_nn.pkl", "rb"))

    # final_model = model_nn

    # ### Predictions ###
    # print("Running predictions")
    # X_test = features_test[feature_cols]
    # X_test = scaler.transform(X_test)
    # X_test = pca.transform(X_test)

    # predictions = final_model.predict(X_test)

    # print("Writing predictions to predictions.json")
    # asin_test = pd.read_json("Toys_and_Games/test3/product_test.json")
    # asin_test.sort_index(axis=0, inplace=True)
    # # print(asin_test.head(10))
    # asin_test.insert(1, "awesomeness", predictions)
    # # print(asin_test.head(10))
    # asin_test.to_json("predictions.json")


if __name__ == '__main__':
    main()

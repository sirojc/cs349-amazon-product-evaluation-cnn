import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

from sklearn.metrics import f1_score
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA

from sklearn_genetic import GASearchCV
from sklearn_genetic.space import Categorical, Continuous, Integer

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

def evolutionary_search(X, Y):
    print("Performing Evolutionary Search")
    model_nn = MLPClassifier(max_iter=2000)
    model_ab = AdaBoostClassifier(RandomForestClassifier())
    model_rf = RandomForestClassifier(n_jobs=-1)
    
    nn_params = {'tol': Continuous(1e-4, 1e-1, distribution='log-uniform'),
        'hidden_layer_sizes': Categorical([(8, 8, 8), (10, 10, 10), (12, 12, 12), (15, 15, 15)]),
        'alpha': Continuous(1e-5, 3e-5),
        'activation': Categorical(['relu', 'tanh', 'logistic']),
        'solver': Categorical(['adam']),
        'learning_rate': Categorical(['adaptive']),
        'learning_rate_init': Continuous(1e-3, 1e-1),
        'shuffle': Categorical([True]),
        'beta_1': Continuous(0.01, 0.999, distribution='uniform'),
        'beta_2': Continuous(0.01, 0.999, distribution='uniform')
        }
    
    ab_params = {'n_estimators': Integer(5, 20),
                'learning_rate': Continuous(1e-3, 1e-1, distribution='log-uniform'),
                'estimator__max_depth': Integer(5, 50),
                'estimator__n_estimators': Integer(5, 20),
                'estimator__criterion': Categorical(['gini', 'entropy', 'log_loss']),
                'estimator__class_weight': Categorical([None, 'balanced']),
                'estimator__max_features': Categorical([None, 'sqrt', 'log2']),
                'estimator__min_samples_split': Integer(2, 10),
                'estimator__min_samples_leaf': Integer(1, 10),
                'estimator__bootstrap': Categorical([True, False])
    }

    rf_params = {'max_depth': Integer(5, 20),
            'n_estimators': Integer(5, 20),
            'criterion': Categorical(['gini', 'entropy', 'log_loss']),
            'class_weight': Categorical([None, 'balanced']),
            'max_features': Categorical([None, 'sqrt', 'log2']),
            'min_samples_split': Integer(2, 10),
            'min_samples_leaf': Integer(1, 10),
    }
    
    nn_grid = GASearchCV(estimator=model_nn, param_grid=nn_params, cv=10, scoring="f1", verbose=True, n_jobs=-1, population_size=20, generations=40)
    nn_grid.fit(X, Y)
    nn_best_params = nn_grid.best_params_
    print("Evolutionary Search Neural Network best parameters: ", nn_best_params)

    ab_grid = GASearchCV(estimator=model_ab, param_grid=ab_params, cv=10, scoring="f1", verbose=True, n_jobs=-1, population_size=20, generations=8)
    ab_grid.fit(X, Y)
    ab_best_params = ab_grid.best_params_
    print("Evolutionary Search Adaboost RF best parameters: ", ab_best_params)

    rf_grid = GASearchCV(estimator=model_rf, param_grid=rf_params, cv=10, scoring="f1", verbose=True, n_jobs=-1, population_size=15, generations=20)
    rf_grid.fit(X, Y)
    rf_best_params = rf_grid.best_params_
    print("Evolutionary Search RF best parameters: ", rf_best_params)

    return nn_best_params, ab_best_params, rf_best_params


### Main ###

def main():

    ### Training ###
    # Read feature vectors from file
    print("Reading feature vectors from file")
    features_training = pd.read_json("features/features_training.json")
    features_test1 = pd.read_json("features/features_test1.json")
    features_test2 = pd.read_json("features/features_test2.json")
    features_test3 = pd.read_json("features/features_test3.json")

    feature_cols = ['avg_compound_text', 'avg_compound_summ', 'avg_pos_text', 'avg_pos_summ', 'avg_neg_text',
                     'avg_neg_summ', 'std_text', 'std_summ', 'pct_verif', 'amt_reviews', 'pct_pos', 'amt_stars', 'starsabove4', 'product_age']
    X = features_training[feature_cols]
    Y = features_training.awesomeness

    # Scaling
    scaler = MinMaxScaler(feature_range=(-1, 1)).fit(X)
    X = scaler.transform(X)
    
    # PCA
    pca = PCA(n_components=13) # usually: improved model performance at cost of accuracy
    X = pca.fit_transform(X)

    # Perform Hyperparameter Optimization
    # nn_best_params = evolutionary_search(X, Y)[0]
    # ab_best_params = evolutionary_search(X, Y)[1]
    # rf_best_params = evolutionary_search(X, Y)[2]

    # 20 gen, max 100 nodes
    # nn_best_params2 = {'tol': 0.0008875737682376969, 'hidden_layer_sizes': (100, 100, 100), 'alpha': 1.7911868165827e-05, 'activation': 'logistic', 
    #                   'solver': 'adam', 'learning_rate': 'adaptive', 'learning_rate_init': 0.014627342326429947, 'shuffle': True, 'beta_1': 0.2718775451755663, 
    #                   'beta_2': 0.11581439166574523}
    # # 40 gen, max 15 nodes
    # nn_best_params1 =  {'tol': 0.00025752203085475364, 'hidden_layer_sizes': (15, 15, 15), 'alpha': 2.8664179218098548e-05, 'activation': 'logistic', 
    #                    'solver': 'adam', 'learning_rate': 'adaptive', 'learning_rate_init': 0.004675519601397775, 'shuffle': True, 'beta_1': 0.8377549786798008, 
    #                    'beta_2': 0.3444312237511523}
    
    # ab_best_params = {'n_estimators': 18, 'learning_rate': 0.017691157767887326}
    # ab_rf_best_params = {'max_depth': 16, 'n_estimators': 15, 'criterion': 'entropy', 'class_weight': None, 'max_features': 'log2', 'min_samples_split': 5, 
    #                 'min_samples_leaf': 7, 'bootstrap': True}
    
    # rf_best_params = {'max_depth': 12, 'n_estimators': 17, 'criterion': 'log_loss', 'class_weight': None, 'max_features': 'log2', 'min_samples_split': 7, 'min_samples_leaf': 8, 'bootstrap': True}

    # # model_ab_old = AdaBoostClassifier(
    # #     RandomForestClassifier(max_depth=5, n_estimators=100, n_jobs=-1, criterion="log_loss", class_weight=None),
    # #     n_estimators=13, learning_rate=2.075)
    
    # model_nn1 = MLPClassifier(**nn_best_params1, max_iter=2000)
    # model_nn2 = MLPClassifier(**nn_best_params2, max_iter=2000)
    # model_ab = AdaBoostClassifier(RandomForestClassifier(**ab_rf_best_params), **ab_best_params)
    # model_rf = RandomForestClassifier(**rf_best_params)

    # models = [value for name, value in locals().items() if name.startswith('model_')]
    # model_names = [name for name, value in locals().items() if name.startswith('model_')]

    # scores = {}
    # scoring_methods = ['recall', 'precision', 'f1']
    # scores_avg = {}

    # # Train models using 10-fold cross validation
    # for i in range(len(models)):
    #     model_name = model_names[i]
    #     print("Running cross validation on " + model_name)
    #     models[i].fit(X, Y)
    #     scores[model_name] = \
    #         cross_validate(models[i], X, Y, cv=10, scoring=scoring_methods)
    #     scores_avg[model_name] = {}
    #     for method in scoring_methods:
    #         scores_avg[model_name][method] = np.mean(scores[model_name]["test_" + method])
    #     pickle.dump(models[i], open("./models/" + model_name + ".pkl", "wb"))

    # ### Visualize Scores ###
    # print("Generating bar plot")
    # generate_bar_plot(models, model_names, scoring_methods[0:3], scores_avg)

    # print("Printing scores to scores_avg.csv")
    # scores_avg_df = pd.DataFrame(scores_avg)
    # scores_avg_df = scores_avg_df.round(4)
    # scores_avg_df.to_csv("./scores/scores_avg.csv")

    ### Read models if dumped ###
    print("Reading dumped model from file")
    # model_ab = pickle.load(open("./models/model_ab.pkl", "rb"))
    # model_nn1 = pickle.load(open("./models/model_nn1.pkl", "rb"))
    # model_nn2 = pickle.load(open("./models/model_nn2.pkl", "rb"))
    # model_rf = pickle.load(open("./models/model_rf.pkl", "rb"))
    model_ab_old = pickle.load(open("./models/model_ab_old.pkl", "rb"))

    final_model = model_ab_old

    ### Predictions ###
    print("Running predictions")
    # X_test1 = features_test1[feature_cols]
    # X_test1 = scaler.transform(X_test1)
    # X_test1 = pca.transform(X_test1)

    # X_test2 = features_test2[feature_cols]
    # X_test2 = scaler.transform(X_test2)
    # X_test2 = pca.transform(X_test2)

    X_test3 = features_test3[feature_cols]
    X_test3 = scaler.transform(X_test3)
    X_test3 = pca.transform(X_test3)

    # predictions1 = final_model.predict(X_test1)
    # predictions2 = final_model.predict(X_test2)
    predictions3 = final_model.predict(X_test3)

    print("Writing predictions to predictions on test3.json")
    asin_test = pd.read_json("Toys_and_Games/test3/product_test.json")
    asin_test.sort_index(axis=0, inplace=True)
    asin_test.insert(1, "awesomeness", predictions3)
    asin_test.to_json("predictions.json")

    # print("Writing predictions to predictions on test2.json")
    # asin_test = pd.read_json("Toys_and_Games/test2/product_test.json")
    # asin_test.sort_index(axis=0, inplace=True)
    # asin_test.insert(1, "awesomeness", predictions2)
    # asin_test.to_json("predictions_test2.json")

    # print("Writing predictions to predictions on test1.json")
    # asin_test = pd.read_json("Toys_and_Games/test1/product_test.json")
    # asin_test.sort_index(axis=0, inplace=True)
    # asin_test.insert(1, "awesomeness", predictions1)
    # asin_test.to_json("predictions_test1.json")


if __name__ == '__main__':
    main()

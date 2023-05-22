import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from sklearn.metrics import f1_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier

from sklearn_genetic import GASearchCV
from sklearn_genetic.space import Integer, Categorical, Continuous

### Helper Functions ###
def nlp(review_list):
    analyzer = SentimentIntensityAnalyzer()
    compounds = []
    for review in review_list:
        if review is None:
            compounds.append(None)
        else:
            compounds.append(analyzer.polarity_scores(review)["compound"])
    return compounds


def get_age_weight(rev_training, index, first_rev_time):
    return ((0.01) / (365 * 24 * 3600)) * (rev_training.unixReviewTime[index] - first_rev_time) + 1


def get_review_age(index, rev_training):
    return rev_training.unixReviewTime[index]


def get_vote_weight(rev_training, index, max_votes):
    return 1 + (0.2 * (np.log(get_vote(index, rev_training) + 1) / np.log(max_votes + 1.1)))


def get_vote(index, rev_training):
    if rev_training.vote[index] is None:
        return 0
    else:
        return int(rev_training.vote[index].replace(",", ""))


def get_image_weight(rev_training, index):
    return 1.3 if rev_training.image[index] is not None else 1


def get_verification_weight(rev_training, index):
    return 1.5 if rev_training.verified[index] else 1.0


def get_avg_weight_compound(compound_list, weight_list):
    weight_compound_list = []
    for i in range(len(compound_list)):
        weight_prod = 1
        for j in range(4):
            weight_prod *= weight_list[j][i]
        if compound_list[i] is not None:
            weight_compound_list.append(compound_list[i] * weight_prod * 10)
    return round(np.sum(weight_compound_list) / len(weight_compound_list), 2) if len(weight_compound_list) != 0 else 0


def get_std_compound(compound_list):
    comp_list = np.array(compound_list)
    comp_list = comp_list[comp_list != None]
    if len(comp_list) > 0:
        return round(np.std(comp_list), 4)
    return 0


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


### Preprocessing ###
def preprocess(associations, prod_training, rev_training):
    # index texts, summary, get avg star rating
    product_text_list = []
    product_summ_list = []
    feature_stars_list = []
    for asin in associations.keys():
        text_list = []
        summ_list = []
        stars_list = []
        star_summary = ["One Star", "Two Stars", "Three Stars", "Four Stars", "Five Stars"]
        for index in associations[asin]:
            text_list.append(rev_training.reviewText[index])
            if rev_training.summary[index] in star_summary:
                summ_list.append(None)
                for i in range(len(star_summary)):
                    if star_summary[i] == rev_training.summary[index]:
                        stars_list.append(i + 1)
                        break
            else:
                summ_list.append(rev_training.summary[index])
        product_text_list.append(text_list)
        product_summ_list.append(summ_list)
        feature_stars_list.append(round(np.mean(stars_list), 1) if stars_list != [] else 3) # change to avg 3 stars if no rating

    # NLP
    product_compound_text_list = []
    product_compound_summ_list = []
    for i in range(len(associations)):
        if i % 5000 == 0:
            print('NLP: {} of {}'.format(i, len(associations)))
        product_compound_text_list.append(nlp(product_text_list[i]))
        product_compound_summ_list.append(nlp(product_summ_list[i]))

    #  Age, Vote, Verification, Image weight, Amount of Reviews, Verification Percentage
    print("Compute remaining features")
    product_age_weight = []
    product_vote_weight = []
    product_verification_weight = []
    product_image_weight = []
    feature_num_rev_list = []
    feature_verification_perc_list = []
    for asin in associations.keys():
        min_age = float("inf")
        age_weight = []
        max_votes = float("-inf")
        vote_weight = []
        verification_weight = []
        image_weight = []
        feature_num_rev_list.append(len(associations[asin]))
        verification_count = 0
        for index in associations[asin]:
            age = get_review_age(index, rev_training)
            vote = get_vote(index, rev_training)
            if age < min_age:
                min_age = age
            if vote > max_votes:
                max_votes = vote

        for index in associations[asin]:
            age_weight.append(get_age_weight(rev_training, index, min_age))
            vote_weight.append(get_vote_weight(rev_training, index, max_votes))
            verification_weight.append(get_verification_weight(rev_training, index))
            image_weight.append(get_image_weight(rev_training, index))
            verification_count += rev_training.verified[index]

        product_age_weight.append(age_weight)
        product_vote_weight.append(vote_weight)
        product_verification_weight.append(verification_weight)
        product_image_weight.append(image_weight)
        feature_verification_perc_list.append(round(verification_count / len(associations[asin]), 2))

    # weighted compound
    feature_avg_compound_text_list = []
    feature_avg_compound_summ_list = []
    feature_std_text_list = []
    feature_std_summ_list = []
    for i in range(len(associations)):
        product_weights = [product_age_weight[i], product_vote_weight[i], product_verification_weight[i],
                           product_image_weight[i]]
        feature_avg_compound_text_list.append(
            get_avg_weight_compound(product_compound_text_list[i], product_weights))
        feature_avg_compound_summ_list.append(
            get_avg_weight_compound(product_compound_summ_list[i], product_weights))
        feature_std_text_list.append(get_std_compound(product_compound_text_list[i]))
        feature_std_summ_list.append(get_std_compound(product_compound_summ_list[i]))

    features = {"avg_compound_text": feature_avg_compound_text_list,
                "avg_compound_summ": feature_avg_compound_summ_list,
                "std_text": feature_std_text_list,
                "std_summ": feature_std_summ_list,
                "pct_verif": feature_verification_perc_list,
                "amt_reviews": feature_num_rev_list,
                "amt_stars": feature_stars_list}

    return features


def add_awesomeness(associations, features, awesomeness_training):
    class_awesomeness = []
    for asin in associations.keys():
        class_awesomeness.append(awesomeness_training[awesomeness_training.asin == asin].awesomeness.values[0])

    features["awesomeness"] = class_awesomeness
    return features


def generate_feature_vectors():
    print('Reading json files for training and testing')
    reviews_training = pd.read_json("Toys_and_Games/train/review_training.json")
    awesomeness_training = pd.read_json("Toys_and_Games/train/product_training.json")
    reviews_test = pd.read_json("Toys_and_Games/test3/review_test.json")
    asin_test = pd.read_json("Toys_and_Games/test3/product_test.json")

    print("Running associations on training")
    associations_training = reviews_training.groupby('asin').apply(lambda x: x.index.tolist())

    print("Running associations on test")
    associations_test = reviews_test.groupby('asin').apply(lambda x: x.index.tolist())
    
    #Preprocessing may run up to 30min (recent Mac M2 Pro)
    print("Preprocessing training")
    features_training = preprocess(associations_training, awesomeness_training, reviews_training) # uncomment to run
    print("Adding ground truth")
    awesomeness = add_awesomeness(associations_training, features_training, awesomeness_training)
    
    df_training = pd.DataFrame(awesomeness, index=list(associations_training.keys()))
    df_training.to_json("./features/features_training.json")
    
    print("Preprocessing test")
    features_test = preprocess(associations_test, asin_test, reviews_test)

    df_test = pd.DataFrame(features_test, index=list(associations_test.keys()))
    df_test.to_json("./features/features_test3.json")


def grid_search(X, Y):
    print("Performing Grid Search")
    # model_ab = AdaBoostClassifier(RandomForestClassifier(max_depth=8, n_estimators=100, n_jobs=-1, criterion="log_loss", class_weight=None))
    model_nn = MLPClassifier(max_iter=2000)

    # ab_params = {'n_estimators': [13],
    #             'estimator__max_depth': [5],
    #             'learning_rate': [2.0725, 2.075, 2.0775]
    #             }
    # nn_params = {'hidden_layer_sizes': [(100, 100, 100, 100), (100, 100, 100, 100, 100)],
    #             'activation': ['relu', 'tanh', 'logistic'],
    #             'solver': ['adam', 'sgd'],
    #             'learning_rate': ['constant', 'adaptive'],
    #             'learning_rate_init': [0.01, 0.025, 0.05]
    #             }
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

    # ab_grid = GridSearchCV(model_ab, ab_params, cv=10, scoring="f1", verbose=2, n_jobs=-1)
    #nn_grid = GridSearchCV(model_nn, nn_params, cv=10, scoring="f1", verbose=2, n_jobs=-1)
    nn_grid = GASearchCV(estimator=model_nn, param_grid=nn_params, cv=10, scoring="f1", verbose=True, n_jobs=-1, population_size=30, generations=40)

    #ab_grid.fit(X, Y)
    nn_grid.fit(X, Y)

    # ab_best_params = ab_grid.best_params_
    nn_best_params = nn_grid.best_params_

    # print("AdaBoost best parameters: ", ab_best_params)
    print("Neural Network best parameters: ", nn_best_params)

    return nn_best_params

### Main ###
def main():
    # Preprocessing - uncomment if needed
    generate_feature_vectors()

    ### Training ###
    # Read feature vectors from file if already preprocessed
    print("Reading feature vectors from file")
    features_training = pd.read_json("features/features_training.json")
    features_test = pd.read_json("features/features_test3.json")

    feature_cols = ['avg_compound_text', 'avg_compound_summ', 'std_text', 'std_summ', 'pct_verif', 'amt_reviews',
                    'amt_stars']
    X = features_training[feature_cols]
    Y = features_training.awesomeness

    # Scaling
    scaler = MinMaxScaler(feature_range=(-1, 1)).fit(X)
    X = scaler.transform(X)

    # Perform Grid Search
    nn_best_params = grid_search(X, Y)
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
    print("Reading dumped model from file")
    # model_ab = pickle.load(open("./models/model_ab.pkl", "rb"))
    model_nn = pickle.load(open("./models/model_nn.pkl", "rb"))

    final_model = model_nn

    ### Predictions ###
    print("Running predictions")
    X_test = features_test[feature_cols]
    X_test = scaler.transform(X_test)

    predictions = final_model.predict(X_test)

    print("Writing predictions to predictions.json")
    asin_test = pd.read_json("Toys_and_Games/test3/product_test.json")
    asin_test.sort_index(axis=0, inplace=True)
    # print(asin_test.head(10))
    asin_test.insert(1, "awesomeness", predictions)
    # print(asin_test.head(10))
    asin_test.to_json("predictions.json")


if __name__ == '__main__':
    main()

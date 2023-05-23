import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

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
    return ((0.03) / (365 * 24 * 3600)) * (rev_training.unixReviewTime[index] - first_rev_time) + 1 # increase weight of newer reviews ( 0.01 to 0.03 per year)


def get_review_age(index, rev_training):
    return rev_training.unixReviewTime[index]


def get_vote_weight(rev_training, index, max_votes):
    return 1 + (0.2 * (np.log(get_vote(index, rev_training) + 1) / np.log(max_votes + 1.1))) # increase weight from 0.2 to 0.5


def get_vote(index, rev_training):
    if rev_training.vote[index] is None:
        return 0
    else:
        return int(rev_training.vote[index].replace(",", ""))


def get_image_weight(rev_training, index):
    return 1 if rev_training.image[index] is not None else 1


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
                        if rev_training.verified[index] == 0:
                            stars_list.append(0.8*(i+1)) # reduce weight of non-verified reviews
                        else:
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


### Generate Feature Vectors ###

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
    features_training = preprocess(associations_training, awesomeness_training, reviews_training)
    print("Adding ground truth")
    awesomeness = add_awesomeness(associations_training, features_training, awesomeness_training)
    
    df_training = pd.DataFrame(awesomeness, index=list(associations_training.keys()))
    df_training.to_json("./features/features_training.json")
    
    print("Preprocessing test")
    features_test = preprocess(associations_test, asin_test, reviews_test)

    df_test = pd.DataFrame(features_test, index=list(associations_test.keys()))
    df_test.to_json("./features/features_test3.json")


if __name__ == '__main__':
    generate_feature_vectors()
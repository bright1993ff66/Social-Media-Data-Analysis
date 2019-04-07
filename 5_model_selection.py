import os
working_directory = r'F:\CityU\Hong Kong Twitter 2016\emoji2vec'
os.chdir(working_directory)
print('The current working directory has changed to: ',os.getcwd())

#===================================================================================================================
# Impore Relevant Packages
# Commonly used
import math
import gensim.models as gs
import pickle as pk
import numpy as np
import pandas as pd
from collections import Counter
import time
import read_data

# Classifiers
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn import tree
from sklearn.utils import shuffle

# Model Evaluations
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score, \
    classification_report
from sklearn.model_selection import GridSearchCV, train_test_split

# This paper requires
import phrase2vec as p2v
from twitter_sentiment_dataset import TweetTrainingExample
from model import ModelParams

# Build my classifier
from keras.utils.np_utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras import backend as K
from keras import models
from keras import layers
from keras.wrappers.scikit_learn import KerasClassifier

# tokenization
import nltk.tokenize as tk

# Cope with the imbalanced issue
from imblearn.pipeline import make_pipeline
from imblearn.over_sampling import ADASYN, SMOTE, RandomOverSampler
from imblearn.metrics import geometric_mean_score

# Visualization
from matplotlib import pyplot as plt

# Ignore the tedious warnings
import warnings
warnings.simplefilter('ignore', DeprecationWarning)
warnings.simplefilter('ignore', FutureWarning)

# Some important paths
w2v_path = './data/word2vec/'


def list_of_array_to_array(list_array):
    shape = list(list_array[0].shape)
    shape[:0] = [len(list_array)]
    arr = np.concatenate(list_array).reshape(shape)
    return arr


def precision(y_true, y_pred):
    """Precision metric.

    Only computes a batch-wise average of precision.

    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall(y_true, y_pred):
    """Recall metric.

    Only computes a batch-wise average of recall.

    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


def get_ffnn_model(dropout_rate=0.2):
    model = models.Sequential()
    # Dense(1000) is a fully-connected layer with 1000 hidden units.
    # in the first layer, you must specify the expected input data shape:
    # here, 100-dimensional vectors.
    model.add(layers.Dense(1000, activation='relu', input_dim=100, name='dense_1'))
    model.add(layers.Dropout(dropout_rate, name='dropout_1'))
    model.add(layers.Dense(1000, activation='relu', name='dense_2'))
    model.add(layers.Dropout(dropout_rate, name='dropout_2'))
    model.add(layers.Dense(1000, activation='relu', name='dense_3'))
    model.add(layers.Dropout(dropout_rate, name='dropout_3'))
    model.add(layers.Dense(3, activation='softmax', name='output_layer'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=[precision, recall, f1])
    return model


"""
def plot_accuracy(clf, fit_history, train_data, train_y_one_hot, test_data, test_y_one_hot):
    # Plot the accuracy score on the training data
    _, train_acc = clf.evaluate(train_data, train_y_one_hot, verbose=0)
    # Plot the accuracy score on the tesing data
    _, test_acc = clf.evaluate(test_data, test_y_one_hot, verbose=0)
    print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))
    # plot history
    plt.plot(fit_history.history['acc'], label='train')
    plt.plot(fit_history.history['val_acc'], label='test')
    plt.legend()
    plt.show()
"""


def prepare_tweet_vector_averages_for_prediction(tweets, p2v):
    """
    Take the vector sum of all tokens in each tweet

    Args:
        tweets: All tweets
        p2v: Phrase2Vec model

    Returns:
        Average vectors for each tweet
        Truth
    """
    tokenizer = tk.TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=True)

    avg_vecs = list()

    for tweet in tweets:
        tokens = tokenizer.tokenize(tweet)
        avg_vecs.append(np.sum([p2v[x] for x in tokens], axis=0) / len(tokens))

    return avg_vecs


# construct the whole review and sample datasets: sample for prediction; review for validation
def construct_whole_sample_datasets(en_sample, zh_sample, en_review, zh_review):
    final_review = pd.concat((en_review, zh_review), axis=0)
    final_sample = pd.concat((en_sample, zh_sample), axis=0)
    final_sample = final_sample.reset_index(drop=True)
    final_sample = shuffle(final_sample)
    sample_index = final_sample.index.tolist()
    final_review = final_review.reset_index(drop=True)
    final_review = final_review.reindex(sample_index)
    return final_sample, final_review


# Need to change
def kfold_with_smote(clf, train_valid_data_X, train_valid_label_y, tuned_parameters, X_test, y_test,
                     whole_tweets_array, save_path, clf_name=None):
    """
    clf: the classifier
    train_valid_data_X: the data used for cross validation with grid search
    train_valid_label_y: the labels used for cross validation with grid search
    tuned_parameters: the hyperparameters we want to tune
    X_test: the test data
    y_test: the test labels
    whole_tweets_array: a numpy array which records the representations of tweets
    save_path: the path used to save predictions
    clf_name: the name of the classifier
    Return: two dictionaries: 1. the best hyperparameters in cross validation; 2. the mean test score in
    4 fold cross validation
    """
    scores = ['f1']
    # The dict which records the performance based on the evaluation metric
    performance_dict = {}
    # The dict which records the best hyparameter setting for the evaluation metric
    params_dict = {}
    for score in scores:
        print()
        print("# Use %s to check the model's performance..." % score)
        # cv = 4 here implements stratified 4 fold cross validation
        # It means that in each fold, the distribution of each label is consistent with the whole training_valid_data
        if score != 'accuracy':
            Grid_clf = GridSearchCV(clf, tuned_parameters, cv=4, scoring='%s_macro' % score)
        else:
            Grid_clf = GridSearchCV(clf, tuned_parameters, cv=4, scoring='%s' % score)
        Grid_clf.fit(train_valid_data_X, train_valid_label_y)
        print("Best parameters set found on development set:")
        print()
        params_dict[score] = Grid_clf.best_params_
        print(Grid_clf.best_params_)
        print()
        print("Grid scores on the validation set:")
        print()
        # The means show the mean f1 score across 4 fold cross validation
        # stds records the standard deviations
        means = Grid_clf.cv_results_['mean_test_score']
        stds = Grid_clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, Grid_clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean, std * 2, params))
        performance_dict[score] = means
        # Call predict on the estimator with the best found parameters.
        y_true, y_pred = y_test, Grid_clf.predict(X_test)
        # ffnn_dataframe = pd.DataFrame({'y_true': y_true, 'y_pred':y_pred})
        # ffnn_dataframe.to_csv(os.path.join(read_data.desktop, 'ffnn_on_test.csv'))

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='macro')
        recall = recall_score(y_true, y_pred, average='macro')
        f1 = f1_score(y_true, y_pred, average='macro')
        print('How '+clf_name+' performs on the test set.....')
        print("Accuracy: ", accuracy, "Precision: ", precision, "Recall: ", recall, 'f1_score: ', f1)
        print()

        # Make predictions on the review data
        whole_predictions_review = Grid_clf.predict(tweets_representations_whole_sample_array)
        np.save(os.path.join(save_path, 'whole_predictions_by_' + clf_name +'_review'), whole_predictions_review)

        # Make predictions on the 2016 data compare with yao
        whole_predictions_2016_compare_with_yao = Grid_clf.predict(tweets_representations_2016_array_compare_with_yao)
        np.save(os.path.join(save_path, 'whole_predictions_2016_by_' + clf_name + '_compare_with_yao'),
                whole_predictions_2016_compare_with_yao)

        # Make predictions on the whole 2016 data
        whole_predictions_2016 = Grid_clf.predict(tweets_representations_2016_array)
        np.save(os.path.join(save_path, 'whole_predictions_2016_by_' + clf_name), whole_predictions_2016)

        # Make predictions on the whole 2017 data
        whole_predictions_2017 = Grid_clf.predict(whole_tweets_array)
        np.save(os.path.join(save_path, 'whole_predictions_2017_by_' + clf_name), whole_predictions_2017)

    return params_dict, performance_dict


if __name__ == '__main__':

    # Set Global Variables for emoji2vec
    in_dim = 100  # Length of word2vec vectors
    out_dim = 100  # Desired dimension of output vectors
    pos_ex = 4
    neg_ratio = 1
    max_epochs = 40
    dropout = 0.1
    params = ModelParams(in_dim=in_dim, out_dim=out_dim, pos_ex=pos_ex, max_epochs=max_epochs,
                         neg_ratio=neg_ratio, learning_rate=0.001, dropout=dropout, class_threshold=0.5)

    e2v_ours_path = params.model_folder('unicode') + '/emoji2vec_100.bin'

    # Load the FastText word vectors and emoji vectors
    w2v = gs.FastText.load(os.path.join(w2v_path, 'fasttext_model'))
    e2v_ours = gs.KeyedVectors.load_word2vec_format(e2v_ours_path, binary=True)
    # Combine the word vectors and emoji vectors together
    p2v_our_emoji = p2v.Phrase2Vec(out_dim, w2v, e2v=e2v_ours)

    """
    # Create the training tweets and testing tweets
    train_tweets, test_tweets = tsd.load_training_test_sets()

    #  Prepare the Training, Testing Vectors and Corresponding Labels
    train_ours, trainy = tsd.prepare_tweet_vector_averages(train_tweets, p2v_our_emoji)
    trainy_nums = []
    for label in trainy:
        num_label = rename_labels(label)
        trainy_nums.append(num_label)

    test_ours, test_y = tsd.prepare_tweet_vector_averages(test_tweets, p2v_our_emoji)
    test_ours = np.array(test_ours)
    test_nums = []
    for label in test_y:
        num_label = rename_labels(label)
        test_nums.append(num_label)

    whole_tweets = np.concatenate((train_ours, test_ours), axis=0)
    whole_nums = np.concatenate((trainy_nums, test_nums), axis=0).tolist()
    """
    

    # Build the Classifiers
    ffnn_model = get_ffnn_model()
    ffnn_model.summary()
    # The KerasClassifier Wrapper helps us GridSearch the hyperparameters of our neural net
    ffnn_model_wrapper = KerasClassifier(build_fn=get_ffnn_model, epochs=5, batch_size=100, verbose=0)

    # How to tune these classifiers?
    # Decision Tree: max_depth
    classifiers_svm = {'SVM': svm.SVC()}

    classifiers_decision_tree = {'Decision Tree': tree.DecisionTreeClassifier()}

    ensembled_classifiers = {'Random Forest': RandomForestClassifier()}

    # Evaluate the model on the human review data
    # Load the files to generate tweet representations for our classifiers
    final_zh_sample_cleaned_and_translated = pd.read_pickle(
        os.path.join(read_data.tweet_2017, 'final_sample_cleaned_and_translated_2.pkl'))
    final_zh_sample_cleaned_and_translated.loc[
        final_zh_sample_cleaned_and_translated['cleaned_text'] == '', 'cleaned_text'] = \
        final_zh_sample_cleaned_and_translated['text']
    final_en_sample_cleaned = pd.read_pickle(
        os.path.join(read_data.tweet_2017, 'final_en_sample_cleaned_and_translated_2.pkl'))
    final_en_sample_cleaned.loc[final_en_sample_cleaned['cleaned_text'] == '', 'cleaned_text'] = \
        final_en_sample_cleaned['text']
    # Delete the Tweet which has conflicts among reviewers
    new_final_zh_sample = final_zh_sample_cleaned_and_translated.drop([381])
    # Get the text of Chinese and English tweets
    sample_zh_tweets = list(new_final_zh_sample['cleaned_text'])
    sample_en_tweets = list(final_en_sample_cleaned['cleaned_text'])
    # Load the tweets in 2016 - one for comparision with the previous papaer and another one for sentiment computation
    tweets_in_2016_dataframe_compare_with_yao = pd.read_pickle(os.path.join(read_data.tweet_2016,
                                                           'tweet_2016_compare_with_Yao.pkl'))
    tweets_in_2016_dataframe_compare_with_yao = \
        tweets_in_2016_dataframe_compare_with_yao.loc[tweets_in_2016_dataframe_compare_with_yao['cleaned_text'] != '']
    tweets_in_2016_compare_with_yao = list(tweets_in_2016_dataframe_compare_with_yao['cleaned_text'])
    tweets_in_2016_dataframe = pd.read_pickle(os.path.join(read_data.tweet_2016,
                                                           'final_zh_en_for_paper_hk_time_2016.pkl'))
    tweets_in_2016_dataframe = tweets_in_2016_dataframe.loc[tweets_in_2016_dataframe['cleaned_text'] != '']
    tweets_in_2016 = list(tweets_in_2016_dataframe['cleaned_text'])


    # Get the representation of each tweet
    tweets_representations_en_sample = prepare_tweet_vector_averages_for_prediction(sample_en_tweets, p2v_our_emoji)
    tweets_representations_zh_sample = prepare_tweet_vector_averages_for_prediction(sample_zh_tweets, p2v_our_emoji)
    tweets_representations_en_sample_array = list_of_array_to_array(tweets_representations_en_sample)
    tweets_representations_zh_sample_array = list_of_array_to_array(tweets_representations_zh_sample)
    tweets_representations_2016_compare_with_yao = \
        prepare_tweet_vector_averages_for_prediction(tweets_in_2016_compare_with_yao, p2v_our_emoji)
    tweets_representations_2016_array_compare_with_yao = \
        list_of_array_to_array(tweets_representations_2016_compare_with_yao)
    tweets_representations_2016 = prepare_tweet_vector_averages_for_prediction(tweets_in_2016, p2v_our_emoji)
    tweets_representations_2016_array = list_of_array_to_array(tweets_representations_2016)


    # Load the human review result
    sample_path = r'F:\CityU\Datasets\Hong Kong Tweets 2017\human review\human review result'
    en_review = pd.read_excel(os.path.join(sample_path, 'en_sample.xlsx'))
    zh_review = pd.read_excel(os.path.join(sample_path, 'zh_sample.xlsx'))
    new_zh_review = zh_review.drop([381])
    # Use the 'final sentiment' column if you let the sentiment be neutral if one reviewer gives neutral
    # Use final_sentiment_2 if you want the sentiment of a tweet be neutral only if two revieweres give neutral

    final_sample, final_review = construct_whole_sample_datasets(en_sample=final_en_sample_cleaned,
                                                                 zh_sample=new_final_zh_sample,
                                                                 en_review=en_review, zh_review=new_zh_review)

    # final sample is used to compute the tweet representations
    # final review is used to save the human review result
    final_sample.to_pickle(os.path.join(read_data.desktop, 'final_sample.pkl'))
    final_review.to_pickle(os.path.join(read_data.desktop, 'final_review.pkl'))

    whole_sample_tweets = list(final_sample['cleaned_text'])
    tweets_representations_whole_sample = prepare_tweet_vector_averages_for_prediction(whole_sample_tweets,
                                                                                       p2v_our_emoji)
    tweets_representations_whole_sample_array = list_of_array_to_array(tweets_representations_whole_sample)
    # Scheme1: If one reviewer annotates neutral, the sentiment of a tweet would be neutral
    whole_review_result_scheme1 = list(final_review['final sentiment'])
    # Scheme2: The sentiment of a tweet would be neutral only if both two reviewers label it neutral
    whole_review_result_scheme2 = list(final_review['final sentiment_2'])
	
	"""
	# Save the tweet representation for samples
    with open('C:\\Users\\Haoliang Chang\\Desktop\\sample_array.pkl', 'wb') as f:
		pk.dump(tweets_representations_whole_sample_array, f)
	"""


    final_whole_data = pd.read_pickle(os.path.join(read_data.tweet_2017,
                                                   'final_zh_en_for_paper_hk_time_2017.pkl'))
    final_whole_data = final_whole_data.loc[final_whole_data['cleaned_text'] != '']
    print(final_whole_data.shape)
    whole_tweets_in_2017 = list(final_whole_data['cleaned_text'])
    tweets_representation_whole_2017_tweets = \
        prepare_tweet_vector_averages_for_prediction(whole_tweets_in_2017, p2v_our_emoji)
    tweets_representation_whole_2017_tweets_array = list_of_array_to_array(tweets_representation_whole_2017_tweets)
	
	"""
	# Save the tweet representations
    with open('C:\\Users\\Haoliang Chang\\Desktop\\whole_tweets_array_2017.pkl', 'wb') as f_whole_2017:
        pk.dump(tweets_representation_whole_2017_tweets_array, f_whole_2017)
    with open('C:\\Users\\Haoliang Chang\\Desktop\\whole_tweets_array_2016_compare_with_yao.pkl', 'wb') \
            as f_whole_2016_compare_with_yao:
        pk.dump(tweets_representations_2016_array_compare_with_yao, f_whole_2016_compare_with_yao)
    with open('C:\\Users\\Haoliang Chang\\Desktop\\whole_tweets_array_2016.pkl', 'wb') as f_whole_2016:
        pk.dump(tweets_representations_2016_array, f_whole_2016)
	"""

    # Use kfold with GridSearch to compare the performance of different classification methods
    print("========================================================================")
    print('The sentiment would be set to neutral only if two reviewer label it neutral...')
    X_train_valid, X_test, y_train_valid, y_test = train_test_split(tweets_representations_whole_sample_array,
                                                                    whole_review_result_scheme2, test_size=0.2,
                                                                    random_state=777)
	
	"""
    # Use SMOTE to do the oversampling
    smt = SMOTE(random_state=777, k_neighbors=2)
    oversampled_train_validate_data, oversampled_train_validate_y = smt.fit_sample(X_train_valid,                                                                                y_train_valid)
    # Save the X_test and y_test for model selection
    np.save(os.path.join(read_data.desktop, 'test_data_for_model_compare'), X_test)
    np.save(os.path.join(read_data.desktop, 'test_label_for_model_compare'), y_test)
	"""

    starting_time = time.time()
    #
    print('Decision Tree...')

    tuned_parameters_tree = {'max_depth': np.arange(3, 11)}

    params_dict_tree, performance_dict_tree = kfold_with_smote(clf=classifiers_decision_tree['Decision Tree'],
                                                                  train_valid_data_X=X_train_valid,
                                                             train_valid_label_y=y_train_valid,
                                                             tuned_parameters=tuned_parameters_tree, X_test=X_test,
                                                           y_test=y_test,
                                                           whole_tweets_array=tweets_representation_whole_2017_tweets_array,
                                                           save_path=read_data.model_selection_path, clf_name='DT')
    print(params_dict_tree)
    print()
    print(performance_dict_tree)

    print('Random Forest...')

    tuned_parameters_rf = {'n_estimators': np.arange(10, 60)}
    params_dict_rf, performance_dict_rf = kfold_with_smote(clf=ensembled_classifiers['Random Forest'],
                                                           train_valid_data_X=X_train_valid,
                                                           train_valid_label_y=y_train_valid,
                                                             tuned_parameters=tuned_parameters_rf, X_test=X_test,
                                                           y_test=y_test,
                                                           whole_tweets_array=tweets_representation_whole_2017_tweets_array,
                                                           save_path=read_data.model_selection_path, clf_name='RF')
    print(params_dict_rf)
    print()
    print(performance_dict_rf)

    print('SVM......')
    tuned_parameters_svm = [{'kernel': ['rbf', 'poly', 'sigmoid'], 'C': [1, 10, 100, 1000]}]
    params_dict_svm, performance_dict_svm = kfold_with_smote(clf=classifiers_svm['SVM'],
                                                             train_valid_data_X=X_train_valid,
                                                             train_valid_label_y=y_train_valid,
                                                           tuned_parameters=tuned_parameters_svm, X_test=X_test,
                                                           y_test=y_test,
                                                           whole_tweets_array=tweets_representation_whole_2017_tweets_array,
                                                           save_path=read_data.model_selection_path, clf_name='SVM')
    print(params_dict_svm)
    print()
    print(performance_dict_svm)

    print('Neural Net...')

    dropout_rate = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    tuned_parameters_ffnn = dict(dropout_rate=dropout_rate)

    params_dict_ffnn, performance_dict_ffnn = kfold_with_smote(clf=ffnn_model_wrapper,
                                                               tuned_parameters=tuned_parameters_ffnn,
                                                               train_valid_data_X=X_train_valid,
                                                               train_valid_label_y=y_train_valid,
                                                               X_test = X_test, y_test = y_test,
                                                               whole_tweets_array = tweets_representation_whole_2017_tweets_array,
                                                               save_path = read_data.model_selection_path, clf_name='ffnn')
    print(params_dict_ffnn)
    print()
    print(performance_dict_ffnn)

    end_time = time.time()
    print('Total time for training is: ', end_time-starting_time)
	
	"""
    # For instance, if SVM performs best, then use the following codes to make predictions of 2016 tweets
    predictions = np.load(read_data.model_selection_path, 'whole_predictions_2016_by_SVM.npy')
    tweets_in_2016_dataframe['sentiment'] = list(predictions)
    tweets_in_2016_dataframe.to_pickle(os.path.join(read_data.tweet_2016,
                                                     'tweet_2016_compare_with_Yao_with_sentiment.pkl'))
	"""







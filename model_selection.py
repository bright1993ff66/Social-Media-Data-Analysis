# import os
# working_directory = r'F:\CityU\Hong Kong Twitter 2016\emoji2vec'
# os.chdir(working_directory)
# print('The current working directory has changed to: ',os.getcwd())

#===================================================================================================================
# Impore Relevant Packages
# Commonly used
import os
import gensim.models as gs
import numpy as np
import pandas as pd
from collections import Counter
import time
import read_data
import utils
import csv

# Classifiers
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn import tree
from sklearn.utils import shuffle, compute_class_weight

# Model Evaluations
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score, \
    classification_report
from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedKFold

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
from tensorflow.keras import backend
from tensorflow import set_random_seed

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
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Specify the random seed
random_seed = 777

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


def weighted_categorical_crossentropy(weights):
    """
    A weighted version of keras.objectives.categorical_crossentropy

    Variables:
        weights: numpy array of shape (C,) where C is the number of classes

    Usage:
        weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
        loss = weighted_categorical_crossentropy(weights)
        model.compile(loss=loss,optimizer='adam')
    """

    weights = K.variable(weights)

    def loss(y_true, y_pred):
        # scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        # calc
        loss = y_true * K.log(y_pred) * weights
        loss = -K.sum(loss, -1)
        return loss

    return loss


def get_ffnn_model(dropout_rate=0.2):
    model = models.Sequential()
    # Dense(1000) is a fully-connected layer with 1000 hidden units.
    # in the first layer, you must specify the expected input data shape:
    # here, 100-dimensional vectors.
    model.add(layers.Dense(1000, activation='relu', input_dim=100, name='dense_1'))
    model.add(layers.Dropout(dropout_rate, name='dropout_1'))
    model.add(layers.Dense(1000, activation='relu',name='dense_2'))
    model.add(layers.Dropout(dropout_rate, name='dropout_2'))
    model.add(layers.Dense(1000, activation='relu',name='dense_3'))
    model.add(layers.Dropout(dropout_rate, name='dropout_3'))
    model.add(layers.Dense(3, activation='softmax', name='output_layer'))
    # loss = weighted_categorical_crossentropy(label_weights)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=[precision, recall, f1])
    return model


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
        skf = StratifiedKFold(n_splits=4)
        cross_validation = skf.get_n_splits(train_valid_data_X, train_valid_label_y)
        if score != 'accuracy':
            Grid_clf = GridSearchCV(clf, tuned_parameters, cv=cross_validation, scoring='%s_macro' % score)
        else:
            Grid_clf = GridSearchCV(clf, tuned_parameters, cv=cross_validation, scoring='%s' % score)
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
        print('The distribution of the test set is: ')
        print(Counter(y_test))
        print('The distribution of the predictions computed by ', clf_name, ' is: ')
        print(Counter(y_pred))
        print()
        # ffnn_dataframe = pd.DataFrame({'y_true': y_true, 'y_pred':y_pred})
        # ffnn_dataframe.to_csv(os.path.join(read_data.desktop, 'ffnn_on_test.csv'))

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average=None)
        precision_macro = precision_score(y_true, y_pred, average='macro')
        recall = recall_score(y_true, y_pred, average=None)
        recall_macro = recall_score(y_true, y_pred, average='macro')
        f1 = f1_score(y_true, y_pred, average=None)
        f1_macro = f1_score(y_true, y_pred, average='macro')
        print('How '+clf_name+' performs on the test set.....')
        print("Accuracy: ", accuracy, "Precision: ", precision, "Recall: ", recall, 'f1_score: ', f1)
        print('The Macro scores are...')
        print("Accuracy: ", accuracy, "Precision: ", precision_macro, "Recall: ", recall_macro,
              'f1_score: ', f1_macro)
        print()

        # Make predictions on the review data
        whole_predictions_review = Grid_clf.predict(tweets_representations_whole_sample_array)
        print('The distribution of the review prediction is:')
        print(Counter(whole_predictions_review))
        np.save(os.path.join(save_path, 'whole_predictions_by_' + clf_name +'_review'), whole_predictions_review)

        # Make predictions on the whole 2017 data
        whole_predictions_combined = Grid_clf.predict(whole_tweets_array)
        print('The sentiment distribution of the combined tweet dataframe is:')
        print(Counter(whole_predictions_combined))
        np.save(os.path.join(save_path, 'whole_predictions_tweet_combined_by_' + clf_name), whole_predictions_combined)

    return params_dict, performance_dict


if __name__ == '__main__':

    # Use kfold with GridSearch to compare the performance of different classification methods
    print("========================================================================")
    print('The sentiment would be set to neutral only if two reviewer label it neutral...')
    # Load the tweet_representation_array
    tweets_representations_whole_sample_array = np.load(os.path.join(read_data.tweet_representation_path,
                                                                     'whole_sample_array.npy'))
    whole_review_result_scheme2 = np.load(os.path.join(read_data.tweet_representation_path,
                                                       'whole_samply_label.npy'))
    # Load the data and label for train and validation
    X_train_valid = np.load(os.path.join(read_data.tweet_representation_path,
                                         'train_valid_cross_validation_data.npy'))
    y_train_valid = np.load(os.path.join(read_data.tweet_representation_path,
                                         'train_valid_cross_validation_label.npy'))
    # Load the data and label for test
    X_test = np.load(os.path.join(read_data.tweet_representation_path, 'test_data_for_model_compare.npy'))
    y_test = np.load(os.path.join(read_data.tweet_representation_path, 'test_label_for_model_compare.npy'))
    # Load the data for the whole tweet combined array
    whole_combined_array = np.load(os.path.join(read_data.tweet_combined_path, 'tweet_representations',
                                                'tweet_combined_repre.npy'))

    # Use SMOTE to do the oversampling
    smt = SMOTE(random_state=random_seed, k_neighbors=1)
    oversampled_train_validate_data, oversampled_train_validate_y = smt.fit_sample(X_train_valid,
                                                                                   y_train_valid)
    print('====================================================')
    print('The distribution of the train_valid_data is: ')
    print(Counter(y_train_valid))
    print('The distribution of the oversampled data is: ')
    print(Counter(oversampled_train_validate_y))
    print('====================================================')

    # Build the Classifiers
    ffnn_model = get_ffnn_model()
    ffnn_model.summary()
    # The KerasClassifier Wrapper helps us GridSearch the hyperparameters of our neural net
    ffnn_model_wrapper = KerasClassifier(build_fn=get_ffnn_model, verbose=0,
                                         epochs=5, batch_size=128)

    classifiers_svm = {'SVM': svm.SVC(random_state=random_seed)}

    classifiers_decision_tree = {'Decision Tree': tree.DecisionTreeClassifier(random_state=random_seed)}

    ensembled_classifiers = {'Random Forest': RandomForestClassifier(random_state=random_seed)}

    starting_time = time.time()

    print('Decision Tree...')

    tuned_parameters_tree = {'max_depth': np.arange(3, 11)}

    params_dict_tree, performance_dict_tree = kfold_with_smote(clf=classifiers_decision_tree['Decision Tree'],
                                                                  train_valid_data_X=oversampled_train_validate_data,
                                                             train_valid_label_y=oversampled_train_validate_y,
                                                             tuned_parameters=tuned_parameters_tree, X_test=X_test,
                                                           y_test=y_test,
                                                           whole_tweets_array=whole_combined_array,
                                                           save_path=read_data.model_selection_path_oversampling,
                                                               clf_name='DT')
    print('The best hyperparameter setting is....')
    print(params_dict_tree)
    print()

    print('Random Forest...')

    tuned_parameters_rf = {'n_estimators': np.arange(10, 60)}
    params_dict_rf, performance_dict_rf = kfold_with_smote(clf=ensembled_classifiers['Random Forest'],
                                                           train_valid_data_X=oversampled_train_validate_data,
                                                           train_valid_label_y=oversampled_train_validate_y,
                                                             tuned_parameters=tuned_parameters_rf, X_test=X_test,
                                                           y_test=y_test,
                                                           whole_tweets_array=whole_combined_array,
                                                           save_path=read_data.model_selection_path_oversampling,
                                                           clf_name='RF')
    print('The best hyperparameter setting is....')
    print(params_dict_rf)
    print()

    print('SVM......')
    tuned_parameters_svm = {'kernel': ['rbf', 'poly', 'sigmoid'], 'C': [1, 10, 100, 1000]}
    params_dict_svm, performance_dict_svm = kfold_with_smote(clf=classifiers_svm['SVM'],
                                                             train_valid_data_X=oversampled_train_validate_data,
                                                             train_valid_label_y=oversampled_train_validate_y,
                                                           tuned_parameters=tuned_parameters_svm, X_test=X_test,
                                                           y_test=y_test,
                                                           whole_tweets_array=whole_combined_array,
                                                           save_path=read_data.model_selection_path_oversampling,
                                                             clf_name='SVM')
    print('The best hyperparameter setting is....')
    print(params_dict_svm)
    print()

    print('Neural Net...')

    dropout_rate = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    tuned_parameters_ffnn = dict(dropout_rate=dropout_rate)

    params_dict_ffnn, performance_dict_ffnn = kfold_with_smote(clf=ffnn_model_wrapper,
                                                               tuned_parameters=tuned_parameters_ffnn,
                                                               train_valid_data_X=oversampled_train_validate_data,
                                                               train_valid_label_y=oversampled_train_validate_y,
                                                               X_test = X_test, y_test = y_test,
                                                               whole_tweets_array = whole_combined_array,
                                                               save_path = read_data.model_selection_path_oversampling,
                                                               clf_name='ffnn')
    print('The best hyperparameter setting is....')
    print(params_dict_ffnn)
    print()

    end_time = time.time()
    print('Total time for training is: ', end_time-starting_time)






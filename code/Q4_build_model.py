import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import cross_validate
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.compose import make_column_selector as selector
from sklearn.linear_model import LogisticRegression
from imblearn.pipeline import make_pipeline as make_pipeline_with_sampler
from imblearn.under_sampling import RandomUnderSampler

with open('./data/dat_transformed.pickle', 'rb') as handle:
    dat = pickle.load(handle)

# Q4 modeling
X = dat[['acqCountry', 'availableMoney', 'cardPresent', 'creditLimit', 'currentBalance', 'merchantCategoryCode',
       'transactionAmount', 'expirationDateKeyInMatch', 'transactionType', 'cardCVV', 'enteredCVV' ,
        'transactionDateTime', 'dateOfLastAddressChange', 'accountOpenDate']]

Y = dat['isFraud']

def create_new_features(X):

    X['address_open_diff'] = (X['dateOfLastAddressChange'] - X['accountOpenDate']) / np.timedelta64(1, 'D')
    X['transaction_address_diff'] = (X['transactionDateTime'] - X['dateOfLastAddressChange']) / np.timedelta64(1, 'D')
    X['cvv_match'] = (X.cardCVV == X.enteredCVV) * 1.
    print(X.cvv_match.value_counts())
    X['utilization_rate'] = X.currentBalance / X.creditLimit
    print(X.utilization_rate.describe())
    X['transaction_over_limit'] = X.transactionAmount / X.creditLimit
    print(X.transaction_over_limit.describe())
    X['transaction_over_balance'] = X.transactionAmount / (X.currentBalance + 1)
    print(X.transaction_over_balance.describe())
    return X

X = create_new_features(X)

cat_vars=['acqCountry','cardPresent','merchantCategoryCode','expirationDateKeyInMatch','transactionType']
# X = create_cat_features(X, cat_vars)
X = X.drop(columns = ['cardCVV', 'enteredCVV', 'transactionDateTime', 'dateOfLastAddressChange', 'accountOpenDate'])


# Start modeling

def fit_model(pipeline, index, model_name, scores):
    index += [model_name]
    cv_result = cross_validate(pipeline, X, Y, scoring=["accuracy", "balanced_accuracy", "roc_auc"])
    scores["Accuracy"].append(cv_result["test_accuracy"].mean())
    scores["Balanced accuracy"].append(cv_result["test_balanced_accuracy"].mean())
    scores["test_roc_auc"].append(cv_result["test_roc_auc"].mean())

    df_scores = pd.DataFrame(scores, index=index)
    return(df_scores, scores, index)

scores = {"Accuracy": [], "Balanced accuracy": [], "test_roc_auc": []}
index = []
df_scores, scores, index = fit_model(DummyClassifier(strategy="most_frequent"), index, 'Dummy classifier', scores)

num_pipe = make_pipeline(
    StandardScaler(), SimpleImputer(strategy="mean", add_indicator=True)
)

cat_pipe = make_pipeline(
    SimpleImputer(strategy="constant", fill_value="missing"),
    OneHotEncoder(handle_unknown="ignore"),
)

preprocessor_linear = make_column_transformer(
    (num_pipe, selector(dtype_include="number")),
    (cat_pipe, selector(dtype_include="category")),
    n_jobs=2,
)

lr_clf = make_pipeline(preprocessor_linear, LogisticRegression(max_iter=1000))
df_scores, scores, index  = fit_model(lr_clf, index, 'Logistic regression', scores)

# logistic regression with class_weight
lr_clf.set_params(logisticregression__class_weight="balanced")
df_scores, scores, index  = fit_model(lr_clf, index, 'Logistic regression class weight', scores)


# Logistic regression with undersampling

lr_clf = make_pipeline_with_sampler(
    preprocessor_linear,
    RandomUnderSampler(random_state=42),
    LogisticRegression(max_iter=1000),
)
df_scores, scores, index  = fit_model(lr_clf, index, 'Logistic regression undersample', scores)

print(df_scores)

# TODO: try randomForest on a beefier machine since took too long on my pathetic laptop :(
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.preprocessing import OrdinalEncoder
#
# num_pipe = SimpleImputer(strategy="mean", add_indicator=True)
# cat_pipe = make_pipeline(
#     SimpleImputer(strategy="constant", fill_value="missing"),
#     OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
# )
#
# preprocessor_tree = make_column_transformer(
#     (num_pipe, selector(dtype_include="number")),
#     (cat_pipe, selector(dtype_include="category")),
#     n_jobs=2,
# )
#
# rf_clf = make_pipeline(
#     preprocessor_tree, RandomForestClassifier(random_state=42, n_jobs=2)
# )
# df_scores, scores, index  = fit_model(rf_clf, index, 'Random Forest', scores)
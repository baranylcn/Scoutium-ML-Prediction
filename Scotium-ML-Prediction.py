import numpy as np
import pandas as pd
import seaborn as sns
import warnings
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, cross_validate
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC


pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.2f' % x)

warnings.simplefilter(action='ignore', category=Warning)

attributes = pd.read_csv("scoutium_attributes.csv", sep=";")
attributes.shape
potential_labels = pd.read_csv("scoutium_potential_labels.csv", sep=";")
potential_labels.shape
merged_df = pd.merge(attributes, potential_labels)


merged_df.drop(merged_df[merged_df["position_id"]==1].index, inplace=True)
merged_df.drop(merged_df[merged_df["potential_label"]=="below_average"].index, inplace=True)


df = pd.pivot_table(merged_df,
                       values='attribute_value',
                       index=['player_id', 'position_id', 'potential_label'],
                       columns='attribute_id')
df = df.reset_index()
df.drop("player_id", axis=1, inplace=True)


df.head()
"""
attribute_id  position_id potential_label  4322  4323  4324  4325  4326  4327  4328  4329  4330  4332  4333  4335  4338  4339  4340  4341  4342  4343  4344  4345  4348  4349  4350  4351  4352  4353  4354  4355  4356  4357  4407  4408  4423  4426
0                       7         average 50.50 50.50 34.00 50.50 45.00 45.00 45.00 45.00 50.50 56.00 39.50 34.00 39.50 39.50 45.00 45.00 50.50 28.50 23.00 39.50 28.50 28.50 45.00 50.50 56.00 34.00 39.50 50.50 34.00 34.00 56.00 34.00 34.00 56.00
1                       9         average 67.00 67.00 67.00 67.00 67.00 67.00 67.00 67.00 67.00 67.00 67.00 67.00 67.00 67.00 67.00 67.00 67.00 67.00 56.00 67.00 67.00 56.00 67.00 67.00 67.00 67.00 78.00 67.00 67.00 67.00 67.00 67.00 56.00 78.00
2                       3         average 67.00 67.00 67.00 67.00 67.00 67.00 67.00 78.00 67.00 67.00 78.00 56.00 67.00 67.00 67.00 67.00 67.00 56.00 56.00 67.00 67.00 56.00 56.00 67.00 67.00 67.00 78.00 67.00 67.00 67.00 67.00 67.00 56.00 78.00
3                       4         average 67.00 78.00 67.00 67.00 67.00 78.00 78.00 78.00 56.00 67.00 67.00 67.00 78.00 78.00 56.00 67.00 67.00 45.00 45.00 56.00 67.00 67.00 67.00 67.00 78.00 67.00 67.00 67.00 56.00 67.00 56.00 67.00 45.00 56.00
4                       9         average 67.00 67.00 78.00 78.00 67.00 67.00 67.00 67.00 89.00 78.00 67.00 67.00 67.00 56.00 56.00 67.00 78.00 56.00 56.00 67.00 56.00 67.00 56.00 67.00 67.00 56.00 67.00 67.00 56.00 67.00 89.00 56.00 67.00 78.00
"""



df["position_id"] = df["position_id"].astype(str)
df["potential_label"] = df["potential_label"].astype(str)

def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

check_df(df)

def grab_col_names(dataframe, cat_th=8, car_th=20):
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_cat = grab_col_names(df)

"""
Observations: 271
Variables: 36
cat_cols: 2
num_cols: 34
cat_but_car: 0
num_but_cat: 0
"""


# Summary of categorical variables
def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)

for col in cat_cols:
    cat_summary(df,col)




# Summary of numerical variables
def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)

for col in num_cols:
    num_summary(df, col)



def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")

for col in num_cols:
    target_summary_with_num(df, "potential_label", col)



# Correlation
def correlation_matrix(dataframe, cols):
    fig = plt.gcf()
    fig.set_size_inches(10, 8)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    fig = sns.heatmap(df[cols].corr(), annot=True, linewidths=0.5, annot_kws={'size': 12}, linecolor='w', cmap='RdBu')
    plt.show(block=True)

correlation_matrix(df, num_cols)




################## MISSING VALUES ###################
def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")
    if na_name:
        return na_columns

na_columns = missing_values_table(df, na_name=True)
# NO MISSING VALUE



################## ENCODING ###################
def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns = categorical_cols, drop_first=drop_first)
    return dataframe

df = one_hot_encoder(df, cat_cols, drop_first=True)



################## SCALE ###################
X_scaled = StandardScaler().fit_transform(df[num_cols])
df[num_cols] = pd.DataFrame(X_scaled, columns=df[num_cols].columns)


X = df.drop("potential_label_highlighted", axis=1)
y = df["potential_label_highlighted"]
# potential_label_highlighted is positive or not? Let's create a prediction model for this.

X.columns = X.columns.astype(str)



################### BASE MODELS ####################

def base_models(X, y, scoring="roc_auc"):
    print("Base Models....")
    classifiers = [('LR', LogisticRegression()),
                   ('KNN', KNeighborsClassifier()),
                   ("SVC", SVC()),
                   ("CART", DecisionTreeClassifier()),
                   ("RF", RandomForestClassifier()),
                   ('Adaboost', AdaBoostClassifier()),
                   ('GBM', GradientBoostingClassifier()),
                   ('XGBoost', XGBClassifier(use_label_encoder=False, eval_metric='logloss')),
                   ('LightGBM', LGBMClassifier()),
                   ('CatBoost', CatBoostClassifier(verbose=False))
                   ]

    for name, classifier in classifiers:
        cv_results = cross_validate(classifier, X, y, cv=5, scoring=scoring)
        print(f"{scoring}: {round(cv_results['test_score'].mean(), 4)} ({name}) ")


base_models(X, y, scoring="accuracy")



################################################
# Random Forests
################################################

rf_model = RandomForestClassifier(random_state=17)
rf_model.get_params()

cv_results_rf = cross_validate(rf_model, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])
cv_results_rf['test_accuracy'].mean()
# 0.8707744107744106
cv_results_rf['test_f1'].mean()
# 0.572008486869168
cv_results_rf['test_roc_auc'].mean()
# 0.907276250880902


# Hyperparameter Optimization
rf_params = {"max_depth": [5, None],
             "max_features": [5, 7, "sqrt"],
             "min_samples_split": [2, 5, 8],
             "n_estimators": [100, 200, 500]}

rf_best_grid = GridSearchCV(rf_model, rf_params, cv=5, verbose=True).fit(X, y)

rf_best_grid.best_params_

rf_final = rf_model.set_params(**rf_best_grid.best_params_, random_state=17).fit(X, y)

rf_final.get_params()

cv_results_rf = cross_validate(rf_final, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])
cv_results_rf['test_accuracy'].mean()
# 0.8892255892255893
cv_results_rf['test_f1'].mean()
# 0.6476982097186701
cv_results_rf['test_roc_auc'].mean()
# 0.916138125440451



################################################
# GBM
################################################

gbm_model = GradientBoostingClassifier(random_state=17)

gbm_model.get_params()

cv_results_gbm = cross_validate(gbm_model, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])
cv_results_gbm['test_accuracy'].mean()
# 0.8523905723905724
cv_results_gbm['test_f1'].mean()
# 0.5640336134453781
cv_results_gbm['test_roc_auc'].mean()
# 0.8705426356589147


# Hyperparameter Optimization
gbm_params = {"learning_rate": [0.01, 0.1],
              "max_depth": [3, 8],
              "n_estimators": [100, 200, 500],
              "subsample": [1, 0.5]}

gbm_best_grid = GridSearchCV(gbm_model, gbm_params, cv=5, n_jobs=1, verbose=True).fit(X, y)

gbm_best_grid.best_params_

gbm_final = gbm_model.set_params(**gbm_best_grid.best_params_, random_state=17, ).fit(X, y)


cv_results_gbm = cross_validate(gbm_final, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])
cv_results_gbm['test_accuracy'].mean()
# 0.882087542087542
cv_results_gbm['test_f1'].mean()
# 0.6891355162754758
cv_results_gbm['test_roc_auc'].mean()
# 0.8911557434813249



################################################
# XGBoost
################################################

xgboost_model = XGBClassifier(random_state=17, use_label_encoder=False)
xgboost_model.get_params()
cv_results_xgb = cross_validate(xgboost_model, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])

cv_results_xgb['test_accuracy'].mean()
# 0.8562289562289562
cv_results_xgb['test_f1'].mean()
# 0.6122917588087867
cv_results_xgb['test_roc_auc'].mean()
# 0.8502818886539817

# Hyperparameter Optimization
xgboost_params = {"learning_rate": [0.1, 0.01],
                  "max_depth": [5, 8],
                  "n_estimators": [100, 500, 1000],
                  "colsample_bytree": [0.7, 1]}

xgboost_best_grid = GridSearchCV(xgboost_model, xgboost_params, cv=5, n_jobs=1, verbose=True).fit(X, y)

xgboost_final = xgboost_model.set_params(**xgboost_best_grid.best_params_, random_state=17).fit(X, y)

cv_results = cross_validate(xgboost_final, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])
cv_results['test_accuracy'].mean()
# 0.8782491582491583
cv_results['test_f1'].mean()
# 0.6318431372549019
cv_results['test_roc_auc'].mean()
# 0.8890768146582101



################################################
# LightGBM
################################################

lgbm_model = LGBMClassifier(random_state=17)
lgbm_model.get_params()

cv_results_lgbm = cross_validate(lgbm_model, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])

cv_results_lgbm['test_accuracy'].mean()
# 0.8671380471380472
cv_results_lgbm['test_f1'].mean()
# 0.6023333333333334
cv_results_lgbm['test_roc_auc'].mean()
# 0.8766032417195208


# Hyperparameter Optimization
lgbm_params = {"learning_rate": [0.01, 0.1],
               "n_estimators": [100, 300, 500],
               "colsample_bytree": [0.5, 0.7, 1]}

lgbm_best_grid = GridSearchCV(lgbm_model, lgbm_params, cv=5, n_jobs=1, verbose=True).fit(X, y)

lgbm_best_grid.best_params_

lgbm_final = lgbm_model.set_params(**lgbm_best_grid.best_params_, random_state=17).fit(X, y)

cv_results = cross_validate(lgbm_final, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])

cv_results_lgbm['test_accuracy'].mean()
# 0.8671380471380472
cv_results_lgbm['test_f1'].mean()
# 0.6023333333333334
cv_results_lgbm['test_roc_auc'].mean()
# 0.8766032417195208


# Hyperparameter Optimization with New Values
lgbm_params = {"learning_rate": [0.01, 0.02, 0.05, 0.1],
               "n_estimators": [200, 300, 350, 400],
               "colsample_bytree": [0.9, 0.8, 1]}

lgbm_best_grid = GridSearchCV(lgbm_model, lgbm_params, cv=5, n_jobs=1, verbose=True).fit(X, y)

lgbm_final = lgbm_model.set_params(**lgbm_best_grid.best_params_, random_state=17).fit(X, y)

cv_results_lgbm = cross_validate(lgbm_final, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])

cv_results_lgbm['test_accuracy'].mean()
# 0.8856565656565657
cv_results_lgbm['test_f1'].mean()
# 0.6764922600619195
cv_results_lgbm['test_roc_auc'].mean()
# 0.8837561663143058


# Hyperparameter Optimization for n_estimators
lgbm_model = LGBMClassifier(random_state=17, colsample_bytree=0.9, learning_rate=0.01)

lgbm_params = {"n_estimators": [200, 400, 1000, 5000, 8000, 9000, 10000]}

lgbm_best_grid = GridSearchCV(lgbm_model, lgbm_params, cv=5, n_jobs=1, verbose=True).fit(X, y)

lgbm_final = lgbm_model.set_params(**lgbm_best_grid.best_params_, random_state=17).fit(X, y)

cv_results_lgbm = cross_validate(lgbm_final, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])

cv_results_lgbm['test_accuracy'].mean()
# 0.8856565656565657
cv_results_lgbm['test_f1'].mean()
# 0.6764922600619195
cv_results_lgbm['test_roc_auc'].mean()
# 0.8841789992952783




#################### IMPORTENCE LEVELS OF VARIABLES ####################
def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(11, 9))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')
        plt.clf()
    plt.show(block=True)


plot_importance(lgbm_final, X)



################### PREDICTION ####################
random_user = X.sample(1, random_state=21)
lgbm_final.predict(random_user)

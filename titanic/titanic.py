
# %%
import ipyenv as uu
uu.chdir(__file__)
dir_data = "data/titanic"

# %% 查看数据
import pandas as pd

data = pd.read_csv(f"{dir_data}/train.csv")
data.head()

# %% 丢弃无用数据
data.drop(columns=["Name", "Ticket", "Cabin"], inplace=True)

# %% 处理性别
data["Sex"] = (data["Sex"] == "male").astype("uint")

# %% 登船港口
labels = data["Embarked"].unique().tolist()
data["Embarked"] = data["Embarked"].apply(lambda x: labels.index(x))

# %% 处理缺失数据
data = data.fillna(0)

# %% 数据查分
from sklearn.model_selection import train_test_split

y = data["Survived"].values
x = data.drop(["Survived"], axis=1).values

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
print('train dataset: {0}; test dataset: {1}'.format(X_train.shape, X_test.shape))

#####################################################################
# %% 模型构建与训练
import numpy as np
from sklearn.tree import DecisionTreeClassifier

def cv_score(d):
    clf = DecisionTreeClassifier(max_depth=d)  # 前剪枝
    clf.fit(X_train, y_train)
    train_score = clf.score(X_train, y_train)
    test_score = clf.score(X_test, y_test)
    # print('train score: {0}; test score: {1}'.format(train_score, test_score))
    return (train_score, test_score)

depths = range(2,20)  # 动态监测前剪枝深度
scores = [cv_score(d) for d in depths]

best_score_index = np.argmax(scores, axis=0)[1]
best_score_depth = depths[best_score_index]
best_score = scores[best_score_index][1]
print(f">> Depth={best_score_depth}, score={best_score}")
# 但实际上，每次运行的剪枝深度并不统一

# %% 通过min_impurity_split进行前剪枝
def cv_score(val):
    # clf = DecisionTreeClassifier(criterion='gini', min_impurity_split=val)  # 已弃用
    clf = DecisionTreeClassifier(criterion='gini', min_impurity_decrease=val)
    clf.fit(X_train, y_train)
    tr_score = clf.score(X_train, y_train)
    cv_score = clf.score(X_test, y_test)
    return (tr_score, cv_score)

# 指定参数范围，分别训练模型并计算评分
values = np.linspace(0, 0.1, 50)
scores = [cv_score(v) for v in values]
tr_scores = [s[0] for s in scores]
cv_scores = [s[1] for s in scores]

# 找出评分最高的模型参数
best_score_index = np.argmax(cv_scores)
best_score = cv_scores[best_score_index]
best_param = values[best_score_index]
print('best param: {0}; best score: {1}'.format(best_param, best_score))

# %% 使用GridSearchCV在多组参数之间选择最优的参数
from sklearn.model_selection import GridSearchCV

entropy_thresholds = np.linspace(0, 1, 50)
gini_thresholds = np.linspace(0, 0.1, 50)

param_grid = [{'criterion': ['entropy'],
    'min_impurity_decrease': entropy_thresholds},
    {'criterion': ['gini'],
    'min_impurity_decrease': gini_thresholds},
    {'max_depth': range(2, 10)},
    {'min_samples_split': range(2, 30, 2)}]

clf = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=5)
clf.fit(x, y)
print("best param: {0}\nbest score: {1}".format(clf.best_params_, clf.best_score_))

#####################################################################
# %% 使用TPOT训练
from tpot import TPOTClassifier
tpot = TPOTClassifier(generations=5, population_size=50, verbosity=2, early_stop=1, n_jobs=-2, random_state=1)
tpot.fit(X_train, y_train)

print(tpot.score(X_test, y_test))
tpot.export(uu.rpath("tpot_mnist_pipeline.py"))

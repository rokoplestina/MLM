# Evaluate using Cross Validation
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import confusion_matrix, classification_report
from lazy_property import LazyProperty


class DataBucket(object):

    def __init__(self):
        pass

    @LazyProperty
    def pima_indians_diabetes(self):
        """
        Data Set Information:
        Several constraints were placed on the selection of these instances from a larger database. In particular, all patients here are females at least 21 years old of Pima Indian heritage. ADAP is an adaptive learning routine that generates and executes digital analogs of perceptron-like devices. It is a unique algorithm; see the paper for details.

        Attribute Information:
        1. Number of times pregnant
        2. Plasma glucose concentration a 2 hours in an oral glucose tolerance test
        3. Diastolic blood pressure (mm Hg)
        4. Triceps skin fold thickness (mm)
        5. 2-Hour serum insulin (mu U/ml)
        6. Body mass index (weight in kg/(height in m)^2)
        7. Diabetes pedigree function
        8. Age (years)
        9. Class variable (0 or 1)
        :return: dataframe with data
        """
        url = "https://goo.gl/vhm1eU"
        names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
        return pd.read_csv(url, names=names)


db = DataBucket()
data = db.pima_indians_diabetes

X = data.drop(labels='class', axis=1)
Y = data['class']

# scaler = MinMaxScaler().fit(dataframe.drop(labels='class', axis=1))
scaler = StandardScaler().fit(data.drop(labels='class', axis=1))
rescaledX = pd.DataFrame(scaler.transform(X), columns=X.columns)

kfold = KFold(n_splits=10, random_state=7)
model = LogisticRegression()
# model = KNeighborsClassifier()
scorings = [
            'neg_log_loss', 
            'accuracy',
            'f1',
            'neg_mean_squared_error'
           ]

for scoring in scorings:
    results = cross_val_score(model, rescaledX, Y, cv=kfold, scoring=scoring)
    print("%s: %.4f (%.4f)") % (scoring, results.mean(), results.std())
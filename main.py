import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB

tdf = pd.read_csv('train.csv')
tdft = pd.read_csv('test.csv')

# Creating dummy columns so strings can be instantiated as
# int types, in order to run them through the model
def create_dummies(df,column_name):
    dummies = pd.get_dummies(df[column_name],prefix=column_name)
    df = pd.concat([df,dummies],axis=1)
    return df

tdf = tdf.dropna(axis=0)    # Dropping the rows with missing values

tdf = create_dummies(tdf, "Sex")
tdf = create_dummies(tdf, "Sex")

tdft['Age'] = tdft['Age'].interpolate(method='pchip')
tdft = create_dummies(tdft,"Sex")
tdft = create_dummies(tdft,"Sex")

features = ['Age', 'Pclass', 'Sex_female', 'Sex_male']
feature = ['Age', 'Pclass', 'Sex_female', 'Sex_male']
y = tdf['Survived']
X = tdf[features]
test_x = tdft[feature]

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

lr1 = LogisticRegression(random_state=1)
lr2 = RandomForestClassifier(n_estimators=220, random_state=1)
lr3 = GaussianNB()
lr1.fit(x_train, y_train)
predictions = lr1.predict(x_test)

accuracy = round(accuracy_score(y_test, predictions), 2)
final_accuracy = round(accuracy * 100)
print('The accuracy of the model is: ', final_accuracy, '%')

eclf = VotingClassifier(
    estimators=[('lr', lr1), ('rf', lr2), ('gnb', lr3)],
    voting='hard')

for clf, label in zip([lr1, lr2, lr3, eclf], ['Logistic Regression', 'Random Forest', 'naive Bayes', 'Ensemble']):
    scores = cross_val_score(clf, X, y, scoring='accuracy', cv=5)
    print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))

print('\n')
eclf.fit(X, y)
unseen_predictions = eclf.predict(test_x)
print(unseen_predictions)

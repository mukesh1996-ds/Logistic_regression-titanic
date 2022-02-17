from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from load_csv import load_data_csv
from get_basic_info import check_top_column
final_train = load_data_csv('final_train.csv')
print(check_top_column(final_train))

cols = ["Age","Fare","TravelAlone","Pclass_1","Pclass_2","Embarked_C","Embarked_S","Sex_male","IsMinor"] 
X = final_train[cols]
y = final_train['Survived']
# Build a logreg and compute the feature importances
model = LogisticRegression()
# create the RFE model and select 8 attributes
model.fit(X,y)    

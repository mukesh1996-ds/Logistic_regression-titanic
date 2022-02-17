from load_csv import load_data_csv

# Loading the training dataset
train_df = load_data_csv('G:\\Kaggle_compitation\\Logistic Regression\\Dataset\\train.csv')
print(f"The training data is :- \n {train_df.head()}")

# Loading the testing dataset

test_df = load_data_csv('G:\\Kaggle_compitation\\Logistic Regression\\Dataset\\test.csv')
print(f"The testing data is :- \n {test_df.head()}")


def check_shape(data):
    return data.shape

def check_null_value(data):
    return data.isnull().sum()

def check_basic_information(data):
    return data.info()

def check_top_column(data):
    return data.head()

def to_save_data(data):
    return data.to_csv()

#Testing

print(f"The shape of the training data is {check_shape(train_df)}")
print(f"The shape of the testing data is {check_shape(test_df)}")
print(f"The information in the training data is {check_basic_information(train_df)}")
print(f"The information in the testing data is {check_basic_information(test_df)}")
print(f"The missing value in the training data is {check_null_value(train_df)}")
print(f"The missing value in the testing data is {check_null_value(test_df)}")
print(f"The top column in the training data is {check_top_column(train_df)}")
print(f"The top column in the testing data is {check_top_column(test_df)}")



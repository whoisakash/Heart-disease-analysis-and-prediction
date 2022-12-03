import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')
sns.set_style('whitegrid')

'''Load Dataset'''
data = pd.read_csv("Heart disease data.csv")
# print(data)

'''Data Understanding'''
print(data.columns)
# print(data.head())
# print(data.info())
# print(data.describe(include="all"))
# print(data.isnull().sum())
print(data.nunique())

'''Column Selection'''
colm = ['Sex', 'Chest pain type', 'FBS over 120', 'EKG results', 'Exercise angina',
        'Slope of ST', 'Number of vessels fluro', 'Thallium', 'Heart Disease']

'''for Even Columns '''
# colm = ['Age', 'Sex', 'Chest pain type', 'FBS over 120', 'EKG results', 'Exercise angina',
#         'Slope of ST', 'Number of vessels fluro', 'Thallium', 'Heart Disease']

'''Individual plotting chart'''
# print(data["Sex"].value_counts())
# sns.countplot(x=data["Sex"], data=data)
# plt.show()

'''Basic method'''
# for col in colm:
#     sns.countplot(x=data[col], data=data)
#     plt.show()

'''Use For loop for Plotting all chart'''
# i = 0
# while i < 10:
#     fig = plt.figure(figsize=(15, 5))
#     plt.subplot(1, 3, 1)
#     sns.countplot(x=colm[i], data=data)
#     i += 1
#
#     plt.subplot(1, 3, 2)
#     sns.countplot(x=colm[i], data=data)
#     i += 1
#     if i == 9:
#         break
#
#     plt.subplot(1, 3, 3)
#     sns.countplot(x=colm[i], data=data)
#     i += 1
#     plt.show()

"way for even no. column"
# i = 0
# while i < 11:
#     fig = plt.figure(figsize=[20, 4])
#     plt.subplot(1, 2, 1)   #(one row, two plots, first one)
#     sns.countplot(x=colm[i], data=data) # read each one, from Data
#     i += 1
#     if i == 10:
#         break
#     plt.subplot(1, 2, 2)
#     sns.countplot(x=colm[i], data=data) # row one, second plot
#     i += 1
#     plt.show()

'''heatmap of data corr'''
# plt.figure(figsize=(12, 10))
# corr = data.corr()
# sns.heatmap(corr, annot=True, linewidths=0.2, linecolor='black', cmap='afmhot')
# plt.show()

'''Data Split and Scaling'''
print(data.columns)
X = data[['Age', 'Sex', 'Chest pain type', 'BP', 'Cholesterol', 'FBS over 120',
          'EKG results', 'Max HR', 'Exercise angina', 'ST depression',
          'Slope of ST', 'Number of vessels fluro', 'Thallium']]# Independable variable
y = data["Heart Disease"]# dependable variable
# print(X.shape, y.shape)


'''Split train And test data'''
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42529)
# print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

'''Converting y_train and y_test categories to 0's and 1's by replacing'''
train_convert = {"Absence": 0, "Presence": 1}
y_train = y_train.replace(train_convert)

test_convert = {"Absence": 0, "Presence": 1}
y_test = y_test.replace(test_convert)

'''Normalising Data by MinMaxScaler'''
mms = MinMaxScaler()
X_train = mms.fit_transform(X_train)
X_test = mms.fit_transform(X_test)
print(y_test)

'''Use RandomForestClassifer Model'''
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
pred = rf.predict(X_test)

'''Model analysis by metrics function clssification_report, confusion_matrix, Accuracy_score'''
cm = confusion_matrix(y_test, pred)
print(classification_report(y_test, pred))

'''Plot Heatmap of Confusion Matrix'''
sns.heatmap(cm, annot=True, fmt="g", cbar=False, cmap="icefire", linewidths=0.5, linecolor="grey")
plt.title("Confusion Matrix")
plt.ylabel("Actual Values")
plt.xlabel("Predicted Value")
plt.show()

print("Accuracy Score = {}".format(round(accuracy_score(y_test, pred), 5)))
# rfc model , etc = 0.78519
# logi_model = 0.79259
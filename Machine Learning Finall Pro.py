import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# the link of data : https://www.kaggle.com/datasets/arnabchaki/data-science-salaries-2023

dataset = pd.read_csv(r"ds_salaries.csv")
# print('Dataset=', dataset)



print('Data Info=',dataset.info())
print('Data type=\n', dataset.dtypes)
print('Data shape=', dataset.shape)
print('Data Max=\n',dataset.max())
print('Data head=\n', dataset.head(0))
print('Data describe=\n', dataset.describe())
print('data nunique=\n', dataset.nunique())
print('Data dropna shape=', dataset.dropna().shape)
print('Data isnull=\n', dataset.isnull().sum())
print('Dataset duplicated =\n', dataset.duplicated())
print('Dataset duplicated sum =', dataset.duplicated().sum())


d_types = dataset.dtypes
for i in range(dataset.shape[1]):
    if d_types[i] == 'object':
        Pr_data = preprocessing.LabelEncoder()
        dataset[dataset.columns[i]] = Pr_data.fit_transform(dataset[dataset.columns[i]])
print('Data type=', dataset.dtypes)
print(dataset)

scaler = preprocessing.MinMaxScaler()
scaled = scaler.fit_transform(dataset.values)
scaled = pd.DataFrame(scaled, columns=dataset.columns)
print('Data After Scaling:', scaled)


r = dataset.corr()
print(r)


sns.heatmap(r, annot=True)
# plt.show()


sns.catplot(data=dataset, x="experience_level", hue="salary", kind="count")
# plt.show()


sns.scatterplot(x='company_location', y='employee_residence', data=dataset)
plt.xlabel('company_location')
plt.ylabel('employee_residence')
plt.title('scatterplot')
# plt.show()


plt.figure(figsize=(4, 4))
sns.countplot(data=dataset, x="work_year", palette="YlOrBr_r")
# plt.show()


sns.heatmap(dataset.isnull(), cmap='viridis', cbar=False, yticklabels=False)
plt.title('Missing Data')
# plt.show()


sns.displot(dataset['salary'])
# plt.show()


round(dataset["experience_level"].value_counts() / dataset.shape[0] * 100, 2).plot.pie(autopct='%1.1f%%')
plt.title('Pie Chart')
# plt.show()


sns.pairplot(dataset)
# plt.show()


plt.hist(dataset, bins=10)
plt.xlabel('company_location')
plt.ylabel('employee_residence')
plt.title('Histogram')
# plt.show()


sns.boxplot(x='company_location', y='employee_residence', data=dataset)
plt.xlabel('company_location')
plt.ylabel('employee_residence')
plt.title('Box Plot')
# plt.show()

X = scaled.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred =model.predict(X_test)
confusion= confusion_matrix(y_test, y_pred)
print('Confusion Matrix:',confusion)
sns.heatmap(confusion, annot=True)
plt.title("Heatmap Correlation  for Logistic Regression")
plt.show()
recallScore= recall_score(y_test, y_pred,average='micro')
print('Logistic Regression Recall  =', recallScore)
precision= precision_score(y_test, y_pred,average='micro')
print("LogisticRegression Precision Score  =",precision)
accuracy= accuracy_score(y_test, y_pred)
print('LogisticRegression Accuracy =',accuracy)
true_count_lr = np.count_nonzero(y_pred == y_test)
false_count_lr = len(y_pred) - true_count_lr
print("Number of true values in Logistic Regression:", true_count_lr)
print("Number of false values in Logistic Regression:", false_count_lr)

x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.3, random_state=42)
model = SVC(kernel='linear', degree=4)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
confusion= confusion_matrix(y_test, y_pred)
print('Confusion Matrix:\n',confusion)
sns.heatmap(confusion, annot=True)
plt.title(" Heatmap Correlation For SVM Regression")
plt.show()
precision= precision_score(y_true=y_test, y_pred=y_pred,average='micro')
print("Precision Score For SVM =",precision)
recallscore = recall_score(y_test, y_pred,average='micro')
print('Recall Score for SVM  =',recall_score)
accuracy= accuracy_score(y_test, y_pred)
print('Accuracy Score for SVM =', accuracy)
# Logistic Regression


# SVM
true_count_svm = np.count_nonzero(y_pred == y_test)
false_count_svm = len(y_pred) - true_count_svm
print("Number of true values in SVM:", true_count_svm)
print("Number of false values in SVM:", false_count_svm)

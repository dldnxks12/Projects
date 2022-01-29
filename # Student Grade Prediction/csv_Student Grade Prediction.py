'''
G1 , G2 : 1st , 2nd grades
G3 : Target Grade

G1, G2는 G3와 똑같이 Grade이기 때문에 서로 상관관계는 매우 높을 것. 하지만 G3를 예측할 때 사용하지 않는 것이 더 좋을 수 있다고 한다.

Regression을 이용해서 문제를 해결할 것
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import sklearn.metrics as metric


student = pd.read_csv('student-mat.csv')
#print(student.head())


# G3 Grade에서 가장 많은 점수대를 순서대로 plotting
'''
plt.subplots(figsize = (8, 12))
grade_counts = student['G3'].value_counts().sort_values().plot.barh(width = .9, color = sns.color_palette('inferno', 40))
grade_counts.axes.set_title('Number of students who scored a particular grade', fontsize = 20)
grade_counts.set_xlabel("Number of students", fontsize = 15)
grade_counts.set_ylabel("Final Grade", fontsize =15)
plt.show()
'''

# G3 Grade에서 점수대의 분포를 plotting
'''
b = sns.countplot(student['G3'])
b.axes.set_title('Distribution of G3', fontsize = 20)
b.set_xlabel('Final Grade', fontsize = 15)
b.set_ylabel('Count', fontsize = 15)
plt.show()
'''

# dataset에서 null data를 가진 놈들이 있는지 찾아보자
#print(student.isna().sum()) ---- There's no NULL Value


# G3 = 0 제거
# Address feature 제거
# Age > 20 제거
preprocessed_student = student.copy()
preprocessed_student = preprocessed_student.drop(['address'], axis = 1) # Remove Columne of 'address'
preprocessed_student = preprocessed_student[preprocessed_student.age < 20]
preprocessed_student = preprocessed_student[preprocessed_student.G3 != 0]
preprocessed_student = preprocessed_student.drop(['school', 'G1', 'G2'], axis = 1)

# One-hot Encoding (순서가 중요하지 않은 data에 대해) for object types ...
preprocessed_student = pd.get_dummies(preprocessed_student)

# Correlation이 높은 것들 골라서 dataframe 재구성하기
most_correlated = preprocessed_student.corr().abs()['G3'].sort_values(ascending=False)
most_correlated = most_correlated[:9]
student_corr = preprocessed_student.loc[:, most_correlated.index]

# ML Modeling
X = student_corr.drop(['G3'], axis = 1)
y = student_corr['G3']

# plit Dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

lr = LinearRegression()
lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)

MSE = metric.mean_absolute_error(y_test, y_pred)
RMSE = np.sqrt(metric.mean_absolute_error(y_test, y_pred))
r2 = metric.r2_score(y_test, y_pred)

print(MSE)
print(RMSE)
print(r2)

plt.scatter(range(len(y_test)), y_test, color = 'blue')
plt.scatter(range(len(y_pred)), y_pred, color = 'red')
plt.title("Actual vs Prediction")
plt.show()


# 모든 feature 다 사용할 것
# Random forest에서 feature importance를 사용해볼 것
# Scaling 해볼 것
# G1, G2 추가해볼 것
# hypermarameter tuning 할 것

# 과제
'''
1. Scaling 해서 G3 다시 예측해볼 것 
2. Hypermarameter tuning 해보고 영향에 대해 논할 것 
3. G1, G2를 포함시켜보고 이것의 영향에 대해 논할 것 
4. 이걸 해야하나 말아야하나 논할 것
'''



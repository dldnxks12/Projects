import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn.metrics as metric
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler

################################################################################################ Load data

student = pd.read_csv('student-mat.csv')

################################################################################################ Drop data

# G3 Grade에 영향이 적은 feature 제거
preprocessed_student = student.copy()
preprocessed_student = preprocessed_student.drop(['address'], axis = 1)
preprocessed_student = preprocessed_student[preprocessed_student.age < 20]
#preprocessed_student = preprocessed_student.drop(['school', 'G1', 'G2'], axis = 1)
preprocessed_student = preprocessed_student.drop(['school'], axis = 1)

# One-hot Encoding
preprocessed_student = pd.get_dummies(preprocessed_student)

# Correlation이 높은 것들 골라서 dataframe 재구성하기
most_correlated = preprocessed_student.corr().abs()['G3'].sort_values(ascending=False)
most_correlated = most_correlated[:9]
student_corr = preprocessed_student.loc[:, most_correlated.index]


################################################################################################ Scaling

# Before Scaling
#preprocessed_student.plot.kde()
#student_corr.plot.kde()

normal_ = preprocessed_student.copy()
normal__ = student_corr.copy()

# StandardScaler
ss = StandardScaler()
ss2 = StandardScaler()
data = ss.fit_transform(preprocessed_student)
data2 = ss2.fit_transform(student_corr)
preprocessed_student = pd.DataFrame(data, columns=list(preprocessed_student.columns))
student_corr = pd.DataFrame(data2, columns=list(student_corr.columns))

# Minmax Scaler
mm = MinMaxScaler()
mm2 = MinMaxScaler()
data = mm.fit_transform(preprocessed_student)
data2 = mm2.fit_transform(student_corr)
preprocessed_student2 = pd.DataFrame(data, columns=list(preprocessed_student.columns))
student_corr2 = pd.DataFrame(data2, columns=list(student_corr.columns))

# After Scaling
#preprocessed_student.plot.kde()
#preprocessed_student2.plot.kde()
#student_corr2.plot.kde()
#student_corr.plot.kde()
#plt.show()


# Target Value G3 Restore
preprocessed_student['G3'] = student['G3']
preprocessed_student2['G3'] = student['G3']
student_corr['G3'] = student['G3']
student_corr2['G3'] = student['G3']

################################################################################################ Modeling

X = student_corr.drop(['G3'], axis = 1)
y = student_corr['G3']
X2 = preprocessed_student.drop(['G3'], axis = 1)
y2 = preprocessed_student['G3']
X3 = student_corr2.drop(['G3'], axis = 1)
y3 = student_corr2['G3']
X4 = preprocessed_student2.drop(['G3'], axis = 1)
y4 = preprocessed_student2['G3']
X5 = normal_.drop(['G3'], axis = 1)
y5 = normal_['G3']
X6 = normal__.drop(['G3'], axis = 1)
y6 = normal__['G3']

# split Dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size = 0.2)
X_train3, X_test3, y_train3, y_test3 = train_test_split(X3, y3, test_size = 0.2)
X_train4, X_test4, y_train4, y_test4 = train_test_split(X4, y4, test_size = 0.2)
X_train5, X_test5, y_train5, y_test5 = train_test_split(X5, y5, test_size = 0.2)
X_train6, X_test6, y_train6, y_test6 = train_test_split(X6, y6, test_size = 0.2)

# Basic Linear Regression
lr = RandomForestClassifier(n_estimators=500)
lr.fit(X_train, y_train)
lr2 = RandomForestClassifier(n_estimators=500)
lr2.fit(X_train2, y_train2)
lr3 = RandomForestClassifier(n_estimators=500)
lr3.fit(X_train3, y_train3)
lr4 = RandomForestClassifier(n_estimators=500)
lr4.fit(X_train4, y_train4)
lr5 = RandomForestClassifier(n_estimators=500)
lr5.fit(X_train5, y_train5)
lr6 = RandomForestClassifier(n_estimators=500)
lr6.fit(X_train6, y_train6)

################################################################################################ Evaluation

y_pred = lr.predict(X_test)
y_pred2 = lr2.predict(X_test2)
y_pred3 = lr3.predict(X_test3)
y_pred4 = lr4.predict(X_test4)
y_pred5 = lr5.predict(X_test5)
y_pred6 = lr6.predict(X_test6)

MSE = metric.mean_absolute_error(y_test, y_pred)
MSE2 = metric.mean_absolute_error(y_test2, y_pred2)
MSE3 = metric.mean_absolute_error(y_test3, y_pred3)
MSE4 = metric.mean_absolute_error(y_test4, y_pred4)
MSE5 = metric.mean_absolute_error(y_test5, y_pred5)
MSE6 = metric.mean_absolute_error(y_test6, y_pred6)

RMSE = np.sqrt(metric.mean_absolute_error(y_test, y_pred))
RMSE2 = np.sqrt(metric.mean_absolute_error(y_test2, y_pred2))
RMSE3 = np.sqrt(metric.mean_absolute_error(y_test3, y_pred3))
RMSE4 = np.sqrt(metric.mean_absolute_error(y_test4, y_pred4))
RMSE5 = np.sqrt(metric.mean_absolute_error(y_test5, y_pred5))
RMSE6 = np.sqrt(metric.mean_absolute_error(y_test6, y_pred6))

r2 = metric.r2_score(y_test, y_pred)
r2_2 = metric.r2_score(y_test2, y_pred2)
r2_3 = metric.r2_score(y_test3, y_pred3)
r2_4 = metric.r2_score(y_test4, y_pred4)
r2_5 = metric.r2_score(y_test5, y_pred5)
r2_6 = metric.r2_score(y_test6, y_pred6)


print("#----------- Model Evaluation -------------#")
print("MSE Normal")
print(f"STD MSE_Important : {MSE} | MSE_FULL : {MSE2}")
print(f"MINMAX MSE_Important : {MSE3} |  MSE_FULL : {MSE4}")
print(f"Normal MSE_Important : {MSE5} |  MSE_FULL : {MSE6}")
print()

print("RMSE Normal")
print(f"STD RMSE_Important : {RMSE} | RMSE_FULL : {RMSE2}")
print(f"MINMAX RMSE_Important : {RMSE3} |  RMSE_FULL : {RMSE4}")
print(f"Normal RMSE_Important : {RMSE5} |  RMSE_FULL : {RMSE6}")
print()

print("R2 Normal")
print(f"STD R2_Important : {r2} | R2_FULL : {r2_2}")
print(f"MINMAX R2_Important : {r2_3} |  R2_FULL : {r2_4}")
print(f"Normal R2_Important : {r2_5} |  R2_FULL : {r2_6}")

print()

print("#-----------Scores-------------#")

print("Std Scaling")
print("Scores1" , lr.score(X_test, y_test))
print("Scores2" , lr2.score(X_test2, y_test2))
print()
print("MinMax Scaling")
print("Scores3" , lr3.score(X_test3, y_test3))
print("Scores4" , lr4.score(X_test4, y_test4))
print()
print("No Scaling")
print("Scores5" , lr5.score(X_test5, y_test5))
print("Scores6" , lr6.score(X_test6, y_test6))


plt.figure(figsize = (20, 8))
plt.subplot(1,6,1)
plt.scatter(range(len(y_test)), y_test, color = 'blue')
plt.scatter(range(len(y_pred)), y_pred, color = 'red')
plt.title("Std with important Data")
plt.subplot(1,6,2)
plt.scatter(range(len(y_test2)), y_test2, color = 'blue')
plt.scatter(range(len(y_pred2)), y_pred2, color = 'red')
plt.title("Std with Full Data")
plt.subplot(1,6,3)
plt.scatter(range(len(y_test3)), y_test3, color = 'blue')
plt.scatter(range(len(y_pred3)), y_pred3, color = 'red')
plt.title("MinMax with important Data")
plt.subplot(1,6,4)
plt.scatter(range(len(y_test4)), y_test4, color = 'blue')
plt.scatter(range(len(y_pred4)), y_pred4, color = 'red')
plt.title("MinMax with Full Data")
plt.subplot(1,6,5)
plt.scatter(range(len(y_test5)), y_test5, color = 'blue')
plt.scatter(range(len(y_pred5)), y_pred5, color = 'red')
plt.title("Normal with important Data")
plt.subplot(1,6,6)
plt.scatter(range(len(y_test6)), y_test6, color = 'blue')
plt.scatter(range(len(y_pred6)), y_pred6, color = 'red')
plt.title("Normal with Full Data")

plt.show()







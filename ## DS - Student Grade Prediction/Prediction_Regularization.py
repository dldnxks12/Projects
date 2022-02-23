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
preprocessed_student = preprocessed_student[preprocessed_student.G3 != 0]
preprocessed_student = preprocessed_student.drop(['school', 'G1', 'G2'], axis = 1)

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
print(student_corr['G3'].shape)
print(student['G3'].shape)

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

# split Dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size = 0.2)
X_train3, X_test3, y_train3, y_test3 = train_test_split(X3, y3, test_size = 0.2)
X_train4, X_test4, y_train4, y_test4 = train_test_split(X4, y4, test_size = 0.2)

# Basic Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
lr2 = LinearRegression()
lr2.fit(X_train2, y_train2)
lr3 = LinearRegression()
lr3.fit(X_train3, y_train3)
lr4 = LinearRegression()
lr4.fit(X_train4, y_train4)

# Linear Regression with Regularization (Ridge)
lr_ = Ridge(alpha = 0.001)
lr_.fit(X_train, y_train)
lr2_ = Ridge(alpha = 0.001)
lr2_.fit(X_train2, y_train2)
lr3_ = Ridge(alpha = 0.001)
lr3_.fit(X_train3, y_train3)
lr4_ = Ridge(alpha = 0.001)
lr4_.fit(X_train4, y_train4)

# Linear Regression with Regularization (Lasso)
lr__ = Lasso(alpha = 0.001)
lr__.fit(X_train, y_train)
lr2__ = Lasso(alpha = 0.001)
lr2__.fit(X_train2, y_train2)
lr3__ = Lasso(alpha = 0.001)
lr3__.fit(X_train3, y_train3)
lr4__ = Lasso(alpha = 0.001)
lr4__.fit(X_train4, y_train4)

################################################################################################ Evaluation

y_pred = lr.predict(X_test)
y_pred2 = lr2.predict(X_test2)
y_pred3 = lr3.predict(X_test3)
y_pred4 = lr4.predict(X_test4)

# With Regularization - Ridge
y_pred5 = lr_.predict(X_test)
y_pred6 = lr2_.predict(X_test2)
y_pred7 = lr3_.predict(X_test3)
y_pred8 = lr4_.predict(X_test4)

# With Regularization - Lasso
y_pred9 = lr__.predict(X_test)
y_pred10 = lr2__.predict(X_test2)
y_pred11 = lr3__.predict(X_test3)
y_pred12 = lr4__.predict(X_test4)

# Normal
MSE = metric.mean_absolute_error(y_test, y_pred)
MSE2 = metric.mean_absolute_error(y_test2, y_pred2)
MSE3 = metric.mean_absolute_error(y_test3, y_pred3)
MSE4 = metric.mean_absolute_error(y_test4, y_pred4)

# Ridge
MSE5 = metric.mean_absolute_error(y_test, y_pred5)
MSE6 = metric.mean_absolute_error(y_test2, y_pred6)
MSE7 = metric.mean_absolute_error(y_test3, y_pred7)
MSE8 = metric.mean_absolute_error(y_test4, y_pred8)

# Lasso
MSE9 = metric.mean_absolute_error(y_test, y_pred9)
MSE10 = metric.mean_absolute_error(y_test2, y_pred10)
MSE11 = metric.mean_absolute_error(y_test3, y_pred11)
MSE12 = metric.mean_absolute_error(y_test4, y_pred12)

# normal
RMSE = np.sqrt(metric.mean_absolute_error(y_test, y_pred))
RMSE2 = np.sqrt(metric.mean_absolute_error(y_test2, y_pred2))
RMSE3 = np.sqrt(metric.mean_absolute_error(y_test3, y_pred3))
RMSE4 = np.sqrt(metric.mean_absolute_error(y_test4, y_pred4))

# Ridge
RMSE5 = np.sqrt(metric.mean_absolute_error(y_test, y_pred5))
RMSE6 = np.sqrt(metric.mean_absolute_error(y_test2, y_pred6))
RMSE7 = np.sqrt(metric.mean_absolute_error(y_test3, y_pred7))
RMSE8 = np.sqrt(metric.mean_absolute_error(y_test4, y_pred8))

# Lasso
RMSE9 = np.sqrt(metric.mean_absolute_error(y_test, y_pred9))
RMSE10 = np.sqrt(metric.mean_absolute_error(y_test2, y_pred10))
RMSE11 = np.sqrt(metric.mean_absolute_error(y_test3, y_pred11))
RMSE12 = np.sqrt(metric.mean_absolute_error(y_test4, y_pred12))

# normal
r2 = metric.r2_score(y_test, y_pred)
r2_2 = metric.r2_score(y_test2, y_pred2)
r2_3 = metric.r2_score(y_test3, y_pred3)
r2_4 = metric.r2_score(y_test4, y_pred4)

# Ridge
r2_5 = metric.r2_score(y_test, y_pred5)
r2_6 = metric.r2_score(y_test2, y_pred6)
r2_7 = metric.r2_score(y_test3, y_pred7)
r2_8 = metric.r2_score(y_test4, y_pred8)

#Lasso
r2_9 = metric.r2_score(y_test, y_pred9)
r2_10 = metric.r2_score(y_test2, y_pred10)
r2_11 = metric.r2_score(y_test3, y_pred11)
r2_12 = metric.r2_score(y_test4, y_pred12)


print("MSE Normal")
print(f"STD MSE_Important : {MSE} | MSE_FULL : {MSE2}")
print(f"MINMAX MSE_Important : {MSE3} |  MSE_FULL : {MSE4}")

print("MSE Ridge")
print(f"STD MSE_Important : {MSE5} | MSE_FULL : {MSE6}")
print(f"MINMAX MSE_Important : {MSE7} |  MSE_FULL : {MSE8}")

print("MSE Lasso")
print(f"STD MSE_Important : {MSE9} | MSE_FULL : {MSE10}")
print(f"MINMAX MSE_Important : {MSE11} |  MSE_FULL : {MSE12}")
print()

print("RMSE Normal")
print(f"STD RMSE_Important : {RMSE} | RMSE_FULL : {RMSE2}")
print(f"MINMAX RMSE_Important : {RMSE3} |  RMSE_FULL : {RMSE4}")

print("RMSE Ridge")
print(f"STD RMSE_Important : {RMSE5} | RMSE_FULL : {RMSE6}")
print(f"MINMAX RMSE_Important : {RMSE7} |  RMSE_FULL : {RMSE8}")

print("RMSE Lasso")
print(f"STD RMSE_Important : {RMSE9} | RMSE_FULL : {RMSE10}")
print(f"MINMAX RMSE_Important : {RMSE11} |  RMSE_FULL : {RMSE12}")
print()

print("R2 Normal")
print(f"STD R2_Important : {r2} | R2_FULL : {r2_2}")
print(f"MINMAX R2_Important : {r2_3} |  R2_FULL : {r2_4}")

print("R2 Ridge")
print(f"STD R2_Important : {r2_5} | R2_FULL : {r2_6}")
print(f"MINMAX R2_Important : {r2_7} |  R2_FULL : {r2_8}")

print("R2 Lasso")
print(f"STD R2_Important : {r2_9} | R2_FULL : {r2_10}")
print(f"MINMAX R2_Important : {r2_11} |  R2_FULL : {r2_12}")

plt.figure(figsize = (32, 8))
plt.subplot(1,12,1)
plt.scatter(range(len(y_test)), y_test, color = 'blue')
plt.scatter(range(len(y_pred)), y_pred, color = 'red')
plt.title("Std with important Data")
plt.subplot(1,12,2)
plt.scatter(range(len(y_test2)), y_test2, color = 'blue')
plt.scatter(range(len(y_pred2)), y_pred2, color = 'red')
plt.title("Std with Full Data")
plt.subplot(1,12,3)
plt.scatter(range(len(y_test3)), y_test3, color = 'blue')
plt.scatter(range(len(y_pred3)), y_pred3, color = 'red')
plt.title("MinMax with important Data")
plt.subplot(1,12,4)
plt.scatter(range(len(y_test4)), y_test4, color = 'blue')
plt.scatter(range(len(y_pred4)), y_pred4, color = 'red')
plt.title("MinMax with Full Data")


plt.subplot(1,12,5)
plt.scatter(range(len(y_test)), y_test, color = 'blue')
plt.scatter(range(len(y_pred5)), y_pred5, color = 'red')
plt.title("Std with important Data")
plt.subplot(1,12,6)
plt.scatter(range(len(y_test2)), y_test2, color = 'blue')
plt.scatter(range(len(y_pred6)), y_pred6, color = 'red')
plt.title("Std with Full Data")
plt.subplot(1,12,7)
plt.scatter(range(len(y_test3)), y_test3, color = 'blue')
plt.scatter(range(len(y_pred7)), y_pred7, color = 'red')
plt.title("MinMax with important Data")
plt.subplot(1,12,8)
plt.scatter(range(len(y_test4)), y_test4, color = 'blue')
plt.scatter(range(len(y_pred8)), y_pred8, color = 'red')
plt.title("MinMax with Full Data")

plt.subplot(1,12,9)
plt.scatter(range(len(y_test)), y_test, color = 'blue')
plt.scatter(range(len(y_pred9)), y_pred9, color = 'red')
plt.title("Ridge with important Data")
plt.subplot(1,12,10)
plt.scatter(range(len(y_test2)), y_test2, color = 'blue')
plt.scatter(range(len(y_pred10)), y_pred10, color = 'red')
plt.title("Ridge with Full Data")
plt.subplot(1,12,11)
plt.scatter(range(len(y_test3)), y_test3, color = 'blue')
plt.scatter(range(len(y_pred11)), y_pred11, color = 'red')
plt.title("Lasso with important Data")
plt.subplot(1,12,12)
plt.scatter(range(len(y_test4)), y_test4, color = 'blue')
plt.scatter(range(len(y_pred12)), y_pred12, color = 'red')
plt.title("Lasso with Full Data")

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




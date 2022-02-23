import random
import pandas as pd
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.integrate import cumtrapz
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

# data load
train_features =pd.read_csv("C:/Users/USER/Desktop/Hackerton/train_features.csv")
train_labels = pd.read_csv("C:/Users/USER/Desktop/Hackerton/train_labels.csv")
test = pd.read_csv("C:/Users/USER/Desktop/Hackerton/test_features.csv")
submission = pd.read_csv('C:/Users/USER/Desktop/Hackerton/sample_submission.csv')

# ---------------------------- data 분할 ---------------------------- #
# x_train acc  / x_train gy
# x_test acc / x_test gy

act_list = train_features.iloc[:,2:].columns
acc_list = ['acc_x','acc_y','acc_z']
gy_list = ['gy_x','gy_y','gy_z']

# data Scaling - like normalize
scaler = StandardScaler()
train_features[act_list] = scaler.fit_transform(train_features[act_list])
test[act_list] = scaler.transform(test[act_list])

print("데이터 분할 1")

# acc data와 gyro data 분리 , 각각 다르게 데이터 processing하기 위함
def sensor_split(data):
    X_acc = []
    X_gy = []

    for i in tqdm(data['id'].unique()):
        temp_acc = np.array(data[data['id'] == i].loc[:, acc_list])
        temp_gy = np.array(data[data['id'] == i].loc[:, gy_list])
        X_acc.append(temp_acc)
        X_gy.append(temp_gy)

    X_acc = np.array(X_acc).reshape(-1, 600, 3)
    X_gy = np.array(X_gy).reshape(-1, 600, 3)

    return X_acc, X_gy

# data 분리
X_train_acc, X_train_gy = sensor_split(train_features)
X_test_acc, X_test_gy = sensor_split(test)

print("데이터 분할 2")

# --------------------------  data processing -------------------------- #

# Time Warping
sigma = 0.2
knot = 4

def GenerateRandomCurves(X, sigma=0.2, knot=4):
    xx = (np.ones((X.shape[1],1))*(np.arange(0,X.shape[0], (X.shape[0]-1)/(knot+1)))).transpose()
    yy = np.random.normal(loc=1.0, scale=sigma, size=(knot+2, X.shape[1]))
    x_range = np.arange(X.shape[0])
    cs_x = CubicSpline(xx[:,0], yy[:,0])
    cs_y = CubicSpline(xx[:,1], yy[:,1])
    cs_z = CubicSpline(xx[:,2], yy[:,2])
    return np.array([cs_x(x_range),cs_y(x_range),cs_z(x_range)]).transpose()

def DistortTimesteps(X, sigma):
    tt = GenerateRandomCurves(X, sigma) # Regard these samples aroun 1 as time intervals
    tt_cum = np.cumsum(tt, axis=0)        # Add intervals to make a cumulative graph
    # Make the last value to have X.shape[0]
    t_scale = [(X.shape[0]-1)/tt_cum[-1,0],(X.shape[0]-1)/tt_cum[-1,1],(X.shape[0]-1)/tt_cum[-1,2]]
    tt_cum[:,0] = tt_cum[:,0]*t_scale[0]
    tt_cum[:,1] = tt_cum[:,1]*t_scale[1]
    tt_cum[:,2] = tt_cum[:,2]*t_scale[2]
    return tt_cum

def TimeWarp(X, sigma):
    tt_new = DistortTimesteps(X, sigma)
    X_new = np.zeros(X.shape)
    x_range = np.arange(X.shape[0])
    X_new[:,0] = np.interp(x_range, tt_new[:,0], X[:,0])
    X_new[:,1] = np.interp(x_range, tt_new[:,1], X[:,1])
    X_new[:,2] = np.interp(x_range, tt_new[:,2], X[:,2])
    return X_new

# Permutation
def Permutation(X, nPerm=4, minSegLength=10):
    X_new = np.zeros(X.shape)
    idx = np.random.permutation(nPerm)
    bWhile = True
    while bWhile == True:
        segs = np.zeros(nPerm+1, dtype=int)
        segs[1:-1] = np.sort(np.random.randint(minSegLength, X.shape[0]-minSegLength, nPerm-1))
        segs[-1] = X.shape[0]
        if np.min(segs[1:]-segs[0:-1]) > minSegLength:
            bWhile = False
    pp = 0
    for ii in range(nPerm):
        x_temp = X[segs[idx[ii]]:segs[idx[ii]+1],:]
        X_new[pp:pp+len(x_temp),:] = x_temp
        pp += len(x_temp)
    return (X_new)

# Rotation
'''
def Rotation(X):
    axis = np.random.uniform(low=-1, high=1, size=X.shape[1])
    angle = np.random.uniform(low=-np.pi, high=np.pi)
    return np.matmul(X , axangle2mat(axis,angle))
'''

# Jittering
def Jitter(X, sigma=0.05):
    myNoise = np.random.normal(loc=0, scale=sigma, size=X.shape)
    return X+myNoise

# Magnitude Warping
def MagWarp(X, sigma):
    return X * GenerateRandomCurves(X, sigma)

def ts_aug(data, method,sigma):
  new_data=[]
  for i in range(data.shape[0]):
    temp=list(method(data[i], sigma))
    new_data.append(temp)
  return np.array(new_data)

print("데이터 처리 1")

# Iterator method for all seperated data
X_train_acc_comb = Permutation(Jitter(X_train_acc))
X_train_acc_perm = Permutation(X_train_acc)
X_train_acc_timew = ts_aug(X_train_acc, TimeWarp, 0.2)
X_train_acc_magw = ts_aug(X_train_acc, MagWarp, 0.2)

X_train_gy_comb = Permutation(Jitter(X_train_gy))
X_train_gy_perm = Permutation(X_train_gy)
X_train_gy_timew = ts_aug(X_train_gy, TimeWarp, 0.2)
X_train_gy_magw = ts_aug(X_train_gy, MagWarp, 0.2)

print("데이터 처리 2")

# Appending data
X_train_acc = np.append(X_train_acc, X_train_acc_comb, axis=0)
X_train_acc = np.append(X_train_acc, X_train_acc_perm, axis=0)
X_train_acc = np.append(X_train_acc, X_train_acc_timew, axis=0)
X_train_acc = np.append(X_train_acc, X_train_acc_magw, axis=0)

X_train_gy = np.append(X_train_gy, X_train_gy_comb, axis=0)
X_train_gy = np.append(X_train_gy, X_train_gy_perm, axis=0)
X_train_gy = np.append(X_train_gy, X_train_gy_timew, axis=0)
X_train_gy = np.append(X_train_gy, X_train_gy_magw, axis=0)

print("데이터 처리 3")

'''
for si in x_train_acc:
    for e in si
        e

print(np.shape(acc)) : (9375000, 3)        
'''

acc = [e for sl in X_train_acc for e in sl]
df_report = np.stack(acc, axis = 0)
df_acc = pd.DataFrame(df_report, columns= ['acc_x', 'acc_y', 'acc_z'])

print("데이터 처리 4")
print(np.shape(df_report))
print(np.shape(df_acc))


gy = [e for sl in X_train_gy for e in sl]
df_report = np.stack(gy, axis = 0)
df_gy = pd.DataFrame(df_report, columns= ['gy_x', 'gy_y', 'gy_z'])

print("데이터 처리 5")

# ---------------- 데이터 병합 ------------------ #

df_aug_result = pd.concat([df_acc, df_gy], axis = 1)

print("데이터 병합 1")
print(df_aug_result.shape)

df_aug_result.insert(0, 'id', 0)
df_aug_result.insert(1, 'time', 1)

print(type(df_aug_result))
print(np.shape(df_aug_result))

print("데이터 병합 2")

list600 = [n for n in range(600)]

for i in range(int(len(df_aug_result)/600)):
  df_aug_result.loc[600*i:600*i+599, 'id'] = i              # id값 설정
  df_aug_result.loc[600*i:600*i+599, 'time'] = list600      # 리스트값 time열에 붙여넣기
  print(i)

print("데이터 병합 3")

print(np.shape(df_aug_result))

df_aug_result.to_csv('js_train_data.csv', index=False)


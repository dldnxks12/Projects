import cv2
import sys
import glob
import pickle
import numpy as np

# Itertaion Maximum 30, moves under 0.001 pixel -> Quit !! [ http://www.gisdeveloper.co.kr/?p=7123 ] 
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Grid 
objp = np.zeros((6*9, 3), dtype = np.float32)
objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1,2)

objpoints = []  # 3D Points in Real world space
imgpoints = []  # 2D Points in Image Plane

# Get image
images = glob.glob('TrainChess*.png')

for img in images:
    img = cv2.imread(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find Chess board corners
    ret,corners = cv2.findChessboardCorners(gray, (9,6), None)

    if ret == True:
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)
        img = cv2.drawChessboardCorners(img, (9,6), corners2, ret)
        cv2.imwrite("img.png",img)    
        cv2.imshow("img",img)
        #cv2.waitKey(10)

cv2.destroyAllWindows()
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

'''

# Save Parameters # 

np.save("mtx_file", mtx)
np.save("dist_file", dist)

obj_file   = open("objpoints.pkl", "wb")
img_file   = open("imgpoints.pkl", "wb")
rvecs_file = open("rvecs.pkl", "wb")
tvecs_file = open("tvecs.pkl", "wb")

pickle.dump(objpoints, obj_file)
pickle.dump(imgpoints, img_file)
pickle.dump(rvecs, rvecs_file)
pickle.dump(tvecs, tvecs_file)

obj_file.close()
img_file.close()
rvecs_file.close()
tvecs_file.close()


print("")
print("obj",   type(objpoints))      # List
print("img",   type(imgpoints))      # List
print("ret", ret, type(ret))         # class : float
print("mtx",   type(mtx))            # numpy
print("dist",  type(dist))           # numpy 
print("rvecs", type(rvecs))          # tuple
print("tvecs", type(tvecs))          # tuple 
print("")

'''

img = cv2.imread("test4.png")    

h, w = img.shape[:2]
newcameramtx , roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
x, y, w, h = roi
dst = dst[y:y+h , x:x+w]
cv2.imwrite("Result_Image2.png", dst)

tot_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
    tot_error += error
print("total error: ", tot_error/len(objpoints))

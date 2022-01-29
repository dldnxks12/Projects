# 각 Class 내의 process_lidar 함수는 2d lidar에서 센서값을 받아서 수행된다.
# 해당 함수의 return 값은 1.차량의 속도 2. 조향각
# 차량은 Ackerman Steering 으로 구현

# Lidar range : -45 ~ 225도까지 값 받아오고, 순서대로 리스트에 저장됨 (0 ~ 35m 까지 측정, 분해능: 0.25도  , list 길이는 1080)

# Steering은 -90 ~ 90도 까지 가능

import numpy as np

# 차량 속도 5로 방향을 바꾸지 않고 걍 돌진
# drives straight ahead at a speed of 5
class SimpleDriver:

    def process_lidar(self, ranges):  # 2D Lidar의 센서 값(ranges)을 받는다.
        speed = 5.0
        steering_angle = 0.0  # 방향을 바꾸지 않고 걍 돌진
        return speed, steering_angle  # return value : 차량 목표 속도 , 방향 각




test_lidar = np.random.randint(0, 35, size = 1080)

print(test_lidar.shape)

test_driver = AnotherDriver()

test_driver.process_lidar(test_lidar)


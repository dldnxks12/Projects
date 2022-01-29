

# Algorithm 2
# drives toward the furthest point it sees
class AnotherDriver:

    def process_lidar(self, ranges):
        # the number of LiDAR points
        NUM_RANGES = len(ranges)

        print("Range Length : ", NUM_RANGES) # 1080

        # angle between each LiDAR point  --- 2*PI / 1080 --- 360도를 1080으로 나눈 것
        # --------------------------------------------------우리가 사용하는 lidar는 -45 ~ 225라고 했는데 왜 이렇게 할까?
        ANGLE_BETWEEN = 2 * np.pi / NUM_RANGES
        print("ANGLE_BETWEEN", ANGLE_BETWEEN) # 0.005817764173314432
        # number of points in each quadrant --- quadrant : 사분면
        # 즉, 사분면 내에 존재하는 각 Lidar data들
        NUM_PER_QUADRANT = NUM_RANGES // 4
        print("NUM_PER_QUADRANT", NUM_PER_QUADRANT)

        # the index of the furthest LiDAR point (ignoring the points behind the car)
        # 가장 멀리 있는 data의 index 찾기

        # 내 앞의 두 공간에 대해서만 최대 거리 찾기 (뒤에 있는 장소는 무시)
        result1 = np.argmax(ranges[NUM_PER_QUADRANT:-NUM_PER_QUADRANT])
        print("result1",result1)
        # offset 으로 한 사분면에 대한 index를 더해주어야 함
        max_idx = result1 + NUM_PER_QUADRANT
        print("max_idx", max_idx) # 284

        # some math to get the steering angle to correspond to the chosen LiDAR point
        steering_angle = max_idx * ANGLE_BETWEEN - (NUM_RANGES // 2) * ANGLE_BETWEEN
        speed = 5.0

        return speed, steering_angle


test_lidar = np.random.randint(0, 35, size = 1080)
#print(test_lidar.shape)

test_driver = AnotherDriver()
test_driver.process_lidar(test_lidar)


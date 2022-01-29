# 84s

import numpy as np


class CustomDriver:
    BUBBLE_RADIUS = 120  # 위험 방울 범위 160 -> 200

    PREPROCESS_CONV_SIZE = 3  # 이동 평균 Window
    BEST_POINT_CONV_SIZE = 80  # 50 사이즈의 Window로 평균내기 이 결과로 나온 값들 중에서 제일 좋은 포인트 찾아 따라갈 것

    MAX_LIDAR_DIST = 3000000  # inf로 바꾸는 부분이겠지

    STRAIGHTS_STEERING_ANGLE = (np.pi / 18) * 0.8  # 10 degrees

    STRAIGHTS_SPEED = 8.0  # 직선 Speed
    CORNERS_SPEED = 6.0  # 코너링 Speed
    MOMENTUM = 0.5  # 관성 모멘텀

    MOMENTUM_INDEX = 0

    def __init__(self):
        self.radians_per_elem = None  # 분해각

    def preprocess_lidar(self, ranges):

        self.radians_per_elem = (2 * np.pi) / len(ranges)

        proc_ranges = np.array(ranges[180:-180])  # 이 범위 내에서만 탐지할 것  ---- 더 넓힐지 말지 고민

        # print("분해능", self.radians_per_elem)
        # print("ranges",len(ranges))
        # print("proc_ranges",len(proc_ranges))

        # print(f"Before MAF : {proc_ranges[0]}")
        # MAF
        proc_ranges = np.convolve(proc_ranges, np.ones(self.PREPROCESS_CONV_SIZE), 'same') / self.PREPROCESS_CONV_SIZE
        # print(f"After MAF : {proc_ranges[0]}")

        # distance의 최대 , 최소 Boundary setting
        proc_ranges = np.clip(proc_ranges, 0, self.MAX_LIDAR_DIST)

        return proc_ranges

    def find_max_gap(self, free_space_ranges):

        masked = np.ma.masked_where(free_space_ranges == 0, free_space_ranges)
        slices = np.ma.notmasked_contiguous(masked)

        max_len = slices[0].stop - slices[0].start
        chosen_slice = slices[0]

        for sl in slices[1:]:
            sl_len = sl.stop - sl.start
            if sl_len > max_len:
                max_len = sl_len
                chosen_slice = sl

        return chosen_slice.start, chosen_slice.stop  # Max Free Space의 시작점과 끝점

    # 수정해볼 부분 1 --- 최적 포인트 골라내기
    def find_best_point(self, start, end, ranges):
        avg_max_gap = np.convolve(ranges[start:end], np.ones(self.BEST_POINT_CONV_SIZE),
                                  'same') / self.BEST_POINT_CONV_SIZE

        return avg_max_gap.argmax() + start  # 가장 먼 Point return

    # 수정해볼 부분 2 --- 조향각도
    def get_angle(self, range_index, range_len):

        lidar_angle = (range_index - (range_len / 2)) * self.radians_per_elem

        steering_angle = lidar_angle / 2  # Furthest Drive와는 다르게 1/2 배를 해주었다. 이렇게 하면 차가 지그재그로 막 움직이는게 좀 덜하다.

        return steering_angle

    def process_lidar(self, ranges):

        proc_ranges = self.preprocess_lidar(ranges)
        closest = proc_ranges.argmin()

        # make bubble
        min_index = closest - self.BUBBLE_RADIUS
        max_index = closest + self.BUBBLE_RADIUS

        if min_index < 0:
            min_index = 0
        if max_index > len(proc_ranges):
            max_index = len(proc_ranges) - 1

        proc_ranges[min_index:max_index] = 0

        gap_start, gap_end = self.find_max_gap(proc_ranges)

        best = self.find_best_point(gap_start, gap_end, proc_ranges)

        steering_angle = self.get_angle(best, len(proc_ranges))

        # STRAIGHTS_STEERING_ANGLE를 작게 줘보면?

        if abs(steering_angle) > self.STRAIGHTS_STEERING_ANGLE:
            speed = self.CORNERS_SPEED
            self.MOMENTUM_INDEX = 0
            if proc_ranges[best] < 5:
                speed = 5
        else:
            if proc_ranges[best] > 25:
                speed = self.STRAIGHTS_SPEED + (self.MOMENTUM * self.MOMENTUM_INDEX)
                if self.MOMENTUM_INDEX <= 20:
                    self.MOMENTUM_INDEX += 2
                else:
                    self.MOMENTUM_INDEX = 20

            elif proc_ranges[best] > 15:
                speed = self.STRAIGHTS_SPEED + (self.MOMENTUM * self.MOMENTUM_INDEX)
                if self.MOMENTUM_INDEX <= 10:
                    self.MOMENTUM_INDEX += 1
                else:
                    self.MOMENTUM_INDEX = 10

            else:
                speed = self.STRAIGHTS_SPEED + (self.MOMENTUM * self.MOMENTUM_INDEX)
                if self.MOMENTUM_INDEX > 0:
                    self.MOMENTUM_INDEX -= 1
                else:
                    self.MOMENTUM_INDEX = 0

        print(f"D : {proc_ranges[best]} MOMENTUM_INDEX {self.MOMENTUM_INDEX}")
        print(f'Steering Angle in Degree : {(steering_angle / (np.pi / 2)) * 90}')

        return speed, steering_angle


class GapFollower:
    BUBBLE_RADIUS = 160

    PREPROCESS_CONV_SIZE = 3
    BEST_POINT_CONV_SIZE = 80

    MAX_LIDAR_DIST = 3000000

    STRAIGHTS_SPEED = 8.0
    CORNERS_SPEED = 5.0
    STRAIGHTS_STEERING_ANGLE = np.pi / 18  # 10 degrees

    def __init__(self):
        # used when calculating the angles of the LiDAR data
        self.radians_per_elem = None

    def preprocess_lidar(self, ranges):
        """ Preprocess the LiDAR scan array. Expert implementation includes:
            1.Setting each value to the mean over some window
            2.Rejecting high values (eg. > 3m)
        """
        self.radians_per_elem = (2 * np.pi) / len(ranges)
        # we won't use the LiDAR data from directly behind us
        proc_ranges = np.array(ranges[135:-135])
        # sets each value to the mean over a given window
        proc_ranges = np.convolve(proc_ranges, np.ones(self.PREPROCESS_CONV_SIZE), 'same') / self.PREPROCESS_CONV_SIZE
        proc_ranges = np.clip(proc_ranges, 0, self.MAX_LIDAR_DIST)
        return proc_ranges

    def find_max_gap(self, free_space_ranges):
        """ Return the start index & end index of the max gap in free_space_ranges
            free_space_ranges: list of LiDAR data which contains a 'bubble' of zeros
        """
        # mask the bubble
        masked = np.ma.masked_where(free_space_ranges == 0, free_space_ranges)
        # get a slice for each contigous sequence of non-bubble data
        slices = np.ma.notmasked_contiguous(masked)
        max_len = slices[0].stop - slices[0].start
        chosen_slice = slices[0]
        # I think we will only ever have a maximum of 2 slices but will handle an
        # indefinitely sized list for portablility
        for sl in slices[1:]:
            sl_len = sl.stop - sl.start
            if sl_len > max_len:
                max_len = sl_len
                chosen_slice = sl
        return chosen_slice.start, chosen_slice.stop

    def find_best_point(self, start_i, end_i, ranges):
        """Start_i & end_i are start and end indices of max-gap range, respectively
        Return index of best point in ranges
        Naive: Choose the furthest point within ranges and go there
        """
        # do a sliding window average over the data in the max gap, this will
        # help the car to avoid hitting corners
        averaged_max_gap = np.convolve(ranges[start_i:end_i], np.ones(self.BEST_POINT_CONV_SIZE),
                                       'same') / self.BEST_POINT_CONV_SIZE
        return averaged_max_gap.argmax() + start_i

    def get_angle(self, range_index, range_len):
        """ Get the angle of a particular element in the LiDAR data and transform it into an appropriate steering angle
        """
        lidar_angle = (range_index - (range_len / 2)) * self.radians_per_elem
        steering_angle = lidar_angle / 2
        return steering_angle

    def process_lidar(self, ranges):
        """ Process each LiDAR scan as per the Follow Gap algorithm & publish an AckermannDriveStamped Message
        """
        proc_ranges = self.preprocess_lidar(ranges)
        # Find closest point to LiDAR
        closest = proc_ranges.argmin()

        # Eliminate all points inside 'bubble' (set them to zero)
        min_index = closest - self.BUBBLE_RADIUS
        max_index = closest + self.BUBBLE_RADIUS
        if min_index < 0: min_index = 0
        if max_index >= len(proc_ranges): max_index = len(proc_ranges) - 1
        proc_ranges[min_index:max_index] = 0

        # Find max length gap
        gap_start, gap_end = self.find_max_gap(proc_ranges)

        # Find the best point in the gap
        best = self.find_best_point(gap_start, gap_end, proc_ranges)

        # Publish Drive message
        steering_angle = self.get_angle(best, len(proc_ranges))
        if abs(steering_angle) > self.STRAIGHTS_STEERING_ANGLE:
            speed = self.CORNERS_SPEED
        else:
            speed = self.STRAIGHTS_SPEED
        print('Steering angle in degrees: {}'.format((steering_angle / (np.pi / 2)) * 90))
        return speed, steering_angle


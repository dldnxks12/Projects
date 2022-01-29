import numpy as np

class CustomDriver:
    BUBBLE_RADIUS = 20

    PREPROCESS_CONV_SIZE = 3
    BEST_POINT_CONV_SIZE = 90
    MAX_LIDAR_DIST = 3000000

    STRAIGHTS_STEERING_ANGLE = (np.pi / 18)  # 10 degrees

    MOVING_LIST = []
    movingAV = 0

    MOVING_LIST2 = []
    movingAV2 = 0

    TIME = 0

    BEFORE = 0
    AFTER = 0

    BEFORE_DISTANCE = 0
    BEFORE_IDX = 0
    # WEIGHT
    b = 0.4

    def __init__(self):
        self.radians_per_elem = None  # 분해각

    def preprocess_lidar(self, ranges):

        self.radians_per_elem = (2 * np.pi) / len(ranges)

        proc_ranges = np.array(ranges[135:-135])  # 이 범위 내에서만 탐지할 것  ---- 더 넓힐지 말지 고민
        proc_ranges = np.convolve(proc_ranges, np.ones(self.PREPROCESS_CONV_SIZE), 'same') / self.PREPROCESS_CONV_SIZE

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

    def find_best_point(self, start, end, ranges):
        avg_max_gap = np.convolve(ranges[start:end], np.ones(self.BEST_POINT_CONV_SIZE),
                                  'same') / self.BEST_POINT_CONV_SIZE

        return avg_max_gap.argmax() + start  # 가장 먼 Point return

    def get_angle(self, range_index, range_len):

        lidar_angle = (range_index - (range_len / 2)) * self.radians_per_elem
        steering_angle = lidar_angle / 2  # Furthest Drive와는 다르게 1/2 배를 해주었다. 이렇게 하면 차가 지그재그로 막 움직이는게 좀 덜하다.

        return steering_angle

    def DISTANCE_TEMP(self, current):

        WINSIZE = 30
        self.MOVING_LIST.append(current)
        if len(self.MOVING_LIST) <= WINSIZE:
            self.movingAV = np.sum(self.MOVING_LIST) / len(self.MOVING_LIST)
            return 0, self.movingAV

        self.BEFORE = self.movingAV
        self.movingAV = self.movingAV - (self.MOVING_LIST[0] / WINSIZE) + (current / WINSIZE)
        self.AFTER = self.movingAV

        del self.MOVING_LIST[0]

        DIFF = self.AFTER - self.BEFORE
        return DIFF, self.movingAV

    def LINE_TEMP(self, LINE):

        WINSIZE = 15
        self.MOVING_LIST2.append(LINE)
        if len(self.MOVING_LIST2) <= WINSIZE:
            self.movingAV2 = np.sum(self.MOVING_LIST2) / len(self.MOVING_LIST2)
            return self.movingAV2

        self.movingAV2 = self.movingAV2 - (self.MOVING_LIST2[0] / WINSIZE) + (LINE / WINSIZE)

        del self.MOVING_LIST2[0]

        return self.movingAV2

    def process_lidar(self, ranges):

        proc_ranges = self.preprocess_lidar(ranges)
        left_ = proc_ranges[60:160].sum() / len(proc_ranges[60:160])
        right_ = proc_ranges[-160:-60].sum()/ len(proc_ranges[-160:-60])
        LINE = np.sqrt(abs(left_ - right_))

        closest = proc_ranges.argmin()

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

        # Weights
        a = 5.0
        c = 0.005
        d = 0.01 # --------------------------- d 또한 조금 더 변화시켜보자 line 차이가 threshold 이상이라면 +의 값 차이가 threshold 이상이라면 -의 값으로

        DIFF, distance = self.DISTANCE_TEMP(proc_ranges[best])

        # distance Gap 이 너무 크면 평균을 내라 ...
        if abs(DIFF) > 0.2 and steering_angle > 0.04:
            steering_angle = steering_angle * 0.8

        if DIFF >= 0:
            self.TIME += 2.5
            self.b += 0.003
            if self.b >= 0.4:
                self.b = 0.4
            if self.TIME > 600:
                self.TIME = 600
        else:
            self.TIME -= 2
            self.b -= 0.0015
            if self.b <= 0.35:
                self.b = 0.35
            if self.TIME < 250:
                self.TIME = 250

        speed = (-a * abs(steering_angle)) + (self.b * distance) + (c * self.TIME) + (-d * LINE) + 6

        print(
            f'Angle : {steering_angle : .3f} | GAP : {DIFF : .3f} | D : {distance : .3f} | T : {self.TIME} | LINEGAP : {LINE : .5f}',
            end='\r')

        return speed, steering_angle

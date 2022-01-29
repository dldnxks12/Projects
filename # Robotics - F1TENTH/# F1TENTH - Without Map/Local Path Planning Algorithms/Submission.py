# 67 - 76 

import numpy as np

class CustomDriver:
    BUBBLE_RADIUS = 20
    PREPROCESS_CONV_SIZE = 3
    BEST_POINT_CONV_SIZE = 85
    MAX_LIDAR_DIST = 3000000
    STRAIGHTS_STEERING_ANGLE = (np.pi / 18)  # 10 degrees

    # For MAF filters
    MOVING_LIST = []
    MOVING_LIST2 = []
    MOVING_LIST3 = []
    movingAV = 0
    movingAV2 = 0
    movingAV3 = 0

    # For DIFFERENCE
    BEFORE = 0
    BEFORE2 = 0
    AFTER = 0
    AFTER2 = 0

    # WEIGHT
    a = 4.0
    b = 0.4
    c = 0.005

    TIME = 0

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
        return avg_max_gap.argmax() + start

    def get_angle(self, range_index, range_len):
        lidar_angle = (range_index - (range_len / 2)) * self.radians_per_elem
        steering_angle = lidar_angle / 2
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

    def Speed_D(self, speed):
        WINSIZE = 15
        self.MOVING_LIST3.append(speed)
        if len(self.MOVING_LIST3) <= WINSIZE:
            self.movingAV3 = np.sum(self.MOVING_LIST3) / len(self.MOVING_LIST3)
            return self.movingAV3

        self.BEFORE2 = self.movingAV3
        self.movingAV3 = self.movingAV3 - (self.MOVING_LIST3[0] / WINSIZE) + (speed / WINSIZE)
        self.AFTER2 = self.movingAV3
        del self.MOVING_LIST3[0]

        return self.BEFORE2 - self.AFTER2


    def process_lidar(self, ranges):
        proc_ranges = self.preprocess_lidar(ranges)
        left_ = proc_ranges[120]
        right_ = proc_ranges[-120]
        LINE = abs(left_ - right_)
        LINE = self.LINE_TEMP(LINE)

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

        if LINE <= 0.5:
            d = -0.2
        else:
            d = 0.2

        DIFF, distance = self.DISTANCE_TEMP(proc_ranges[best])

        if DIFF >= 0:
            self.TIME += 2.5
            self.b += 0.003
            if self.b >= 0.4:
                self.b = 0.4
            if self.TIME > 650:
                self.TIME = 650
        else:
            self.TIME -= 2
            self.b -= 0.0015
            if self.b <= 0.35:
                self.b = 0.35
            if self.TIME < 300:
                self.TIME = 300

        # 최대한 선형적으로 만들어야 가감속에 의한 Jittering이 적다.
        speed = (-self.a * abs(steering_angle)) + (self.b * distance) + (self.c * self.TIME) + (-d * LINE) + 6

        Speed_DIFF = self.Speed_D(speed)*1.5
        speed += Speed_DIFF
        print(
            f'S : {speed : .3f} | D : {distance : .3f} | B Value : {self.b : .3f} | Speed_DIFF : {Speed_DIFF : .3f} | LINEGAP : {LINE : .5f}',
            end='\r')

        return speed, steering_angle

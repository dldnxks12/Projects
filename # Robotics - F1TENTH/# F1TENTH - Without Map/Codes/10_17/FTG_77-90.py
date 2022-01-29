import numpy as np

class CustomDriver:
    BUBBLE_RADIUS = 60  # 위험 방울 범위

    PREPROCESS_CONV_SIZE = 3
    BEST_POINT_CONV_SIZE = 200

    MAX_LIDAR_DIST = 3000000  # inf로 바꾸는 부분이겠지

    STRAIGHTS_STEERING_ANGLE = (np.pi / 18)  # 10 degrees

    MOVING_LIST = []
    MOVING_LIST2 = []
    movingAV = 0
    movingAV2 = 0
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

        return avg_max_gap.argmax() + start  # 가장 먼 Point return


    def get_angle(self, range_index, range_len):

        lidar_angle = (range_index - (range_len / 2)) * self.radians_per_elem
        steering_angle = lidar_angle / 2  # Furthest Drive와는 다르게 1/2 배를 해주었다. 이렇게 하면 차가 지그재그로 막 움직이는게 좀 덜하다.

        return steering_angle

    def temp(self, current):

        WINSIZE = 20
        self.MOVING_LIST.append(current)
        if len(self.MOVING_LIST) <= WINSIZE:
            self.movingAV = np.sum(self.MOVING_LIST) / len(self.MOVING_LIST)
            return  self.movingAV

        self.movingAV = self.movingAV - (self.MOVING_LIST[0] / WINSIZE) + (current / WINSIZE)

        del self.MOVING_LIST[0]

        return self.movingAV


    def process_lidar(self, ranges):

        proc_ranges = self.preprocess_lidar(ranges)
        left_  = proc_ranges[30]
        right_ = proc_ranges[-30]
        LINE = abs(left_ - right_)
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

        # Speed = -a*Steering + b*distance + c*time
        # Weights
        a = 0.5
        b = 0.3
        c = 0.0025
        d = 0.01

        distance = self.temp(proc_ranges[best])

        if distance > 27:
            self.BEST_POINT_CONV_SIZE = 150
            self.BUBBLE_RADIUS = 40
            b = 0.325
            self.TIME += 3
            if self.TIME > 2500:
                self.TIME = 2500
        elif distance > 25:
            self.BEST_POINT_CONV_SIZE = 140
            self.BUBBLE_RADIUS = 40
            b = 0.3
            self.TIME += 1
            if self.TIME > 2500:
                self.TIME = 2500
        elif distance > 15:
            self.BEST_POINT_CONV_SIZE = 150
            self.BUBBLE_RADIUS = 40
            b = 0.275
            self.TIME -= 3
            if self.TIME < 800:
                self.TIME = 800
        else:
            b = 0.25
            self.BEST_POINT_CONV_SIZE = 150
            self.BUBBLE_RADIUS = 40
            self.TIME -= 3.0
            if self.TIME < 20:
                self.TIME = 200


        speed = (-a * abs(steering_angle)) + (b * distance) + (c * self.TIME) + (-d * LINE) + 6

        print(f'A : {(steering_angle / (np.pi / 2)) * 90 : .3f} | S : {speed : .3f} | D : {distance : .3f} | T : {self.TIME}, '
              f'A_ : {-a * abs(steering_angle) : .3f} D_ : {(b * distance) : .3f}  T_ : {(c * self.TIME) : .3f}',  end = '\r')


        return speed, steering_angle

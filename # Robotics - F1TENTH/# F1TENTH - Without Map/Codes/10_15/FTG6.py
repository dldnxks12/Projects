# 92 s 안정적

import numpy as np


class CustomDriver:
    BUBBLE_RADIUS = 60  # 위험 방울 범위

    PREPROCESS_CONV_SIZE = 3  # 이동 평균 Window
    BEST_POINT_CONV_SIZE = 200

    MAX_LIDAR_DIST = 3000000  # inf로 바꾸는 부분이겠지

    STRAIGHTS_STEERING_ANGLE = (np.pi / 18) # 10 degrees

    STRAIGHTS_SPEED = 8.0  # 직선 Speed
    CORNERS_SPEED = 5.0  # 코너링 Speed
    MOMENTUM = 0.5  # 관성 모멘텀
    CORNER_MOMENTUM = 0.3
    MOMENTUM_INDEX = 0

    i = 0 # print

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

            self.BEST_POINT_CONV_SIZE = 80
            if proc_ranges[best] > 10:
                if abs(steering_angle) < self.STRAIGHTS_STEERING_ANGLE*1.3:
                    speed = self.CORNERS_SPEED + (self.CORNER_MOMENTUM * proc_ranges[best])
                else:
                    speed = self.CORNERS_SPEED

            else:
                if abs(steering_angle) < self.STRAIGHTS_STEERING_ANGLE*1.3:
                    speed = self.CORNERS_SPEED + self.CORNER_MOMENTUM
                else:
                    speed = self.CORNERS_SPEED

        else:
            self.BEST_POINT_CONV_SIZE = 200
            if proc_ranges[best] > 25: # max length
                steering_angle = (steering_angle)*0.8
                self.MOMENTUM_INDEX += 5
                if self.MOMENTUM_INDEX > 40:
                    self.MOMENTUM_INDEX = 40
                speed = self.STRAIGHTS_SPEED + (self.MOMENTUM * self.MOMENTUM_INDEX)

            elif proc_ranges[best] > 15 and proc_ranges[best] <= 25 :
                speed = self.STRAIGHTS_SPEED + (0.3 * proc_ranges[best])

            elif proc_ranges[best] > 10 and proc_ranges[best] <= 15 :
                speed = self.STRAIGHTS_SPEED + (0.1 * proc_ranges[best])

            else:
                speed = self.STRAIGHTS_SPEED


        print(f"D : {proc_ranges[best]} MOMENTUM_INDEX {self.MOMENTUM_INDEX}")
        print(f'Steering Angle in Degree : {(steering_angle / (np.pi / 2)) * 90}')



        return speed, steering_angle

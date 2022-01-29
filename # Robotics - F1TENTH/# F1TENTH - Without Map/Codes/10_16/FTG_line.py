import numpy as np

class CustomDriver:
    BUBBLE_RADIUS = 20  # 위험 방울 범위

    PREPROCESS_CONV_SIZE = 3
    BEST_POINT_CONV_SIZE = 150

    MAX_LIDAR_DIST = 3000000  # inf로 바꾸는 부분이겠지

    STRAIGHTS_STEERING_ANGLE = (np.pi / 18)  # 10 degrees

    MOVING_LIST = []
    MOVING_LIST2 = []
    movingAV = 0
    movingAV2 = 0
    TIME = 0

    BEFORE_ANGLE = 0
    BEFORE = 0
    BEFORE_IDX = 0

    def __init__(self):
        self.radians_per_elem = None  # 분해각

    def preprocess_lidar(self, ranges):

        self.radians_per_elem = (2 * np.pi) / len(ranges)

        proc_ranges = np.array(ranges[135:-135])
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

        WINSIZE = 15
        self.MOVING_LIST.append(current)
        if len(self.MOVING_LIST) <= WINSIZE:
            self.movingAV = np.sum(self.MOVING_LIST) / len(self.MOVING_LIST)
            return  self.movingAV

        self.movingAV = self.movingAV - (self.MOVING_LIST[0] / WINSIZE) + (current / WINSIZE)

        del self.MOVING_LIST[0]

        return self.movingAV

    def speed_temp(self, speed):

        WINSIZE = 5
        self.MOVING_LIST2.append(speed)
        if len(self.MOVING_LIST2) <= WINSIZE:
            self.movingAV2 = np.sum(self.MOVING_LIST2) / len(self.MOVING_LIST2)
            return self.movingAV2

        self.movingAV2 = self.movingAV2 - (self.MOVING_LIST2[0] / WINSIZE) + (speed / WINSIZE)

        del self.MOVING_LIST2[0]

        return self.movingAV2

    def temp_D(self, current):
        # 둘의 부호가 다른지?
        # 둘의 거리차가 일정 이상인지?
        before = self.BEFORE
        gap = current - before

        return abs(gap)

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
        '''
        # Deal with Sudden Gap
        if self.BEFORE != 0:
            GAP = self.temp_D(proc_ranges[best])
            #print("GAP", GAP)
            if GAP > 10:
                best = (best + self.BEFORE_IDX) // 2
               # print("#--------------------- Occured !! -------------- # ")

        self.BEFORE = proc_ranges[best] # 다음을 위해 값 저장
        self.BEFORE_IDX = best
        '''
        steering_angle = self.get_angle(best, len(proc_ranges))

        # Speed = -a*Steering + b*distance + c*time
        # Weights
        a = 3.0
        b = 0.4
        c = 0.002
        d = 0.5
        distance = self.temp(proc_ranges[best])

        if distance > 15:
            self.TIME += 5
            if self.TIME > 1500:
                self.TIME = 1500
        else:
            self.TIME -= 5
            if self.TIME < 100:
                self.TIME = 100

        speed = (-a * abs(steering_angle)) + (b * distance) + (c * self.TIME) - (d * LINE) + 6

        speed = self.speed_temp(speed)

        print(f'A : {(steering_angle / (np.pi / 2)) * 90 : .3f} | S : {speed : .3f} | D : {distance : .3f} | T : {self.TIME}',  end = '\r')

              #f' ||| A_ : {-a * abs(steering_angle) : .3f} | D_ : {(b * distance) : .3f}  | T_ : {(c * self.TIME) : .3f} | LINE : {LINE}',  end = '\r')

        # print(f'After : {(steering_angle / (np.pi / 2)) * 90 : .5f} | Before : {(angle / (np.pi / 2)) * 90 : .5f} ')


        return speed, steering_angle

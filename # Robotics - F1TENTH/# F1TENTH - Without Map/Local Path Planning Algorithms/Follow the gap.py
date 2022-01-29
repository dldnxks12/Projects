import numpy as np


class GapFollower:
    BUBBLE_RADIUS = 160 # 위험 방울

    PREPROCESS_CONV_SIZE = 3 # 이동 평균 Window

    BEST_POINT_CONV_SIZE = 80

    MAX_LIDAR_DIST = 3000000  # inf로 바꾸는 부분이겠지

    STRAIGHTS_SPEED = 8.0 # 직선 Speed
    CORNERS_SPEED = 5.0   # 코너링 Speed
    STRAIGHTS_STEERING_ANGLE = np.pi / 18  # 10 degrees

    def __init__(self):
        # used when calculating the angles of the LiDAR data
        self.radians_per_elem = None

    def preprocess_lidar(self, ranges):
        """ Preprocess the LiDAR scan array. Expert implementation includes:
            1.Setting each value to the mean over some window
            2.Rejecting high values (eg. > 3m) (3m 이상인 거는 다 max로 바꿀 것)
        """
        # Lidar 분해각
        self.radians_per_elem = (2 * np.pi) / len(ranges)

        # we won't use the LiDAR data from directly behind us
        proc_ranges = np.array(ranges[135:-135])

        # sets each value to the mean over a given window --- Convolution (이동 평균)
        # np.convolve(데이터, 필터)
        # same ---- 연산되는 window와 같은 크기로 return
        proc_ranges = np.convolve(proc_ranges, np.ones(self.PREPROCESS_CONV_SIZE), 'same') / self.PREPROCESS_CONV_SIZE

        '''
        numpy.clip(array, min, max)
            array 내의 element들에 대해서        
            min 값 보다 작은 값들을 min값으로 바꿔주고        
            max 값 보다 큰 값들을 max값으로 바꿔주는 함수.
        '''
        proc_ranges = np.clip(proc_ranges, 0, self.MAX_LIDAR_DIST)
        return proc_ranges


    def find_max_gap(self, free_space_ranges): # free_space_ranges 에는 bubble zone이 포함되어 있다.
        """
            Return the start index & end index of the max gap in free_space_ranges
            free_space_ranges: list of LiDAR data which contains a 'bubble' of zeros
        """
        # mask the bubble
        # 조건이 충족되는 배열을 mask 씌워서 return 한다.
        '''
        
        mask = np.ma.masked_where(a <= 4, a) --- (condition, mask씌울 배열)
            [1 2 3 4 5]
            [-- -- -- -- 5]        
        '''
        masked = np.ma.masked_where(free_space_ranges == 0, free_space_ranges)

        # get a slice for each contigous sequence of non-bubble data
        # return endpoints(list of start and end index + 1)
        slices = np.ma.notmasked_contiguous(masked) # default axis --- flatten된 데이터에 대해서 수행


        # loop문을 통해 maximum length free space 찾기
        max_len = slices[0].stop - slices[0].start
        chosen_slice = slices[0]
        # I think we will only ever have a maximum of 2 slices but will handle an
        # indefinitely sized list for portablility
        for sl in slices[1:]:
            sl_len = sl.stop - sl.start
            if sl_len > max_len:
                max_len = sl_len
                chosen_slice = sl

        return chosen_slice.start, chosen_slice.stop # 시작과 끝점 return

    def find_best_point(self, start_i, end_i, ranges):
        """
        Start_i & end_i are start and end indices of max-gap range, respectively

        Return index of best point in ranges

        Naive: Choose the furthest point within ranges and go there ---- Naive 그대로 사용할 것 인가??

        """
        # do a sliding window average over the data in the max gap, this will
        # help the car to avoid hitting corners
        averaged_max_gap = np.convolve(ranges[start_i:end_i], np.ones(self.BEST_POINT_CONV_SIZE),
                                       'same') / self.BEST_POINT_CONV_SIZE

        return averaged_max_gap.argmax() + start_i

    def get_angle(self, range_index, range_len): # best index, scan data length(1080)

        """
            Get the angle of a particular element in the LiDAR data and transform it into an appropriate steering angle
        """
        lidar_angle = (range_index - (range_len / 2)) * self.radians_per_elem
        '''
            Furthest Drive와는 다르게 1/2 배를 해주었다. 이렇게 하면 차가 지그재그로 막 움직이는게 좀 덜하다 .
        '''
        steering_angle = lidar_angle / 2
        return steering_angle

    def process_lidar(self, ranges):
        """
            Process each LiDAR scan as per the Follow Gap algorithm & publish an AckermannDriveStamped Message
        """

        # Lidar sensing data를 전처리해서 가져오기
        proc_ranges = self.preprocess_lidar(ranges)

        # Find closest point to LiDAR  ---- 가장 가까운 data의 index
        closest = proc_ranges.argmin()

        # Eliminate all points inside 'bubble' (set them to zero)
        min_index = closest - self.BUBBLE_RADIUS
        max_index = closest + self.BUBBLE_RADIUS

        # 최대 index 최소 index 처리
        if min_index < 0:
            min_index = 0

        if max_index >= len(proc_ranges):
            max_index = len(proc_ranges) - 1

        # bubble 안에 있는 값들 모두 0으로 처리 ---- 0이 아닌 data들은 이제 free space 내의 데이터들이다.
        proc_ranges[min_index:max_index] = 0

        # Find max length gap --- 가장 긴 Free Space의 시작 index와 끝 index
        gap_start, gap_end = self.find_max_gap(proc_ranges)

        # Find the best point in the gap ---- best index get !
        best = self.find_best_point(gap_start, gap_end, proc_ranges)

        # Publish Drive message
        steering_angle = self.get_angle(best, len(proc_ranges))
        if abs(steering_angle) > self.STRAIGHTS_STEERING_ANGLE: # STRAIGHTS_STEERING_ANGLE == 10도
            speed = self.CORNERS_SPEED
        else:
            speed = self.STRAIGHTS_SPEED
        print('Steering angle in degrees: {}'.format((steering_angle / (np.pi / 2)) * 90))
        return speed, steering_angle
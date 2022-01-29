import numpy as np

# Car에서 가장 가까운 곳 까지의 거리를 Disparity라 한다.
class DisparityExtender: # Disparity를 넓혀가며 주행하는 방법

    CAR_WIDTH  = 0.31
    DIFFERENCE_THRESHOLD = 2.0
    SPEED = 5.0

    # 벽을 따라가기 위한 최소한의 공간이라고 이해
    SAFETY_PERCENTAGE = 300.0

    def preprocess_lidar(self, ranges):

        # 뒤에 있는 lidar point 무시
        eighth = int(len(ranges) / 8)  # lidar data를 8등분
        return np.array(ranges[eighth : -eighth])

    # 각 lidar 값 끼리의 차이
    def get_differences(self, ranges):
        differences = [0.0]
        for i in range(1, len(ragnes)):
            differences.append(abs(ranges[i] -  ranges[i-1]))
        return differences

    def get_disparities(self, differences, threshold):
        disparities = []
        for index, difference in enumerate(differences):
            if difference > threshold:
                disparities.append(index)

        return disparities

    def get_num_pointes_to_cover(self, dist, width):
        # Cover할 width(이 width 안의 lidar point를 처리)를 distance와 angle을 가지고 계산
        angle = 2 * np.arcsin(width / (2 * dist))
        num_points = int(np.ceil(angle / self.radians_per_point))
        return num_points

    def cover_points(self, num_points, start_idx, cover_right, ranges):

        new_dist = ranges[start_idx]
        if cover_right:
            for i in range(num_points):
                next_idx = start_idx + i + 1
                if next_idx >= len(ranges):
                    break
                if ranges[next_idx] > new_dist:
                    ranges[next_idx] = new_dist

        else:
            for i in range(num_points):
                next_idx = start_idx - i - 1
                if next_idx < 0:
                    break
                if ranges[next_idx] > new_dist:
                    ranges[next_idx] = new_dist

        return ranges

    def extend_disparities(self, disparities, ranges, car_width, extra_pct):
        width_to_cover = (car_width / 2) * (1 + extra_pct / 100)

        for index in disparities:
            first_idx = index - 1
            points = ranges[first_idx : first_idx + 2]
            close_idx = first_idx + np.argmin(points)
            far_idx = first_idx + np.argmax(points)

            clost_dist = ranges[close_dist]
            num_point_to_cover = self.get_num_pointes_to_cover(clost_dist,width_to_cover)

            cover_right = close_idx < far_idx # bool
            ranges = self.cover_points(num_point_to_cover, close_idx, cover_right, ranges)

        return ranges

    def get_steering_angle(self, range_index, range_len):
        lidar_angle = (range_index - (range_len / 2)) * self.radians_per_point
        steering_angle = np.clip(lidar_angle, np.radians(-90), np.radians(90))
        return steering_angle

    def _process_lidar(self, ranges):

        self.radians_per_point = (2 * np.pi) / len(ranges)
        proc_ranges = self.preprocess_lidar(ranges)
        differences = self.get_differences(proc_ranges)
        disparities = self.get_disparities(differences, self.DIFFERENCE_THRESHOLD)
        proc_ranges = self.extend_disparities(disparities, proc_ranges,
                                              self.CAR_WIDTH, self.SAFETY_PERCENTAGE)
        steering_angle = self.get_steering_angle(proc_ranges.argmax(),
                                                 len(proc_ranges))
        speed = self.SPEED
        return speed, steering_angle

    def process_observation(self, ranges, ego_odom):
        return self._process_lidar(ranges)



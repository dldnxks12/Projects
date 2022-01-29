### ROS & RL base Path Finder  

- `Dataset` 

        ROS rqt base Camera Image

<div align="center">

![img.png](img.png)

</div>

- `ROS`

        rqt tool     

- `Model`

        # Object Detection 
            1. RCNN Base
            2. YOLO Base

        # PG Base Algorithm
            1. REINFORCE
            2. A2C
            3. DPG
            4. DDPG

- `HW`

        Arduino or Rbpi 3 or 4
        DC Motor x 2
        Camera x 1

<br>

- `Project - 이미지 내 물체까지의 최적 경로 학습` 

    

<div align="center">

![img_1.png](img_1.png)

</div>

카메라 정면에 End Effector 모형을 부착하는 것으로 목표 지점 정의

<div align="center">

![img_2.png](img_2.png)

</div>

1. 이미지 내 객체 탐지 
2. 카메라를 움직이며 해당 물체와 End Effector가 만나도록 학습

<br>


- `학습 환경 - state / action` 

1. Discrete State Base

        이미지를 N x N 픽셀로 나누고, EE에서 각 픽셀까지의 Optimal Path를 학습 

<div align="center">

![img_3.png](img_3.png)

</div>

2. Continuous States Base

          물체가 있는 이미지 하나를 State로 판단하고, 연속적인 State에 대한 Optimal Policy를 학습

<div align="center">

![img_4.png](img_4.png)

</div>

3. Discrete Action

          Discrete State Base   - 상하좌우 이동 거리 적당히
          Continuous State Base - 상하좌우 이동 거리 정밀하게 

  




        

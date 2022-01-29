### Train Motion Classification `AI Hackerthon 2021 9.17 ~ 10.8`

[Hackerthon Link](https://dacon.io/competitions/official/235815/overview/description)

- `Competition Goal `

      3 axis 가속도계 , 3 axis 자이로스코프 데이터를 이용한 운동 동작 분류 
      
<br>      

- `데이터 형태 : csv file`

      # Train 
      Train Data  : 1875000 x 8 (ID 별 600 time step 간 동작 데이터 = 3125 x 600 = 1875000)
      (Sequential한 데이터가 1875000 x 8의 1차원 Vector 형태로 주어진다.)
      
      Train Label : 3125 x 3 (ID 별 동작)
      
      # Test
      Test Data   : 782 x 600 

<br>

- `평가지표`

      MSE Loss


<br>

- `Problem Solving`

      # 1) Data Processing
      
            Sequential한 데이터를 학습에 사용할 모델 Input의 형태로 맞춰준다.
      
            GRU/LSTM/RNN Mixture 모델과 1D CNN 모델 두 종류로 테스트.. 
            
            두 종류 모두 Sequential한 데이터를 받으므로 알맞게 Reshaping
            
      # 2) Data Augumentation / Normalization / Drop Out / Emsembel ...
      
            학습 양을 늘리고, 정규화와 Drop Out을 사용하여 Loss를 최대한 줄이는 한편
            
            앙상블 기법을 통해 학습 Quality를 높여보았다. 
                       
      
      # 3) RNN / LSTM / GRU
      
            Sequential한 데이터를 Recursive하게 학습하여 일련의 데이터 구조를 학습시킨다.
      
      # 4) 1D CNN
      
            2D CNN으로 유명하지만, Convolution 연산을 Sequential하게 수행하여 데이터의 공간적 특성이 아닌 순서적 특성을 학습할 수 있다.
      
<br>      

- `Competition Result`

<div align='center'>

25개 팀 중 12위로 마무리하였음.
      
![image](https://user-images.githubusercontent.com/59076451/151660776-662bd741-c92e-48e9-9c08-9bf0cbe6c889.png)
      
</div>

      
      




### 2021 Segmentation Project in NI²L LAB at KW Univ

#### Purpose of this Project  

  - Building Segmentation Model for Ultrasonic-Wave Image Processing for medical purposes

        what to do? 
        
        가공된 데이터를 넣어 질병의 유무를 Classification 해줄 Segmentation 모델을 만들 것

        BUSI dataset + (FCN / SegNet / UNET / DeepLab v1 / DeepLab v2)
        
---               
                
### To do list

#### 1주차

  - Pytorch remind

        1. Linear Regression   - ok
        2. Logistic Regression - ok
        3. Softmax Regression  - ok
        4. MLP/ANN             - ok
        5. CNN                 - ok

  - Segmentation ?

        1. Segmentation 이란? - ok

  - Code Review

        * code review ( segtrain.py ) - ok
        
        1. argparse           - ok 
        2. torch - dataloader - ok
        3. torchvision        - ok 

  - 추가 개념 

        1. SOTA Browser 
        
        2. Kaggle 

            Dataset Archive 
          
        3. torchvision - deeplabv3

            Semantic Segmentation model provided by Pytorch
            
              - 현재 UltraSound에서 사용하는 모델 : deeplabv-resnet50 , deeplabv-resnet101         

        4. contigous - ok
        
            데이터 읽어 들이는 순서에 관한 함수
          
        5. permute VS view - ok 
          
            view    : 데이터 읽어들이는 순서 변경 (shape는 사실 변하지 않는다.)
            permute : Dimension Index 순서 변경
              
               permute를 사용할 경우 contigous와 같이 사용하는 경우가 많다.
          
        6. SubsetRandomSampler - ok

            전체 dataset에서 Train dataset과 Test dataset을 각각 부분 집합으로 만들어 dataloader에 넣어주는 방법 

---

#### 2주차

  - Paper & Notation 

        1. FCN                - ok
        2. SegNet             - ok
        3. Deconvolution      - ok
        4. AE  (Auto Encoder) - ok
        5. CAE                - ok
        6. deeplab v2    ~ing 
        7. 평가 지표           - ok
        8. ResNet             - ok
  
  [Deconvolution](https://zzsza.github.io/data/2018/06/25/upsampling-with-transposed-convolution/)

  - 기본적인 Segmentation 구현 with Kaggle BUSI Dataset  - 80 %

        1. Segmentation 구현 with FCN, BUSI 

            BUSI Dataset 중 feature가 확실한 benign dataset을 이용
            
            Lower layer : VGG-16
            Upper layer : FCN-8s 
            
            * 문제 *
            
              데이터 전처리 부분에서 model을 모두 같은 사이즈로 만드는 과정에서 불필요한 여백이 생김
              
              이 부분을 감안하고 학습을 시켰고, 결과적으로 성능이 그다지 좋지 않았다.
              
              해당 여백이 문제인 것인지는 아직 잘 모르지만, 우선 해결하는 것이 좋아보임 
              
              (사이즈 상관없이 넣어주어도 되는 것으로 아는데, 적어도 데이터셋의 크기를 통일시켜야하나?)
                
<br>

<div align="center">

**2주차 진행결과** 

**FCN**

preprocessed test data
  
![image](https://user-images.githubusercontent.com/59076451/129725196-72cc0b4d-50bb-4f8e-8dbd-c18cfd8e7c93.png)
  
test mask
  
![image](https://user-images.githubusercontent.com/59076451/129725093-f61ebf10-a38d-4cd2-815c-53e6548d4575.png)
  
test result

![image](https://user-images.githubusercontent.com/59076451/129725036-cdc0b1ee-f10d-4abb-a55b-aafcbcecd1fe.png)


</div>

---
<br>

#### 3주차


- Model 
        
          1. SegNet Review    - ok
          2. U-Net Review     - ok 
          3. DeConvNet Review - ok 
          4. SegNet 구현       - ok
          5. U-Net 구현        - ok

- 추가 구현 

          VGG        - ok
          GoogLeNet  - ok
          ResNet     - ok

- Data Preprocessing
    
          1. torchvision.datasets.ImageFolder    - ok
          2. dataloader class __init__에서 전처리 - ok
          3. data augmentation (720 -> 4320)     - ok  

- 기존 모델 향상   

          Loss_fn : MSE or BCE - BCE
          channel : 1 channel or 2 channel - 1 channel 
          activation_fn : Sigmoid or Softmax - Sigmoid 
          
          1. FCN with BCE + sigmoid / MSE + softmax - ok 
          2. FCN -> SegNet / BCE + Sigmoid          - ok
          3. FCN -> U-Net + BUSI + Softmax
          4. FCN -> U-Net + BUSI + Sigmoid          - ok
          5. FCN -> U-Net + Carvana + Sigmoid       - ok
          6. Weight Initialize                      - ok
          7. Tensorboard                            - ok
          8. Progressbar with tqdm                  - ok
          9. Color map                              - ok
          10. GPU Setting                           - ok 

---
<br>


<div align=center>

**3주차 진행 결과** 

**FCN**

FCN + BUSI + BCELoss + 2ch-Sigmoid  
  
![image](https://user-images.githubusercontent.com/59076451/130317090-6d769014-2c5a-413b-9f7e-06fe4929a766.png)
  
FCN - BUSI + MSELoss + 1ch-Sigmoid  
  
![image](https://user-images.githubusercontent.com/59076451/130567067-2b951db7-d418-4dec-b2f8-2b3e06ecb536.png)
  
FCN - BUSI + MSELoss + 1ch-Sigmoid with GPU (batch_size 30 , epoch 15)
    
![image](https://user-images.githubusercontent.com/59076451/130675604-a6bcd3b5-93db-4e96-bea2-61b4f9b4e75d.png)  
  
**SegNet**

SegNet - Carvana + BCELOSS + 1ch-Sigmoid  GPU bathsize 100, epoch 1
  
![image](https://user-images.githubusercontent.com/59076451/130656027-00d92940-80ef-4223-8afe-7f04b0ec9e87.png)

**UNET**

U-Net - Carvana + BCELOSS + 1ch-Sigmoid

![image](https://user-images.githubusercontent.com/59076451/131260464-5af6a99b-d607-48ff-9341-7a95169f8d73.png)  
  
U-Net - BUSI + BCELOSS + 1ch-Sigmoid (적은 데이터와 적은 학습에도 객체를 잘 찾는다.)

![img.png](img.png)

하지만 작은 객체는 비교적 잘 찾지 못하는 모습을 보여주었음 

</div>  
  
  
  
  
  
  
- 링크 

[Deconvolution-CAE](https://wjddyd66.github.io/pytorch/Pytorch-AutoEncoder/) <br>
[deeplabv3](https://shangom-developer.tistory.com/4) <br>
[deeplabv3](https://github.com/jfzhang95/pytorch-deeplab-xception) <br>
[Reference Code](https://github.com/spmallick/learnopencv/tree/master/PyTorch-Segmentation-torchvision)

### Human Parsing Dataset를 통한 Segmentation Model 성능 비교  



- `사용 모델` 

       # FCN
       # SegNet 
       # U-NET 

- `사용 데이터셋` 

       # Human Parsing dataset, 
       # BUSI dataset 
       # Carvana dataset 

---

<br>

<div align=Center>

**FCN**
    
Human Parsing - 2ch sigmoid
    
![image](https://user-images.githubusercontent.com/59076451/130016299-604180d0-9926-4f7a-9e82-65d6dd49225d.png)

BUSI Dataset - 2ch softmax
    
![image](https://user-images.githubusercontent.com/59076451/130358077-dcd75094-4ef1-46b6-b32e-da79b28e380e.png)
    
BUSI Dataset - 1ch sigmoid with GPU batch_size 30 , epoch 15
    
![image](https://user-images.githubusercontent.com/59076451/130675604-a6bcd3b5-93db-4e96-bea2-61b4f9b4e75d.png)


**SegNet**

SegNet - Carvana + BCELOSS + 1ch-Sigmoid  GPU bathsize 100, epoch 1
  
![image](https://user-images.githubusercontent.com/59076451/130656027-00d92940-80ef-4223-8afe-7f04b0ec9e87.png)

**UNET**

U-Net - Carvana + BCELOSS + 1ch-Sigmoid

![image](https://user-images.githubusercontent.com/59076451/131260464-5af6a99b-d607-48ff-9341-7a95169f8d73.png)  

    
</div>    

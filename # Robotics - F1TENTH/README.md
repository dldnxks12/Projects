### Robot Navigation project - `F1 TENTH Driving Algorithm`

<div align="center">

![img_5.png](img_5.png)

</div>

`Took 1st Place in F1TENTH Mid   EXAM !! ๐`  
`Took 1st Place in F1TENTH Final EXAM !! ๐๐`

<div align=center>
   
`MID EXAM`

![image](https://user-images.githubusercontent.com/59076451/143087485-7b1094a7-34f1-4215-aab1-459737e2ff5d.png)
        

`FINAL EXAM`

![img_3.png](img_3.png)

</div>        

<br>

- `Project Goal`

1. ๋ค์ ์ง๋์ ์ค์ฌ ๊ธฐ์ค ์๋จ ๋ถ๋ถ์ `Unknown Field` ์ด๋ค.  
์ฆ, ์๋ ค์ง์ง ์์ ์ฅ์ ๋ฌผ์ ํผํ  ์ ์๋ ์ฃผํ ์๊ณ ๋ฆฌ์ฆ์ ์ง์ผํ๋ค.


2. ๋งต ์ ๋ณด๊ฐ ์ฃผ์ด์ง ๊ฒฝ์ฐ ์ฒดํฌ ํฌ์ธํธ๋ฅผ ์ง๋๋๋ก ์ฃผํ ์๊ณ ๋ฆฌ์ฆ์ ์ง์ผํ๋ค.

<div align="center">

`์ฃผ์ด์ง ๋งต ์ ๋ณด`

![img_4.png](img_4.png)

`์ค์  ์ ๋ต ๊ทธ๋ฆผ`

![img_1.png](img_1.png)

</div>

#### Furthest Drive Algorithm 

- Steering Part : `Ackerman Steering ๊ตฌ์กฐ ์ฌ์ฉ`

```python

    class FurtestDrive():
        
        def lidar(self, ranges):
          
          NUM_RANGES = len(ranges)        
          ANGLE_BETWEEN = 2 * np.pi / NUM_RANGES
 
          NUM_PER_QUADRANT = NUM_RANGES // 4
          max_idx = np.argmax(ranges[NUM_PER_QUADRANT:-NUM_PER_QUADRANT]) + NUM_PER_QUADRANT
          
          steering_angle = max_idx * ANGLE_BETWEEN - (NUM_RANGES // 2) * ANGLE_BETWEEN
          speed = 5.0

          return speed, steering_angle 

```    

steering์ ๋ํด์ ๋ค์ ๊ทธ๋ฆผ์ ํตํด ์ดํดํด๋ณด์.

<div align="center">

![img.png](img/img.png)

`์กฐํฅ๊ฐ๋ ๊ฒฐ์ `

![img_1.png](img/img_1.png)

</div>

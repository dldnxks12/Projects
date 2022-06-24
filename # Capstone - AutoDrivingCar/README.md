#### Camera based auto driving car


    1. Make virtual driving line with Segmentation 
    2. Line based driving 



#### Used Method 

                Camera Calibration
                Side Walk Segmentation 
                Remove noise with Morpholgy - Opening
                Get Contours and select biggest Area Contour
                Momentum
                HoughLinesP
                Get Main Left / Right Line
                Get ROI 
                Interpolation
                Line Moving Average
                Get Angle    + moving average + remove outliar
                Get Distance + moving average + remove outliar
                Control Output
                Communication related work 

                    # Canny Edge
                    # Bird-eye-view


#### Hardware


<div align=center>

![image](https://user-images.githubusercontent.com/59076451/175515957-6341b9ba-02db-4690-93e0-239367bbbcd6.png)


![image](https://user-images.githubusercontent.com/59076451/175516041-436ea907-dfd1-4042-926a-de69a2efe979.png)

`control input`

</div>

<br>

#### Software

<div align=center>

`virtual line base`

![image](https://user-images.githubusercontent.com/59076451/175516212-ddc71b29-ac90-4bf5-ad54-615659cfc8e8.png)

![image](https://user-images.githubusercontent.com/59076451/175516295-2dff6dbf-b2ed-4767-890f-bdab9e2f69f4.png)

![image](https://user-images.githubusercontent.com/59076451/175516271-009e987b-994e-4d70-afea-c2fd1676e567.png)

<br>


`actual line base`

![image](https://user-images.githubusercontent.com/59076451/175516110-8ddfa99a-5fcf-4589-9393-822c8790a015.png)

![image](https://user-images.githubusercontent.com/59076451/175516165-49c65495-5abd-47e2-b495-66b321b063cd.png)

<br>


`gps base object detection recording`

![image](https://user-images.githubusercontent.com/59076451/175516364-f25c4e99-236a-48bd-b95e-82e982759bfa.png)

</div>



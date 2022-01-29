### Remote CCTV 


<div align="center">

 ![image](https://user-images.githubusercontent.com/59076451/151537584-647d496a-5a64-49e4-85a5-5d21b72c8f1d.png)
  
</div>  

#### NAT 설정을 통한 Port Forwarding 

      라즈베리 파이에 직접 VNC로 접속하여 제어 : 기본적인 지식이 필요
      라즈베리 파이에서 호스팅하는 Web Server에 접속하여 제어 : 단순한 UI로 제어 가능

#### Camera 
    
    - UV4L을 이용한 WebRTC로는 단순히 영상을 띄울 수 있을 뿐 다른 제어와 함께 사용하기 힘듦
    - 또한 UV4L을 사용하면 OpenCV의 Imshow() 함수와 카메라 점유에 있어 충돌을 일으키게 된다. 
      - 이것을 해결하기 위해 OpenCV로 얻어온 이미지를 웹 페이지에 전송하는 방법으로 실시간 스트리밍을 구현한다.
 
#### os.system Command
  
      Web Hosting을 원격으로 종료하기 위하여 python file 내부에서 os.system command를 사용
  
#### SMTP Alarm
  
      Web Hosting 시작과 Camera를 동작시킬 시 개인 메일로 알림을 보내, 보안을 강화   

<div align="center">

  ![image](https://user-images.githubusercontent.com/59076451/151537973-f50ad441-dad0-493b-912a-b11497a01388.png)
  
</div>

#### Web에서 각 서보모터를 제어할 수 있도록 구성 

      총 2개의 서보모터를 0 ~ 180도 까지 제어 
  
  
  


  
  
  

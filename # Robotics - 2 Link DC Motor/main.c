/*
 * GccApplicatikhyojon1.c
 *
 * Author : KHS
 * Revised : LJS
 */

#include "mcu_init.h" // atmega128 초기화
#include "dataType.h" // Packet data

/////////////////////////////////////////////////////////////////////////////////////////////////////////////

volatile int32_t g_Cnt, g_preCnt; // Encoder ...

// Position
volatile double g_Ppre; // 이전 Position
volatile double g_Pcur; // 현재 Position
volatile double g_Pdes = 0.; // 목표 Position
volatile double g_Perr; // 에러
volatile double g_Pvcur;
volatile double g_Ppre_err = 0;
volatile double g_Perr_dif = 0;
volatile double g_Ppre = 0;

// Velocity
volatile double g_Vpre; // 이전 속도
volatile double g_Vcur; // 현재 속도
volatile double g_Vdes = 0.2; // 목표 속도
volatile double g_Verr; // 에러
volatile double g_Verr_sum;
volatile double g_Vlimit = 1.; // 제한 속도

// Current
volatile double g_Ccur; // 현재 전류
volatile double g_Cdes; // 목표 전류
volatile double g_Cerr; // 현재 - 목표 사이 에러
volatile double g_Cerr_sum; // 에러 합 for I Control
volatile double g_Climit = 1.; // 제한 전류

volatile int cur_control = 0;
volatile double g_vel_control;
volatile double g_pos_control;
volatile unsigned char g_TimerCnt; // For 제어주기

// 제어 게인 튜닝 ~ing
volatile double pos_P = 0.5; //0;  --- 초기값 1로
volatile double pos_D = 0.5; //0;  --- 초기값 1로
volatile double vel_P = 2;
volatile double vel_I = 0.001;
volatile double cur_P = 8.26;
volatile double cur_I = 0.7;
//////////////////////////////////////////////////////

volatile double g_ADC; // Analog to Digital Convert
volatile int g_SendFlag = 0;
volatile int g_Direction;

// For Packet Data
// Packet으로 Position , Velocity, Current 값 주고 받기
volatile Packet_t g_PacketBuffer; // Packet 받을 Buffer
volatile unsigned char g_PacketMode; // 패킷 명령 종류
volatile unsigned char g_ID = 1; // 수신할 장치 ID
volatile unsigned char checkSize; //
volatile unsigned char g_buf[256], g_BufWriteCnt, g_BufReadCnt;

// PWM 제어 -- DC 모터 출력 제어
void SetDutyCW(double v){ // 우리 출력 값을

	while(TCNT1  == 0);

	// TCNT1 값이 증가해서 위 while문 통과하는 지 확인 할 것 --------------- Check 1
	//PORTA = ~PORTA;  // --- LED Check

	int ocr = v * (200. / 24.) + 200; // 전압에 따라 OCR 값 처리

	// OCR_MAX : 390
	// OCR_MIN : 10
	if(ocr > OCR_MAX)	ocr = OCR_MAX;
	else if(ocr < OCR_MIN)	ocr = OCR_MIN;
	//OCR1A = OCR1B = ocr;

	// PC PWM --- A 하나만 써서
	OCR1A = OCR3B = ocr + 8; 		//1 H
	OCR1B = OCR3A = ocr - 8;		//1 L
}

// No touch
void InitLS7366(){

	// SPI 통신을 통해 Motor Encoder 값을 읽어옴
	// PORT B 입력 -> 마스터로부터 데이터 읽기
	// PORT B 출력 -> 마스터로 데이터 쓰기
	PORTB = 0x00;
	SPI_MasterSend(SELECT_MDR0 | WR_REG);
	SPI_MasterSend(X4_QUAD | FREE_RUN | DISABLE_INDEX | SYNCHRONOUS_INDEX |FILTER_CDF_1);
	PORTB = 0x01;

	PORTB = 0x00;
	SPI_MasterSend(SELECT_MDR1 | WR_REG);
	SPI_MasterSend(FOUR_BYTE_COUNT_MODE | ENABLE_COUNTING);
	PORTB = 0x01;

	PORTB = 0x00;
	SPI_MasterSend(SELECT_CNTR | CLR_REG);
	PORTB = 0x01;
}

int getADC(char ch){
	ADMUX = (ADMUX & 0xF0) + ch;
	ADCSRA |= 0x40;
	while(!(ADCSRA & 0x10));
	return ADC;
}

ISR(USART0_RX_vect){
	g_buf[g_BufWriteCnt++] = UDR0; // UART에서 받은 값
}

// Timer 0 , 2 : 8 bit (0 ~ 255)
// Timer 1, 3 : 16 bit (0 ~ 65535)

//ISR(TIMER1_OVF_vect){}
//ISR(TIMER2_OVF_vect){}
//ISR(TIMER3_OVF_vect){}

ISR(TIMER0_OVF_vect){

	// ISR 진입 0.0001초 세팅
	// 제어주기 --- 위치 : 0.1초 , 속도 : 0.01초 , 전류 : 0.001초
	TCNT0 = 256 - 231;

	//Read LS7366
	int32_t cnt;

	// ????
	PORTC = 0x01;

	g_ADC = getADC(0); // 전류 측정한 값 AD 변환해서 Get

	// PORTB 출력 -> 마스터에 보내기
	// PORTB 입력 -> 마스터로 부터 데이터 읽기
	PORTB = 0x00;
	SPI_MasterSend(SELECT_OTR | LOAD_REG);
	PORTB = 0x01;

	PORTB = 0x00;
	SPI_MasterSend(SELECT_OTR | RD_REG);
	cnt = SPI_MasterRecv();
	cnt = cnt<< 8;
	cnt |= SPI_MasterRecv();
	cnt = cnt<< 8;
	cnt |= SPI_MasterRecv();
	cnt = cnt<< 8;
	cnt |= SPI_MasterRecv();
	PORTB = 0x01;
	g_Cnt = -cnt;

	// ????
	PORTC = 0x03;

	// 현재 각도 --- LS7366 Encoder에서 읽어온 값을 통해 계산
	g_Pcur = (g_Cnt / (4096. * 81.)) * 2 * M_PI;
	// 81 = 기어비
	// 4096 = 버퍼 사이즈
	// g_cnt = 엔코더 카운팅
	// M_PI = 3.141592...

	//TO DO
	//Position Control
	if((g_TimerCnt % 1000) == 0) // 제어주기 : 0.1초
	{
		g_Perr = g_Pdes - g_Pcur; // 목표 위치 - 현재 위치
		g_Perr_dif = (g_Perr - g_Ppre_err) / 0.1; // 수치 미분
		g_Ppre_err = g_Perr; // 다음 계산을 위해 Error 저장

		// 위치 제어 출력 값 -> 속도 제어 입력으로 넣어줄 것
		g_pos_control = g_Perr * pos_P + g_Perr_dif * pos_D;

		// 출력 속도 제한 = 1 rad/s
		// 강의자료에서는 -6140 ~ 6140 RPM으로 제한 (Radian으로 바꿔서 사용)
		if(g_pos_control > g_Vlimit)
			g_pos_control = g_Vlimit;

		else if(g_pos_control <= -g_Vlimit)
			g_pos_control = -g_Vlimit;
	}

	//Velocity Control
	if((g_TimerCnt % 100) == 0){ // 제어주기 : 0.01 초

		if(g_Pdes != 0)
		{
			g_Vdes = g_pos_control; // 속도 타겟 값 (Reference value)
		}

		g_Vcur = (g_Pcur - g_Ppre) / 0.01; // 수치 미분
		g_Verr = g_Vdes - g_Vcur; // 잔차
		g_Verr_sum += g_Verr; // 속도 잔차 적분 값 for I Control

		// Integral Anti Wind-UP...  delay 없애기 위한 방법 중 하나 ..  (로제 과제 참고)
		if(g_Verr_sum < -200 || g_Verr_sum > 200)
			g_Verr_sum -= g_Verr;

        // 속도 제어 출력 값 -> 전류 제어 입력으로 넣어줄 것
		g_vel_control = g_Verr * vel_P + g_Verr_sum * vel_I * 0.01;

		g_Ppre = g_Pcur; // 현재 위치를 이전 위치에 넣기

		// Method 1
		// 전류 제한 = 1 (강의 자료에서는 -27.3 ~ 27.3 A로 제한)
		if(g_vel_control > g_Climit){
			g_vel_control = g_Climit;
		}else if(g_vel_control <= -g_Climit){
			g_vel_control = -g_Climit;
		}

		// Method 2  ---> I-term Anti WindUp 추가 (Delay 감소)
		/*
		if(g_vel_control > g_Climit){
			g_Verr_sum -= (g_vel_control - 1) * 1. /  vel_P / 3.0; // 적분값 줄이기
			g_vel_control = g_Climit;
		}
		else if(g_vel_control <= -g_Climit){
			g_Verr_sum -= (g_vel_control + 1) * 1. /  vel_P / 3.0;
			g_vel_control = -g_Climit;
		}
		*/
	}

	//Current Control
	if(g_TimerCnt % 10 == 0){ // 제어 주기 : 0.001초

		// 위치 또는 속도가 0이 아닐 때
		if(g_Vdes != 0 || g_Pdes != 0){
			g_Cdes = g_vel_control; // Reference Current 설정
		}

		// g_Ccur = -(((g_ADC / 1024.0 * 5.0) - 2.454) * 10.0 - 0.264 + 0.048);

		// 1024 : Encoder 분해능  // g_ADC : 전류 AD 변환된 값
		g_Ccur = -( ((g_ADC / 1024. * 5.) - 2.5) * 10.); // 전류 수치 변환
		g_Cerr = g_Cdes - g_Ccur; // 전류 잔차

		g_Cerr_sum += g_Cerr;  // 잔차 합 for 적분제어

		// Integral Anti Wind-UP... (로제 과제 참고)
		if(g_Cerr_sum < -1200 || g_Cerr_sum > 1200){
			g_Cerr_sum -= g_Cerr;
		}

		// Current Control Input
		cur_control = g_Cerr * cur_P + g_Cerr_sum * cur_I * 0.001;

		// Method 1
		// 강의자료에서는 정격전압 -48 ~ 48V 사이로 제한
		if(cur_control > 24){
			cur_control = 24;
			}else if(cur_control <= -24){
			cur_control = -24;
		}

		// Method 2 --- > I-term Anti Wind UP 추가 (Delay 감소)
		/*
		if(cur_control > 24){
			g_Cerr_sum -= (cur_control - 24.0) * 1.0 / cur_P / 3.0;
			cur_control = 24;
		}else if(cur_control <= -24){
			g_Cerr_sum -= (cur_control + 24.0) * 1.0 / cur_P / 3.0;
			cur_control = -24;
		}
		*/
	}

	// AD 변환된 전류와 속도 제어기의 출력으로 받은 전류 사이의 에러를 통해
	// PWM(OCR Value)을 조정하며 모터 제어
	SetDutyCW(cur_control);
	/////////////////////////////////////////

	g_TimerCnt++;
	g_SendFlag++;

}



int main(void){

	Packet_t packet;
	packet.data.header[0] = packet.data.header[1] = packet.data.header[2] = packet.data.header[3] = 0xFE;

	InitIO(); // PORT Setting

	//Uart
	InitUart0();

	//SPI
	InitSPI();

	//Timer
	InitTimer0(); // Normal Count - 여기서 PID 제어기 돌려야함
	InitTimer1(); // PC PWM
	InitTimer3(); // PC PWM

	TCNT1 = TCNT3 = 0;

	// PWM
	SetDutyCW(0.);

	//ADC
	InitADC();

	//LS7366 -- Encoder
	InitLS7366();

	TCNT0 = 256 - 231; //1 clk = 0.0001초
	sei();

	// MFC와 통신하는 코드
	unsigned char check = 0;

    while (1) {
		// UART ISR에서 UDR value 담았다. 즉, 담아놓은 데이터 다 읽어라 이 말
		for(;g_BufReadCnt != g_BufWriteCnt; g_BufReadCnt++){

			switch(g_PacketMode){
			case 0:

				// Header Check (header 4개 모두 0xFF라면 ... )
				if (g_buf[g_BufReadCnt] == 0xFF) {
					checkSize++;
					if (checkSize == 4) {
						g_PacketMode = 1;
					}
				}
				else {
					checkSize = 0;
				}
				break;

			case 1:

				g_PacketBuffer.buffer[checkSize++] = g_buf[g_BufReadCnt];

				if (checkSize == 8) {
					if(g_PacketBuffer.data.id == g_ID){
						g_PacketMode = 2;
					}
					else{
						g_PacketMode = 0;
						checkSize = 0;
					}
				}

				break;

			case 2:

				// 앞 데이터 모두 Check 됬으면 실제로 받은 데이터 읽어오기
				g_PacketBuffer.buffer[checkSize++] = g_buf[g_BufReadCnt];
				check += g_buf[g_BufReadCnt];

				if (checkSize == g_PacketBuffer.data.size) {

					if(check == g_PacketBuffer.data.check){

						switch(g_PacketBuffer.data.mode){

							case 2:

							// Check Code 추가
							// PORTA = ~PORTA; // Data 통신이 이루어지는 지 LED로 Check ------------------- Check 2

							// 값이 너무 작아 1000을 곱해 보내주었다. 따라서 다시 1000으로 나누어 사용한다.
							// Packet에 담겨져서 돌아온 데이터에서 Position / Velocity / Current 가져오기
							g_Pdes = g_PacketBuffer.data.pos / 1000.;
							g_Vlimit = g_PacketBuffer.data.velo / 1000.;
							g_Climit = g_PacketBuffer.data.cur / 1000.;
							break;
							}
					}

					check = 0;
					g_PacketMode = 0;
					checkSize = 0;
				}
				else if(checkSize > g_PacketBuffer.data.size || checkSize > sizeof(Packet_t)) {
					TransUart0('f');
					check = 0;
					g_PacketMode = 0;
					checkSize = 0;
				}
			}  // Switch END
		} // Data Read From Buffer END


		if(g_SendFlag > 19){
			g_SendFlag = 0;

			packet.data.id = g_ID;
			packet.data.size = sizeof(Packet_data_t);
			packet.data.mode = 3;
			packet.data.check = 0;

			packet.data.pos = g_Pdes * 1000;
			packet.data.velo = g_Vlimit * 1000;
			packet.data.cur = g_Climit * 1000;

			for (int i = 8; i < sizeof(Packet_t); i++)
				packet.data.check += packet.buffer[i];

			// packet.buffer[0]에 packet 객체 하나씩 들어가있다.
			for(int i=0; i<packet.data.size; i++){
				TransUart0(packet.buffer[i]);
			}
			// Packet 잘 보내는지 LED로 Check ----------------------------- Check 3
		}
	}
}

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
volatile double g_Ppre = 0; // 이전 Position
volatile double g_Pcur = 0; // 현재 Position
volatile double g_Pdes = 0.; // 목표 Position
volatile double g_Perr = 0; // 에러
volatile double g_Pvcur = 0;
volatile double g_Ppre_err = 0;
volatile double g_Perr_dif = 0;

// Velocity
volatile double g_Vpre = 0; // 이전 속도
volatile double g_Vcur = 0; // 현재 속도
volatile double g_Vdes = 0; // 목표 속도
volatile double g_Verr = 0; // 에러
volatile double g_Verr_sum = 0;
volatile double g_Vlimit = 1.; // 제한 속도

// Current
volatile double g_Ccur = 0; // 현재 전류
volatile double g_Cdes = 0; // 목표 전류
volatile double g_Cerr; // 현재 - 목표 사이 에러
volatile double g_Cerr_sum = 0; // 에러 합 for I Control
volatile double g_Climit = 1.; // 제한 전류

volatile int cur_control = 0;
volatile double g_vel_control = 0;
volatile double g_pos_control = 0;
volatile unsigned char g_TimerCnt = 0; // For 제어주기


// 제어 게인 튜닝 ~ing
volatile double pos_P = 1;
volatile double pos_D = 0.1;
volatile double vel_P = 2;
volatile double vel_I = 0.1;
volatile double cur_P = 1.0;
volatile double cur_I = 0.1;

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

ISR(TIMER1_OVF_vect){}
ISR(TIMER2_OVF_vect){}
ISR(TIMER3_OVF_vect){}

ISR(TIMER0_OVF_vect){

	// ISR 진입 0.0001초 세팅
	// 제어주기 --- 위치 : 0.1초 , 속도 : 0.01초 , 전류 : 0.001초
	TCNT0 = 256 - 231;

	//Read LS7366
	int32_t cnt;

	PORTC = 0x01;

	g_ADC = getADC(0);

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

	PORTC = 0x03;

	g_Pcur = (g_Cnt / (4096. * 81.)) * 2 * M_PI;

	//TO DO
	//Position Control --- Check
	if((g_TimerCnt % 1000) == 0) // 제어주기 : 0.1초
	{
		PORTA = ~PORTA;

	}

	//Velocity Control
	if((g_TimerCnt % 100) == 0)  // 제어주기 : 0.01 초
	{
	}

	//Current Control
	if(g_TimerCnt % 10 == 0){ // 제어 주기 : 0.001초

		g_Ccur = -( ((g_ADC / 1024. * 5.) - 2.5) * 10.) ;  // 현재 전류
		g_Cerr = g_Cdes - g_Ccur; // 에러

		g_Cerr_sum += g_Cerr; // 에러합
		cur_control = g_Cerr * cur_P + g_Cerr_sum * cur_I * 0.001;

	}

	SetDutyCW(cur_control);

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
	TCNT0 = 256 - 231; //1 clk = 0.0001초

	//ADC
	InitADC();

	//LS7366 -- Encoder
	InitLS7366();

	sei();

	// PWM
	SetDutyCW(0.);

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
							g_Vdes = g_PacketBuffer.data.velo / 1000.;
							g_Cdes = g_PacketBuffer.data.cur / 1000.;

							//g_Pdes = g_PacketBuffer.data.pos / 1000.;
							//g_Vdes = g_PacketBuffer.data.velo / 1000.;
							//g_Cdes = g_PacketBuffer.data.cur / 1000.;

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

			packet.data.pos = g_Pcur * 1000;
			packet.data.velo = g_Vcur * 1000;
			packet.data.cur = g_Ccur * 1000;

			//packet.data.pos = g_Pdes * 1000;
			//packet.data.velo = g_Vlimit * 1000;
			//packet.data.cur = g_Climit * 1000;

			for (int i = 8; i < sizeof(Packet_t); i++)
				packet.data.check += packet.buffer[i];

			// packet.buffer[0]에 packet 객체 하나씩 들어가있다.
			for(int i=0; i<packet.data.size; i++){
				TransUart0(packet.buffer[i]);
			}
		}
	}
}

#define F_CPU 16000000UL
#define UBRR 103 // 16Mhz -> baud rate 9600 , U2X = 0
#include <avr/interrupt.h>
#include <util/delay.h>
#include <avr/io.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define TRIG 6 // E PORT 6 PIN
#define ECHO 7 // E PORT 7 PIN
#define SOUND_VELOCITY 340UL // 초음파 센서 음속


// USART

ISR(TIMER1_OVF_vect)
{  }

ISR(TIMER1_COMPA_vect)
{}

ISR(TIMER3_COMPA_vect)
{}

unsigned char USART0_RX_vect(void)
{
	
	while(!(UCSR0A & (1<<RXC0)));  // 버퍼에 읽지 않은 데이터가 있으면 무한루프 탈출
	
	return UDR0; // UDR 레지스터에 있는 값 받기 (PC to Atmega)
	
}

void USART0_TX_vect(unsigned char data)  // character 하나씩 받아와서 PC로 Transmit
{
	while(!(UCSR0A & (1<<UDRE0))); // 버퍼가 비어 있으면 무한루프 탈출
	
	UDR0 = data; // UDR 레지스터에 data 넣어주기 (Atmega to PC)

}

int GetADC0(void) // thermistor
{
	ADMUX = 0x03;
	ADCSRA |= (1 << ADSC);
	while(!(ADCSRA & (1<< ADIF)));
return ADC;}
int GetADC1(void) // CDS
{
	ADMUX = 0x01;
	ADCSRA |= (1<< ADSC);
	while(!(ADCSRA & (1<< ADIF)));
return ADC;}
int GetADC2(void) // 가변저항
{
	ADMUX = 0x00;
	ADCSRA |= (1 << ADSC);
	while(!(ADCSRA & (1<< ADIF)));
return ADC;}
int GetADC3(void) // PSD
{
	ADMUX = 0x04;
	ADCSRA |= (1<< ADSC);
	while(!(ADCSRA & (1<< ADIF)));
return ADC;}
int GetADC4(void) // LM35
{
	ADMUX = 0x02;
	ADCSRA |= (1<< ADSC);
	while(!(ADCSRA & (1<< ADIF)));
return ADC;}


// Direct 2 form IIR 사용

float IIR(int data[], int datalength, float a1, float a2, float b0, float b1, float b2)
{
	// datalength : 4
	float result[10] = {0,}; // 값을 임시저장한 result 버퍼 선언
	
	int i;
	for(i = 2; i < datalength; i++) // Recursive하게 더하여 IIR 계산
	{
		result[i] = (b0*data[i]) + (b1*data[i-1]) + (b2*data[i-2]) - (a1*result[i-1]) - (a2*result[i-2]) ;
	}
	
	return result[datalength-1]; // 최종적으로 얻은 y[n]값 넘겨주기
}


volatile int n_cnt = 0; // Timer 조정을 위한 변수 언언

float TH_a1, TH_a2, TH_b0, TH_b1, TH_b2; // IIR LPF 전역 변수
int TH_Var = 0; // 더한 횟수 count할 변수
int TH_Var_flag = 0; // flag
int TH_array[5]; // 계수 구하는 식에 넘겨줄 Thermistor 값 저장할 buffer

void TH_Coeff(float Fc)
{
	float Wc = 2*3.14*Fc; // Cut off Frequency
	float tau = 1/Wc;    // 시정수
	float ts = 1/Fc;     // Sampling time
	
	TH_a1 = -tau/(tau+ts);
	TH_a2 = 0;
	TH_b0 = ts/(tau+ts);
	TH_b1 = 0;
	TH_b2 = 0;
}


float CDS_a1, CDS_a2, CDS_b0, CDS_b1, CDS_b2; // IIR LPF 변수
int CDS_Var = 0;
int CDS_Var_flag = 0;
int CDS_array[5];

void CDS_Coeff(float Fc)
{
	float Wc = 2*3.14*Fc;
	float tau = 1/Wc;
	float ts = 1/Fc; // Sampling time
	
	CDS_a1 = -tau/(tau+ts);
	CDS_a2 = 0;
	CDS_b0 = ts/(tau+ts);
	CDS_b1 = 0;
	CDS_b2 = 0;
}


float ADC_a1, ADC_a2, ADC_b0, ADC_b1, ADC_b2; // FIR LPF 변수
int ADC_Var = 0;
int ADC_Var_flag = 0;
int ADC_array[5];

void ADC_Coeff(float Fc)
{
	float Wc = 2*3.14*Fc;
	float tau = 1/Wc;
	float ts = 1/Fc; // Sampling time
	
	ADC_a1 = -tau/(tau+ts);
	ADC_a2 = 0;
	ADC_b0 = ts/(tau+ts);
	ADC_b1 = 0;
	ADC_b2 = 0;
}


float PSD_a1, PSD_a2, PSD_b0, PSD_b1, PSD_b2; // FIR LPF 변수
int PSD_Var = 0;
int PSD_Var_flag = 0;
int PSD_array[5];

void PSD_Coeff(float Fc)
{
	float Wc = 2*3.14*Fc;
	float tau = 1/Wc;
	float ts = 1/Fc; // Sampling time
	
	PSD_a1 = -tau/(tau+ts);
	PSD_a2 = 0;
	PSD_b0 = ts/(tau+ts);
	PSD_b1 = 0;
	PSD_b2 = 0;
}

float LM_a1, LM_a2, LM_b0, LM_b1, LM_b2; // FIR LPF 변수
int LM_Var = 0;
int LM_Var_flag = 0;
int LM_array[5];

void LM_Coeff(float Fc)
{
	float Wc = 2*3.14*Fc;
	float tau = 1/Wc;
	float ts = 1/Fc; // Sampling time
	
	LM_a1 = -tau/(tau+ts);
	LM_a2 = 0;
	LM_b0 = ts/(tau+ts);
	LM_b1 = 0;
	LM_b2 = 0;
}

float Sonic_a1, Sonic_a2, Sonic_b0, Sonic_b1, Sonic_b2; // FIR LPF 변수
int Sonic_Var = 0;
int Sonic_Var_flag = 0;
int Sonic_array[5];

void Sonic_Coeff(float Fc)
{
	float Wc = 2*3.14*Fc;
	float tau = 1/Wc;
	float ts = 1/Fc; // Sampling time
	
	Sonic_a1 = -tau/(tau+ts);
	Sonic_a2 = 0;
	Sonic_b0 = ts/(tau+ts);
	Sonic_b1 = 0;
	Sonic_b2 = 0;
}

// Sensor

ISR(TIMER2_OVF_vect)
{
	TCNT2 = 241;  // Overflow 후에도 초기값 고정하기 위해 지속적으로 초기값 설정
	n_cnt++;
	
	if(n_cnt == 20)
	{
		float nData0 = GetADC0(); // Thermistor 값 받아오기
		// 수치 변환
		int nData  = 0 ;
		double R_th, T;
		double R_10 = 4700;
		double R_o  = 1000;
		double T_o  = 298.15;
		double B = 3650;
		
		R_th = R_10 * ( (1023 / (double)nData0) - 1); // 666.18
		T = 1 / ((1/T_o)+ ((1/B)*log(R_th/R_o)));         //308.38
		nData = int(T - 273.15); // 35
		
		// IIR filter에 넘겨줄 값들 모아주기
		if(TH_Var  != 5)
		{
			TH_array[TH_Var] = nData;
			TH_Var++;
		}
		else
		{
			TH_Var_flag =1; // 해당 개수만큼 모았으면 flag set
		}
		
		if(TH_Var_flag == 1) // flag = 1 -> if 문 진입
		{
			// --------------------------------------------------- Moter Control
			float Fc; // 차단할 주파수 선택
			int TH_result =0; // IIR return 값 받을 변수
			Fc = 10; // 차단 주파수 10Hz
			TH_Coeff(Fc); // Fc를 이용하여 IIR 계수 구하기
			TH_result = IIR(TH_array, sizeof(TH_array)/sizeof(int), TH_a1, TH_a2, TH_b0, TH_b1, TH_b2); // 구해온 IIR 계수와 모아둔 array 넘겨주기
			
			// 출력은 1번 Timer를 Fast PWM모드를 이용한 서보 모터 제어
			if(TH_result > 50) // 온도 40 이상  -> 90도로 이동
			{
				// 진입 확인
				OCR1A = 24;
			}
			else // 온도 20 이하 -> 0도로 이동
			{
				OCR1A = 8;
			}
			// --------------------------------------------------- Show Serial Chart
			int count = 0;
			int nData2 = TH_result;
			
			while(nData2 != 0)
			{
				nData2 = nData2/10;
				++count;
			}
			
			unsigned char nDataStr[10];
			
			int i = 0;
			while(TH_result != 0)
			{
				nDataStr[count-i-2]=TH_result%10+48;
				TH_result=TH_result/10;
				i++;
			}
			
			int k = 0;
			while( k < count)
			{
				USART0_TX_vect(nDataStr[k-1]); // 변환된 ADC 값을 USART로 송신 -> Serial 모니터에 나타날 것
				k++;
			}
			
			TH_Var = 0;
			TH_Var_flag = 0;
			USART0_TX_vect(0x60);
			USART0_TX_vect(0x43);
			USART0_TX_vect(0x2C);
		}
	}
	
	// 가변저항 + melody buzzer / 초음파 + LED 밝기 조절
	if(n_cnt == 40)
	{
		int Var0 = GetADC2();
		// 수치변환
		float fVoltage = (Var0*5.0) / 1023.0 ; // 가변저항 값 전압값으로 변환해주기
		Var0 = fVoltage*500;
		
		if(ADC_Var != 5)
		{
			ADC_array[ADC_Var] = Var0;
			ADC_Var++;
		}
		else
		{
			ADC_Var_flag = 1;
		}
		
		if(ADC_Var_flag == 1)
		{
			float Fc;
			int ADC_result=0;
			Fc = 10; // 차단 주파수 10 Hz로 설정
			ADC_Coeff(Fc);
			ADC_result = IIR(ADC_array, sizeof(ADC_array)/sizeof(int), ADC_a1, ADC_a2, ADC_b0, ADC_b1, ADC_b2);
			
			int melody = 30; // Buzzer를 몇번 울릴 것인지 결정하는 변수
			int k;
			unsigned int distance = 0; // 초음파 거리를 위한 변수
			
			if(ADC_result < 1000) // 가변저항 값이 1000이하 라면 초음파 센서를 사용하여 거리를 감지
			{
				// 1번 Timer는 서보 모터 PWM 제어에 사용하므로 3번 Timer를 이용하여 PWM 제어
				cli(); // 안정적으로 제어하기 위해 전체 인터럽트 잠시 중단
				int n;
				for(n = 0; n<5; n++) // IIR Filter에 넣어줄 배열을 만들어주기 위한 for문
				{
					TCCR3A = 0x00;  // WGM , COM, CS 제어
					TCCR3B = 0x03;  // Normal Count mode + 64 분주 + OC Pin disconnect
					// Trigger Pulse Generator PIN : 10us Pulse 생성 -> 센서 : 8개 40kHz pulse 생성
					PORTE |= (1<<TRIG);   // 6 PIN ON
					_delay_us(10); // 최소 10us 동안 전압 인가해주기
					// Trigger Pulse Stop
					PORTE &= ~(1<<TRIG); // 6 PIN OFF
					
					// 센서에서 자동으로 8개 40kHz Pulse 생성해서 쏜다
					
					// Echo Pin에서 Rising Edge를 기다린다 -> Rising Edge감지되면 ECHO Signal 발생
					
					while(!(PINE & (1<<ECHO))); // Echo Signal 발생 안하면 timer x
					TCNT3 = 0x0000; // timer reset
					
					while(PINE & (1<<ECHO));    // Echo Signal 발생 -> timer count start 물체까지 다녀온 시간을 이용하여 거리를 계산할 것
					TCCR3B = 0x00; // Timer Count Stop 후 Count 된 TCNT값 사용할 것
					
					// 거리 수치변환 식 : 거리 = 속도 x 시간 -> 속도는 음속으로 340 m/s, 시간은 분주비와 16Mhz를 고려하여 1us로 맞추어 계산한다.
					// 1cm를 이동하는데 약 29us.
					// 1000을 곱하여 준 것은 값을 소수점 까지 가져와서 보기 위함
					distance = (unsigned int)(SOUND_VELOCITY * (TCNT3 * 4 / 2) / 1000);
					
					Sonic_array[Sonic_Var] = distance;
					Sonic_Var++;
				}
				
				float Fc;
				int Sonic_result = 0;
				Fc = 10; // 차단 주파수 10Hz로 설정
				Sonic_Coeff(Fc);
				Sonic_result = IIR(Sonic_array , sizeof(Sonic_array)/sizeof(int), Sonic_a1, Sonic_a2, Sonic_b0, Sonic_b1, Sonic_b2);
				

				TCCR3A = 0x81; // Fast PWM Mode, Clear on match for LED 밝기 변화, OCR3A 출력 PIN인 PE3 PIN 사용
				TCCR3B = 0x0B; // 64 분주로 설정
				
				if(Sonic_result > 200)
				{
					OCR3A = 5; // 거리가 20cm 이상이라면 밝기 작게
				}
				else if(Sonic_result <= 200)
				{
					OCR3A = 60000; // 거리가 20cm 이하라면 아주 밝게
				}
				
				Sonic_result = Sonic_result/10; // cm로 다시 환산
				
				int count2 = 0;
				int nData3 = Sonic_result;
				
				while(nData3 != 0)
				{
					nData3 = nData3/10;
					++count2;
				}
				
				unsigned char nDataStr2[20];
				
				int i = 0;
				while(Sonic_result != 0)
				{
					nDataStr2[count2-i-1]=Sonic_result%10+48;
					Sonic_result=Sonic_result/10;
					i++;
				}
				
				int p = 0;
				while( p < count2)
				{
					USART0_TX_vect(nDataStr2[p]); // 변환된 ADC 값을 USART로 송신 -> Serial 모니터에 나타날 것
					p++;
				}
				
				Sonic_Var = 0;
				// cm
				USART0_TX_vect(0x63);
				USART0_TX_vect(0x6D);
				USART0_TX_vect(0x2C);
				sei();
			}
			
			else if(ADC_result >= 1000 && ADC_result < 2000) // 만일 가변저항 값이 1000 이상 2000 이하 라면 Passive Buzzer로 멜로디 생성 도미레파파
			{cli(); // 안정적인 출력을 위해 전체 인터럽트 중단
				for(k = 0; k< melody; k++)
				{      // 도
					PORTB = 0x01;
					_delay_ms(1.9);
					PORTB = 0x00;
				_delay_ms(1.9);         }
				_delay_ms(500);
				for(k = 0; k< melody; k++)
				{      // 미
					PORTB = 0x01;
					_delay_ms(1.5);
					PORTB = 0x00;
				_delay_ms(1.5);        }
				_delay_ms(500);
				for(k = 0; k< melody; k++)
				{      // 레
					PORTB = 0x01;
					_delay_ms(1.7);
					PORTB = 0x00;
				_delay_ms(1.7);         }
				_delay_ms(500);
				for(k = 0; k< melody; k++)
				{      // 파
					PORTB = 0x01;
					_delay_ms(1.42);
					PORTB = 0x00;
				_delay_ms(1.42);         }
				for(k = 0; k< melody; k++)
				{      // 파
					PORTB = 0x01;
					_delay_ms(1.42);
					PORTB = 0x00;
				_delay_ms(1.42);         }
				sei();
			}
			
			else       // 가변저항 값이 2000 이상이라면 Buzzer off, 초음파 센서 off
			{
				PORTB = 0x00; // Buzzer off
			}
			
			int count = 0;
			int nData2 = ADC_result;

			while(nData2 != 0)
			{
				nData2 = nData2/10;
				++count;
			}
			unsigned char nDataStr[10];
			
			int i = 0;
			while(ADC_result != 0)
			{
				nDataStr[count-i-2]=ADC_result%10+48;
				ADC_result=ADC_result/10;
				i++;
			}
			
			int h = 0;
			while( h < count)
			{
				USART0_TX_vect(nDataStr[h-1]); // 변환된 ADC 값을 USART로 송신 -> Serial 모니터에 나타날 것
				h++;
			}
			ADC_Var = 0;
			ADC_Var_flag = 0;
			// ohm
			USART0_TX_vect(0x6F);
			USART0_TX_vect(0x68);
			USART0_TX_vect(0x6D);
			USART0_TX_vect(0x2C);
		}
	}
	
	if(n_cnt == 60) // PSD + Active & Passive Buzzer melody
	{
		int PSD0 = GetADC3();
		int PSD;
		// 수치변환
		double fVoltage = PSD0*5.0 / 1023.0; // PSD 값을 전압값으로 바꿔주기
		
		if(fVoltage > 3.0)
		{
			fVoltage = 3.0; // 100cm 이상은 감지하지 못하며, 이 이상은 3.0v 로 고정
		}
		
		else if(fVoltage < 0.4)
		{
			fVoltage = 0.4; // 마찬가지로 7.14cm 이하는 감지하지 못하며, 0.4v 로 고정
		}

		fVoltage = fVoltage - 0.4; // 0.4v를 0v로 치환하여 0.4v ~ 3.0v 를 0.0 ~ 2.6v로 고정
		
		// 거리의 역수 형태로 변경 (Volt * 최대거리 역수와 최소거리 역수의 차 /전압차), 0.4 거리의 역수를 더한다.
		fVoltage = (fVoltage*0.13/2.6) + 0.01;
		
		PSD = int(1.0 / fVoltage ); // 역수를 취해서 cm로 환산
		
		if(PSD_Var != 5)
		{
			PSD_array[PSD_Var] = PSD;
			PSD_Var++;
		}
		else
		{
			PSD_Var_flag = 1;
		}
		
		if(PSD_Var_flag == 1)
		{
			float Fc;
			int PSD_result = 0;
			Fc = 20; // 차단 주파수 20Hz
			PSD_Coeff(Fc);
			PSD_result = IIR(PSD_array, sizeof(PSD_array)/sizeof(int), PSD_a1, PSD_a2, PSD_b0, PSD_b1, PSD_b2);
			
			if(PSD_result < 10) // PSD 감지거리가 10cm 이하라면 진입
			{
				cli(); // 해당 출력을 안전하게 보기 위해 SREG 7 bit = 0
				int j, a;

				
				for(j = 0; j< 3; j++)
				{      // Passive Buzzer 소리 : 도
					for(a = 0; a < 5; a++)
					{
						PORTB = 0x02;
						_delay_ms(50);
						PORTB = 0x00;
						_delay_ms(50);
					}  // Active Buzzer
					for(a = 0; a < 60; a ++)
					{
						PORTB = 0x01;
						_delay_ms(1.9);
						PORTB = 0x00;
						_delay_ms(1.9);
					}
					_delay_ms(300);
				}
				sei(); // 다시 SREG 7 bit = 1로 Set
			}
			
			int count = 0;
			int nData2 = PSD_result;
			while(nData2 != 0)
			{
				nData2 = nData2/10;
				++count;
			}
			
			unsigned char nDataStr[10];
			
			int i = 0;
			while(PSD_result != 0)
			{
				nDataStr[count-i-2]=PSD_result%10+48;
				PSD_result=PSD_result/10;
				i++;
			}
			
			int k = 0;
			while( k < count)
			{
				USART0_TX_vect(nDataStr[k-1]); // 변환된 ADC 값을 USART로 송신 -> Serial 모니터에 나타날 것
				k++;
			}
			
			PSD_Var = 0;
			PSD_Var_flag = 0;
			//cm
			USART0_TX_vect(0x63);
			USART0_TX_vect(0x6D);
			USART0_TX_vect(0x2C);
		}
	}
	
	if(n_cnt == 80)
	{
		double LM350 = GetADC4();
		int LM35;
		// ------------------------- 수치변환
		LM350 = 10*LM350; // 전압값으로 변환
		LM35 = LM350/100;
		// ------------------------- 수치변환
		if(LM_Var != 5)
		{
			LM_array[LM_Var] = LM35;
			LM_Var++;
		}
		else
		{
			LM_Var_flag = 1;
		}
		if(LM_Var_flag == 1) // Flag Set 됬다면 진입
		{
			float Fc; // 차단 주파수
			int LM_result = 0;
			Fc = 20; // 20 Hz로 설정
			LM_Coeff(Fc);
			LM_result = IIR(LM_array, sizeof(LM_array)/sizeof(int), LM_a1, LM_a2, LM_b0, LM_b1, LM_b2);
			
			if(LM_result > 7) // 만일 LM35 값이 7보다 크다면 진입
			{
				PORTA = 0x00; // LED ON
			}
			else
			{
				PORTA = 0xff; // 그렇지 않다면 LED Off
			}
			int count = 0;
			int nData2 = LM_result;

			while(nData2 != 0)
			{
				nData2 = nData2/10;
				++count;
			}

			unsigned char nDataStr[10];

			int i = 0;
			while(LM_result != 0)
			{
				nDataStr[count-i-2]=LM_result%10+48;
				LM_result=LM_result/10;
				i++;
			}

			int k = 0;
			while( k < count)
			{
				USART0_TX_vect(nDataStr[k-1]); // 변환된 ADC 값을 USART로 송신 -> Serial 모니터에 나타날 것
				k++;
			}

			LM_Var = 0;
			LM_Var_flag = 0;
			USART0_TX_vect(0x60);
			USART0_TX_vect(0x43);
			USART0_TX_vect(0x2C);
		}
		
	}
	
	if(n_cnt == 100) // CDS + 3 color LED
	{
		float CDS0 = GetADC1(); // ADC 값 받아오기
		int CDS;
		// 수치변환
		
		float R_9 = 4.7;
		float AVCC = 1023.0;
		float R_CDS;
		float gamma = 0.8;
		
		R_CDS = R_9*((AVCC/CDS0) - 1);
		
		CDS = pow ( 10 , 1 - ( ( log(R_CDS) - log(40) ) / gamma) ); // 수치 변환한 값

		// IIR filter에 넣어줄 배열 만들기
		if(CDS_Var != 5)
		{
			CDS_array[CDS_Var] = CDS;
			CDS_Var++;
		}
		else // 넣어줄 배열이 완성되면 flag set
		{
			CDS_Var_flag = 1;
		}
		
		if(CDS_Var_flag == 1)
		{
			float Fc;
			int CDS_result = 0;
			Fc = 10; // 차단 주파수 : 10hz
			CDS_Coeff(Fc);
			CDS_result = IIR(CDS_array, sizeof(CDS_array)/sizeof(int), CDS_a1, CDS_a2, CDS_b0, CDS_b1, CDS_b2);
			
			if(CDS_result < 100) // 만일 정제된 CDS 값이 100보다 작다면 RGB LED 출력
			{
				cli();
				PORTC = 0x01;
				_delay_ms(300);
				PORTC = 0x02;
				_delay_ms(300);
				PORTC = 0x04;
				sei();
			}
			else // 그렇지 않다면 RGB LED Off
			{
				PORTC = 0x00;
			}

			// USART 통신 코드
			int count = 0;
			int nData2 = CDS_result;
			
			// 정수형 데이터 Char 형 데이터로 바꾸기
			// 바꿀 데이터의 길이 구하기
			while(nData2 != 0)
			{
				nData2 = nData2/10;
				++count;
			}
			
			// 데이터 담을 버퍼 선언
			unsigned char nDataStr[10];
			
			int i = 0;
			// 한 자리씩 쪼개서 버퍼에 담기
			while(CDS_result != 0)
			{
				nDataStr[count-i-2]=CDS_result%10+48;
				CDS_result=CDS_result/10;
				i++;
			}
			
			int k = 0;
			// 버퍼에 담긴 순서가 반대이므로, 거꾸로 하나씩 보내주기
			while( k < count)
			{
				USART0_TX_vect(nDataStr[k-1]); // 변환된 ADC 값을 USART로 송신 -> Serial 모니터에 나타날 것
				k++;
			}
			
			// IIR 변수 초기화
			CDS_Var = 0;
			CDS_Var_flag = 0;
			// USART 단위, 줄바꿈 문자 출력
			USART0_TX_vect(0x4C);
			USART0_TX_vect(0x55);
			USART0_TX_vect(0x58);
			USART0_TX_vect(0x0D);
		}
		n_cnt = 0; // ISR loop 종료
	}
}


int main(void)
{
	// 모터 제어 -> 1번 Timer로
	// LED 밝기 제어 -> 2번 Timer로
	// 센서 ISR 제어 -> 0번 Timer로
	
	// LED
	DDRA = 0xff;
	PORTA = 0xff;
	
	// Piezo Buzzer (Active + Passive) & Servo Motor
	DDRB = 0xff;  // Thermistor PINB 5 출력 모드로 -> PWM 출력 -> PORT B로 서보모터 제어할 것
	PORTB = 0x00; // Interrupt 1 : 0도 Interrupt 2 : 180도 , 기본 : 90도
	
	// 3 Color LED
	DDRC = 0xff;
	PORTC = 0x00;
	
	// Sonic
	DDRE = 0x48;  // PE 7 : Echo PIN, PE6 : Trigger PIN
	
	// PSD
	DDRF = 0x00;
	PORTF = 0x00;
	
	// 1번 Timer : 모터제어    ->  Fast PWM 모드 : 주기 20ms
	TCCR1A = 0x82; // 1000 0011 -> Clear Compare match == Fast PWM  / Top : OCR , Compare match : TOP
	TCCR1B = 0x1D; // 0001 1101

	// 3번 Timer : LED 밝기 제어
	TCCR3A = 0x81; // Fast PWM, Clear on match
	TCCR3B = 0x0B; // 64 분주
	
	// 1번 Timer Initial Count Value
	TCNT1H = 0x00;
	TCNT1L = 0x00;
	ICR1 = 312; // ICR 값과 매치되면 Clear
	OCR1A = 8; // Initial 90도
	
	// Interrupt Selection
	TIMSK = 0x54; // TOIE2 , TOIE1 , OCIE1A Set  // 서보 모터 제어를 위한 PWM Interrupt
	ETIMSK = 0x10; // OCIE3A enable // LED 밝기 제어를 위한 PWM Interrupt
	
	// ADC 관련
	DDRF = 0x00; // F Port ADC 값 받아올 센서 선정
	PORTF = 0x00; // F PORT : ADC Sensor Pin -> 센서 출력 값을 Atmega로 받을 거니까 PORT모두 출력으로 Setting
	
	// USART 관련
	UBRR0L = (unsigned char) UBRR; // Baud rate 9600
	UBRR0H = (unsigned char) (UBRR << 8);

	UCSR0A = 0x00; // 비동기 1배속
	UCSR0B = (1 << RXEN0) | (1<<TXEN0); // 송신 ok, 수신 ok
	UCSR0C = (1<<UCSZ01) | (1 << UCSZ00);  // 비동기, No Parity bit , Stop bit : 1bit, data : 8 bit
	
	// 2번 Timer : ADC 센서 값 -> Fast PWM 모드 : 주기 10ms
	TCCR2 = 0x4D;  // 0100 1111 : Fast PWM Mode + Normal Count + 1024 분주
	TCNT2 = 241; // clock 156번 들어오면 overflow 발생 -> ISR 진입    (10ms)

	// A/D 동작 Enable
	ADCSRA = (1 << ADEN) | (1 << ADPS2) | (1 << ADPS1) | (1 << ADPS0);

	sei(); // Global Interrupt Enable
	while (1);
}

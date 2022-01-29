/*
 * Author : KHS
 * Revised : LJS
 */ 
 
#include "mcu_init.h"

void InitIO(){

	// DDRx : Set Input / Output Mode --- 0 : Input  , 1 : Output 
	
	DDRA = 0xFF; // LED
	DDRC = 0xFF; 
	DDRD = 0x08; // SWITCH --- 외부 인터럽트 
	DDRB = 0x67; // PWM & SPI
	DDRE = 0x1A; // USART / ENCODER

	PORTA = 0x00; // LED
	PORTB = 0x07; // PWM , SPI
}

void InitExtInt(){
	
	//TO DO
	DDRD = 0x00; // D PORT 모두 INPUT Mode
	
	EICRA = INT1_FALLING | INT0_FALLING;  // External Interrupt Falling Edge On
	
	EIMSK = INT1_ENABLE | INT0_ENABLE; // External Interrupt 0 1 Enable
}

void InitTimer0(){
	
	//TO DO	
	TCCR0 = 0x04; // 64 분주 / Normal Count  / OC PIN Disconnected 
	TIMSK = 0x01; // OCIE Enable 
	
}

void InitTimer1(){
	
	//TO DO
	TCCR1A = 0b11100010; // 0xE2 // A : Set OC on Compare Match on Up Counting. Clear OC on Down Counting // B : Clear OC on Compare Match  // C : OC Disconnected 
	TCCR1B = 0b00010001; // 0x11 ---- PC PWM Mode (Top : ICR ) // 분주비 1 
		
	ICR1 = 399; // Top : ICR (ICR 값까지 Count)
	OCR1C = 0;
		
	OCR1A = 0;		//1 L
	OCR1B = 0;		//2 L
	
	TCNT1 = 0;
}

void InitTimer2(){
	
	//TO DO
}

void InitTimer3(){
	
	//TO DO
	//TCCR3A = 0b00000000;
	//TCCR3B = 0b00000011;

	//ETIMSK = 0x04;
 
	TCCR3A = 0b10110010;  // COMA : Clear COMB : Set  // 분주비 1 --- 1clock 1/16MHZ 초 
	TCCR3B = 0b00010001;  // PC PWM 
		
	ICR3 = 399;
	OCR3C = 0;
	
	OCR3A = 0;		//1 L
	OCR3B = 0;		//2 L
		
	TCNT3 = 0;
}

void InitADC(){
	
	//TO DO
	ADMUX = 0x40;  // 0100 0000  - ADC 0번 채널 
	ADCSRA = 0x86; // 1000 0110  - ADC Enable + 64 분주 
	
}

int GetADC(char ch){
	
	//TO DO
	ADMUX = (ADMUX & 0xf0) + ch; 
	ADCSRA |= 0x40; // ADSC = 1 --- AD 시작 
	while(!(ADCSRA & 0x10)); // ADIF = 1 --- AD 완료 ISR 
	return ADC;
}


void InitUart0(){
	
	//TO DO
	//Uart
	UCSR0A = 0x00; // Buffer 상태 Register
	UCSR0B = 0x98; // 통신 Interrupt ... 1001 1000 = 수신 완료 ISR + 송수신 동작 ok 
	UCSR0C = 0x06; // 0110 : Stop bit 1개 + 8 bit data 전송 
	
	UBRR0L = 103; // Baud Rate : 9600 
	
}

void InitUart1(){
	
	//TO DO
	DDRD = (DDRD & 0xF3) | 0x08;
	
	UCSR1A = 0x00;
	UCSR1B = USART_RECV_ENABLE | USART_TRANS_ENABLE;
	UCSR1C = USART_CHAR_SIZE_8BIT;
	
	UBRR1L = USART_115200BPS;
}

void TransUart0(unsigned char data){
	
	//TO DO
	while(!(UCSR0A & 0x20));
	UDR0 = data;
}

void TransUart1(unsigned char data){
	
	//TO DO
	while(!(UCSR1A & 0x20));
	UDR1 = data;
}

unsigned char RecvUart0(){
	
	//TO DO
	//while(!(UCSR0A&(1<<RXC0))); // 코드 추가
	return UDR0;
}

unsigned char RecvUart1(){
	
	//TO DO	
	//while(!(UCSR1A&(1<<RXC1))); // 코드 추가	
	return UDR1;
}


void TransNumUart0(int num){
	
	//TO DO
	if(num < 0){
		TransUart0('-');
		num *= -1;
	}
	
	TransUart0( ((num%10000000) / 1000000) + 48);
	TransUart0( ((num%1000000) / 100000) + 48);
	TransUart0( ((num%100000) / 10000) + 48);
	TransUart0( ((num%10000) / 1000) + 48);
	TransUart0( ((num%1000) / 100) + 48);
	TransUart0( ((num%100) / 10) + 48);
	TransUart0( num%10 + 48 );
}

void SendShortUART0(int16_t num){
	
	if(num < 0){
		TransUart0('-');
		num *= -1;
	}

	TransUart0( ((num%100000) / 10000) + 48);
	TransUart0( ((num%10000) / 1000) + 48);
	TransUart0( ((num%1000) / 100) + 48);
	TransUart0( ((num%100) / 10) + 48);
	TransUart0( num%10 + 48 );
}


void TransNumUart1(int num){
	
	//TO DO
	if(num < 0){
		TransUart1('-');
		num *= -1;
	}
	
	TransUart1( ((num%10000000) / 1000000) + 48);
	TransUart1( ((num%1000000) / 100000) + 48);
	TransUart1( ((num%100000) / 10000) + 48);
	TransUart1( ((num%10000) / 1000) + 48);
	TransUart1( ((num%1000) / 100) + 48);
	TransUart1( ((num%100) / 10) + 48);
	TransUart1( num%10 + 48 );
}

void SendShortUART1(int16_t num){
	
	if(num < 0){
		TransUart1('-');
		num *= -1;
	}

	TransUart1( ((num%100000) / 10000) + 48);
	TransUart1( ((num%10000) / 1000) + 48);
	TransUart1( ((num%1000) / 100) + 48);
	TransUart1( ((num%100) / 10) + 48);
	TransUart1( num%10 + 48 );
}

void InitSPI(){
	
	SPCR = 0x50; // 0101 0000 -- SPI 활성화 + 마스터 모드 
	SPSR = 0x01; // 0000 0001 -- 마스터 모드일 경우 SPI 속도 2배 
}

void SPI_MasterSend(unsigned char data){
	
	SPDR = data; // SPI 데이터 레지스터 - 여기에 데이터를 쓰면 전송이 시작 
	while (!(SPSR & 0x80)); // 전송이 완료되면 7 bit Set + ISR 발생 가능 
	data = SPDR; // 데이터 읽기 
}

unsigned char SPI_MasterRecv(void)
{
	SPDR = 0x00; // 데이터 레지스터에 0 값
	while (!(SPSR & 0x80)); // 전송이 완료되면 ..
	return SPDR; // 데이터 레지스터 값 return 
}

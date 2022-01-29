/*
 *  Author: KHS
 *  Revised : LJS
 */ 


#ifndef DATATYPE_H_
#define DATATYPE_H_

typedef struct _packet_data{
	
	unsigned char header[4];
	unsigned char size, id;
	unsigned char mode, check;

	int32_t pos;
	int32_t velo;
	int32_t cur;
	
}Packet_data_t;


// Union --- Packet 통신에 자주 사용하고 메모리를 공유하려고 할 때 사용 - Struct 보다 효율적으로 메모리 사용이 가능 
// 가장 큰 변수 기준으로 메모리 할당 
typedef union _packet{ 
	
	Packet_data_t data; // Packet_data_t type struct 객체 생성 
	unsigned char buffer[sizeof(Packet_data_t)];
	
}Packet_t; // alias 


#endif /* DATATYPE_H_* */

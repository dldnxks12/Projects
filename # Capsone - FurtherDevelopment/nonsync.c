#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>  // c compiler
#include <sys/types.h>
#include <termios.h> // for serial communication
#include <fcntl.h>   // getting or setting properties of files (file control)
#define BUF_MAX 1024

int main(void){

	// use read / write / open / close with this file descriptor
	int fd = 0; // fd : file descriptor
	char buf[BUF_MAX];

	// Open Log file
	FILE *pFile = NULL;
	pFile       = fopen("log.txt", "r");

	struct termios newtio;

	// O_RDWR   : Read + Write
	// O_NOCTTY : Some flag aboout terminal control...
	fd = open("/dev/ttyUSB0", O_RDWR | O_NOCTTY);
	//fd = open("log2.txt", O_RDWR | O_NOCTTY);	
	if(fd < 0){ // file open fail ...
		fprintf(stderr, "ERROR\n");
		exit(-1); // process end
	}

	memset(&newtio, 0, sizeof(newtio)); // Assign dynamic memory

	// c_iflag : input mode
	// c_oflag : output mode
	// c_cflag : control mode
	// c_lflag : local mode
	// cc_t    : special characters

	// control mode
	newtio.c_cflag  = B115200; // Boad Rate
	newtio.c_cflag |= CS8;     // 8bit , parity X , Stopbit 1
	newtio.c_cflag |= CLOCAL;  // Use internal port
	newtio.c_cflag |= CREAD;   // READ , WRITE OKAY

	// input mode
	newtio.c_iflag = IGNPAR; // Ignore parity bit (No parity check)
	//newtio.c_iflag = ICRNL; // Carriage Return

	newtio.c_oflag = 0;
	newtio.c_lflag = 0;

	newtio.c_cc[VTIME] = 0; // Timeout in deciseconds
	newtio.c_cc[VMIN]  = 0; // Minumum number of characters

	// transmit info to file descriptor
	tcflush(fd, TCIFLUSH);
	tcsetattr(fd, TCSANOW, &newtio); // TCSANOW : change occurs immediately

	sleep(1);
	if (pFile != NULL){

		char strTemp[BUF_MAX];
		char *pStr = NULL;

		while(!feof(pFile)){
			pStr = fgets(strTemp, sizeof(strTemp),pFile);
			if(feof(pFile))
				break;
			printf("%s", strTemp);
			write(fd, strTemp, 30);
			fflush(stdout);
			usleep(120000); // 1.0 second
		}
		fclose(pFile);
	}else{
		fprintf(stderr, "Error\n");
		exit(-1);
	}

	close(fd);
	return 0;

}



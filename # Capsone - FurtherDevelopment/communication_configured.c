#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>  // c compiler
#include <sys/types.h>
#include <termios.h> // for serial communication
#include <fcntl.h>   // getting or setting properties of files (file control)

#define BUF_MAX 1024
#define FALSE 0
#define TRUE  1

volatile int STOP = FALSE;

int main(void){

	// use read / write / open / close with this file descriptor
	int fd = 0; // fd : file descriptor
	int fd2 = 0;
	int res; 
	char buf[BUF_MAX];

	struct termios newtio;
	
	// O_RDWR   : Read + Write
	// O_NOCTTY : Some flag aboout terminal control...

	fd  = open("log.txt", O_RDWR | O_NOCTTY);	    // Data IN	
	//fd2 = open("log2.txt", O_RDWR | O_NOCTTY);    // Data OUT
	fd2 = open("/dev/ttyUSB0", O_RDWR | O_NOCTTY);    // Data OUT
	
	//fd  = open("buf.txt", O_RDWR | O_NOCTTY);	    // Data IN		
	//fd2 = open("buf2.txt", O_RDWR | O_NOCTTY);	

	if(fd < 0){ // file open fail ...
		fprintf(stderr, "ERROR\n");
		exit(-1); // process end
	}

	memset(&newtio, 0, sizeof(newtio)); // Assign dynamic memory

	// control mode
	newtio.c_cflag  = B115200; // Boad Rate
	newtio.c_cflag |= CS8;
	newtio.c_cflag |= CLOCAL; // Use internal port
	newtio.c_cflag |= CREAD;  // READ , WRITE OKAY

	// input mode
	newtio.c_iflag = IGNPAR; // Ignore parity bit (No parity check)
	// newtio.c_iflag = ICRNL // Carriage Return

	newtio.c_oflag = 0;
	newtio.c_lflag = 0;

	newtio.c_cc[VTIME] = 0; // Timeout in deciseconds
	newtio.c_cc[VMIN]  = 0; // Minumum number of characters

	// transmit info to file descriptor
	tcflush(fd, TCIFLUSH);
	tcsetattr(fd, TCSANOW, &newtio); // TCSANOW : change occurs immediately

	// Wait 1 seconds ... 
	sleep(1);	

	while(STOP == FALSE)
	{	
		res = read(fd, buf, 30);
		if (res < 30)
			continue;		
		printf("%s\n", buf);
		buf[res-1] = 0;    
		write(fd2, buf, 30);				
		fflush(stdout);
		usleep(50000); // 0.5 second
	}	
	
	close(fd);
	close(fd2);
	
	return 0;

}



from serial import Serial
import time
import os


ser=Serial('/dev/ttyUSBPC',115200) 

print("Check point 1 : file open & serial port open check")

while True:
	f=open("log.txt","r") 
	if os.stat("log.txt").st_size == 0:
		f.close()
		continue
	else:
		print("log file is not empty, sending start")
		f.close()
		break

print("Check point 2 : empty check")

#for line in f:
#	print("Check")
#	current_line = line
#	encoded_line = line.encode('utf-8')
#	ser.write(encoded_line)
#	time.sleep(0.1)
#	print(current_line)

while True:
	time.sleep(0.1)
	with open('log.txt', 'rb') as f:
		try:  # catch OSError in case of a one line file 
			f.seek(-2, os.SEEK_END)
			while f.read(1) != b'\n':
				f.seek(-2, os.SEEK_CUR)
		except OSError:
			f.seek(0)
		last_line = f.readline().decode()		
		encoded_line = last_line.encode('utf-8')
		ser.write(encoded_line)
		print(last_line)

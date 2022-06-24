import os
import sys
from multiprocessing import Process

def start_server(program):    
    os.system(program)

#programs = ["python3 test.py", "python3 test2.py"]
#programs = ["python3 0616_2_segmentation.py", "./a.out"]
#programs = ["python3 0618_HSV_Canny.py", "./a.out"]
programs = ["python3 0618_HSV_Canny_weight.py", "./a.out"]

if __name__ == '__main__':
    for program in programs:
        proc = Process(target  = start_server, args = (program,))        
        proc.start()



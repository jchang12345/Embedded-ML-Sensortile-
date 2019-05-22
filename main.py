import serial
import time
import sys
import numpy
import subprocess as sp
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import SensorTile_Serial
from gen_model import gen_model
from realtime import RealTime_Classifier

'''****IN ORDER TO RUN:****
python3 main.py /dev/cu.usb[serialportaddr]

'''

#  Serial setup according to command line arguments
numberruns=0
python_version = sys.version[0]
if len(sys.argv) != 2:
    print ("Wrong Number of Arguments!")
    print ("Please use format: python SensorTile_Animation_args.py SerialAddress")
else:
    address = sys.argv[1]
    if python_version == "2":
        python3 = False
    else:
        python3 = True

baud_rate = 9600
timeout = 2


#main program
if __name__ == "__main__":
	#initialize sensortile and serial connection, as well as # of input motions and time per motion
	sensortile = SensorTile_Serial.serial_SensorTile(address, baud_rate, timeout, python3)
	numofruns,times=sensortile.init_connection()

	#collect data
	sensortile.collect_data(numofruns,times)
	sensortile.collect_data(numofruns,times)


	#classify the data by creating a classifier
	clf=RealTime_Classifier()

	#train the classifier on csv data. 
	clf.train(['acc_data_1.csv','acc_data_2.csv'],40)
	
	#set the time to test here, real time testing input motion
	timetest=40
	numofruns=1

	#actually test the input motion.
	sensortile.collect_test_data(numofruns,timetest,clf)
	sensortile.close_connection()

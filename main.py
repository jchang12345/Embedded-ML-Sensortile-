#code that brings everything together
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



if __name__ == "__main__":
	sensortile = SensorTile_Serial.serial_SensorTile(address, baud_rate, timeout, python3)
	times=30#30 or bigger when needed
	numofruns=sensortile.init_connection(times)
	accelx, accely, accelz = sensortile.collect_data(numofruns,times)
	accelx, accely, accelz = sensortile.collect_data(numofruns,times)
	clf=RealTime_Classifier()

	clf.train(['acc_data_1.csv','acc_data_2.csv'],40)#gen_model()
	timetest=40
	numofruns=1
	#clf=RealTime_Classifier('model.h5')
	#clf.validate(['acc_data_1.csv','acc_data_2.csv'])
	accelx, accely, accelz = sensortile.collect_test_data(numofruns,timetest,clf)
	sensortile.close_connection()

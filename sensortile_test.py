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

#import the other files from other side




if __name__ == "__main__":
	sensortile = SensorTile_Serial.serial_SensorTile(address, baud_rate, timeout, python3)
	times=5
	numofruns,times=sensortile.init_connection(numofruns,times)
	accelx, accely, accelz = sensortile.collect_data(numofruns,times)
	#sensortile.init_connection(numofruns)
	accelx, accely, accelz = sensortile.collect_data(numofruns,times)
	sensortile.close_connection()
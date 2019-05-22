import serial
import time
import sys
import numpy
import os
from realtime import RealTime_Classifier


numbrofruns=0
times=0
class serial_SensorTile():
	def __init__(self, address, baud_rate=9600, timeout=2, python3=False):
		self.ser = None
		# serial information
		self.address = address
		self.baud_rate = baud_rate
		self.timeout = timeout
		# flag 
		self.data_check = 0
		self.python3 = python3

	def init_connection(self):
		print ("Start Serial Connection")
		try:
			ser = serial.Serial(self.address, self.baud_rate, timeout=self.timeout)
			numberofruns=input("Enter number of runs:")
			times=input("Enter time for runs:")
		except:
			print ("Wrong serial address and shut down system")
			sys.exit()
		self.ser = ser
		# sleep 500ms before accepting data
		time.sleep(0.5) 
		self.ser.flushInput()
		return numberofruns, times

	def close_connection(self):
		print ("Close Serial Connection")
		self.ser.close()
		
	def is_ready(self, bytes_expected):
		return self.ser.in_waiting >= bytes_expected

	def collect_data(self, numberruns,times):
		print(numberruns)
		if self.data_check:
			# read a line
			ser_bytes = self.ser.readline()
			# discard \r\n
			tmp = ser_bytes.rstrip()
			# convert byte to string python 3
			if self.python3:
				tmp = tmp.decode("utf-8")
			# split data
			data = tmp.split(',')
			try:
				print ("{}".format(data))
	
				accelx = float(data[0])
				accely = float(data[1])
				accelz = float(data[2])
				ttimes=times


				for i in range(int(numberruns[0])): 
					#preliminary time delay to 3 seconds, flushing out buffer
					t_firstend=time.time()+3
					while time.time()<t_firstend:
						ser_bytes = self.ser.readline()
						# discard \r\n
						tmp = ser_bytes.rstrip()
						if self.python3:
							tmp = tmp.decode("utf-8")
						# split data
						data = tmp.split(',')
						accelx = float(data[0])
						accely = float(data[1])
						accelz = float(data[2])
						print ("{}".format(data))
					
					#to check which motion we are working on.
					print("We are on motion " +str((i)+1))
					f=open(os.path.join("data","acc_data_"+str(i+1)+".csv"),"w") #write option on first
					print("Created file: acc_data_"+str((i)+1)+".csv")
					time.sleep(2) #small delay to note 

					t_end=time.time()+float(ttimes) 
					while time.time() < t_end:

						ser_bytes = self.ser.readline()
						tmp = ser_bytes.rstrip()
						# convert byte to string python 3
						if self.python3:
							tmp = tmp.decode("utf-8")
						# split data
						data = tmp.split(',')
						accelx = float(data[0])
						accely = float(data[1])
						accelz = float(data[2])
						print ("{}".format(data))

						#recording values of x y and z into csv file. f is the file descriptor for file we open corresponding to input
						f.write(str(accelx))
						f.write(",")
						f.write(str(accely))
						f.write(",")
						f.write(str(accelz))
						f.write('\n')

						
					print ("Finished motion : " +str((i)+1))
					print ("prepare for next motion \n")
					print ("prepare for next motion \n")
					print ("prepare for next motion \n")
					print ("prepare for next motion \n")
					print ("prepare for next motion \n")
					print ("prepare for next motion \n")
					print ("prepare for next motion \n")
					print ("prepare for next motion \n")
					f.close()

					#sleep for 3 seconds so we know we are prepping for next motion.
					time.sleep(3)


					#MAKE SURE TO CTRL C to exit this data collection phase of script.
				while 1:
					print("done with test. please ctrl c to exit data collect")

					tstall=time.time()+3
					time.sleep(3)
			except:
				print ("Wrong serial read:")
		else:
			# discard the first corrupted line
			self.ser.reset_input_buffer()
			self.ser.readline()
			self.data_check = 1
	def collect_test_data(self, numberruns,times,clf):
		print(numberruns)
		if self.data_check:
			# read a line
			ser_bytes = self.ser.readline()
			# discard \r\n
			tmp = ser_bytes.rstrip()
			# convert byte to string python 3
			if self.python3:
				tmp = tmp.decode("utf-8")
			# split data
			data = tmp.split(',')
			# str to float and store dis, accel
			try:
				print ("{}".format(data))
				accelx = float(data[0])
				accely = float(data[1])
				accelz = float(data[2])
				ttimes=times
				print("Running test for this amount of time in seconds: "+str(ttimes))
				sleep(2) #let them know how long u are about to run for

				t_firstend=time.time()+float(times)

				#initialize a counter to count 10 internal clk cycle. we could easily change this to match seconds instead. before each classification to slow it down
				counter=0
				while time.time()<t_firstend:
					ser_bytes = self.ser.readline()
					tmp = ser_bytes.rstrip()
					if self.python3:
						tmp = tmp.decode("utf-8")
					data = tmp.split(',')
					
					#data collection and update predicts based on this
					accelx = float(data[0])
					accely = float(data[1])
					accelz = float(data[2])
					clf.update([accelx, accely, accelz])
					if counter==10:
						print(clf.test()) #print prediction, motion 1 on first index, motion 2 on second index.
						counter=0
					counter+=1
					
			except:
				print ("Wrong serial read:")
		else:
			# discard the first corrupted line
			self.ser.reset_input_buffer()
			self.ser.readline()
			self.data_check = 1


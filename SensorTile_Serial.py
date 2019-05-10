import serial
import time
import sys
import numpy

numbrofruns=0

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

	def init_connection(self, numberofruns):
		print ("Start Serial Connection")
		try:
			ser = serial.Serial(self.address, self.baud_rate, timeout=self.timeout)
			numberofruns=input("Enter number of runs:")

		except:
			print ("Wrong serial address and shut down system")
			sys.exit()
		self.ser = ser
		# sleep 500ms before accepting data
		#time.sleep(0.5) #why is there a sleep 500 ms????
		self.ser.flushInput()
		print(numberofruns)
		return numberofruns

	def close_connection(self):
		print ("Close Serial Connection")
		self.ser.close()
		
	def is_ready(self, bytes_expected):
		return self.ser.in_waiting >= bytes_expected

	def collect_data(self, numberruns):
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
				#dis = float(data[0])
				#print(data)

				#ok so now, ask user to input like 1
				#then fopen foo.csv 1 
				#do that motion for abt a minute 

				#next step is to then, try to ask user to inpiut like 2
				#then fopen foo.csv 2 
				#do that motion for abt a minute. 


				accelx = float(data[0])
				accely = float(data[1])
				accelz = float(data[2])
				#he wants accelerometer x y z data. 
				#storeme=[]
				#storeme.append(dis)
				#storeme.append(str(','))
				#storeme.append(accel)
				print(numberruns)
				#we can try to debug this tmrw
				for i in range(int(numberruns)): #so if i said 2, then 0 1 , i want 1 2
					
					#below code will run it for 900 seconds
					#how do i indicate i want to run this for 1 minute
					#span a t_end
					#print(c)

					t_firstend=time.time()+5
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

					t_end=time.time()+5 #thats for 6 secs?, lets make it collect data for 
					# 6 seconds instead.
					print("hi")
					print("run \n")
					print(i)
					#rint(i)
					f=open("acc_data_"+str(i+1)+".csv","w") #write option on first

					#f=open("acc_data_"+str(i+1)+".csv","a") #append option on 
					print("acc_data_"+str(i+1)+".csv")
					while time.time() < t_end:
						#this is kinda wrong, i needa poll
						# and try to collect data inside this function.
						ser_bytes = self.ser.readline()
						# discard \r\n
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

					#numpy.savetxt(f, storeme)
						f.write(str(accelx))
						f.write(",")
						f.write(str(accely))
						f.write(",")
						f.write(str(accelz))
					#numpy.savetxt(f,storeme[1])
						f.write('\n')

						
					print("done with testing input ")
					print (i+1)
					print("\n")
					f.close()
					time.sleep(3)
					#import time

#t_end = time.time() + 60 * 15
#while time.time() < t_end:

				while 1:
					print("done with test. please ctrl z to exit")
					tstall=time.time()+3
					time.sleep(3)
					#exit()

				# print ("{}".format(dis))
				# print ("{}".format(accel))
				return accelx, accely, accelz
			except:
				print ("Wrong serial read:")
				return 0, 0, 0
		else:
			# discard the first corrupted line
			#print("corrupted 1st line")
			#print(numberruns)
			self.ser.reset_input_buffer()
			self.ser.readline()
			self.data_check = 1
			return 0, 0, 0


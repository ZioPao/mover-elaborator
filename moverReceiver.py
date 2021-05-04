import serial
import matplotlib.pyplot as plt
import re
import array

fig, ax = plt.subplots()


com_id = 2
mover = None
while mover is None:
    try:
        mover = serial.Serial('COM' + str(com_id))
        ser_bytes = mover.readline()
        decoded_bytes = ser_bytes.decode()
        if re.match("MASTER", decoded_bytes) is False:
            mover = None


    except:
        mover = None
        print(com_id)
        com_id += 1

print("Started!")

mover.flushInput()

accList = list()
gyrList = list()
for i in range(0,50):
    ser_bytes = mover.readline()
    decoded_bytes = ser_bytes.decode()

    if re.match("main", decoded_bytes):
        #Read main data
        ser_bytes = mover.readline()
        decoded_bytes = ser_bytes.decode()
        tmp = re.findall("(\S*),(\S*),(\S*),(\S*),(\S*),(\S*)", decoded_bytes)[0]
        acc = array.array('i', [int(tmp[0]), int(tmp[1]), int(tmp[2])])
        gyr = array.array('i', [int(tmp[3]), int(tmp[4]), int(tmp[5])])

        accList.append(acc)
        gyrList.append(gyr)

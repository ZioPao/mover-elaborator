import serial
from serial import SerialException
import tkinter as tk
import matplotlib.pyplot as plt
import re
import array


def init_mover(master: bool, mov_id: int):

    mov = None
    matched = False

    if master:
        str_chk = "MASTER"
    else:
        str_chk = "SLAVE"
    while matched is False:
        try:
            mov = serial.Serial('COM' + str(mov_id))
            ser_bytes_tmp = mov.readline()
            decoded_bytes_tmp = ser_bytes_tmp.decode()
            if re.match(str_chk, decoded_bytes_tmp):
                matched = True

        except SerialException:
            mov = None
            mov_id += 1

    if master:
        print("Connected Master -> COM" + str(mov_id))
    else:
        print("Connected Slave -> COM" + str(mov_id))
    return mov

# GUI Part

'''
window = tk.Tk()
greeting = tk.Label(text="Mover Receiver")
greeting.pack()
window.mainloop()
'''
########################################################################
# Main

master_id = 2      #has to be checked
slave_id = master_id

mover = init_mover(True, master_id)
mover_slave = init_mover(False, slave_id + 1)
mover.flushInput()

acc_list = list()
gyr_list = list()

# plots stuff

ax = plt.axes(projection='3d')
# todo can be done with lambdas

acc_x_values = list()
acc_y_values = list()
acc_z_values = list()

gyr_x_values = list()
gyr_y_values = list()
gyr_z_values = list()
while True:
    ser_bytes = mover.readline()
    decoded_bytes = ser_bytes.decode()

    # Read main data
    if re.match("main", decoded_bytes):
        ser_bytes = mover.readline()
        decoded_bytes = ser_bytes.decode()
        tmp = re.findall("(\S*),(\S*),(\S*),(\S*),(\S*),(\S*)", decoded_bytes)[0]
        acc = array.array('i', [int(tmp[0]), int(tmp[1]), int(tmp[2])])
        gyr = array.array('i', [int(tmp[3]), int(tmp[4]), int(tmp[5])])

        # for plots
        acc_x_values.append(int(tmp[0]))
        acc_y_values.append(int(tmp[1]))
        acc_z_values.append(int(tmp[2]))

        gyr_x_values.append(int(tmp[3]))
        gyr_y_values.append(int(tmp[4]))
        gyr_z_values.append(int(tmp[5]))

        acc_list.append(acc)
        gyr_list.append(gyr)


ax.scatter(acc_x_values, acc_y_values, acc_z_values, c=acc_z_values, cmap="Greens")
import serial
from serial import SerialException
import tkinter as tk
import matplotlib.pyplot as plt
import re
import array


def init_movers():

    mov_tmp = None
    id_tmp = 0
    mov_master = None
    id_master = 0
    mov_slave = None
    id_slave = 0

    while (mov_master and mov_slave) is None:

        try:
            mov_tmp = serial.Serial('COM' + str(id_tmp))
            ser_bytes_tmp = mov_tmp.readline()
            decoded_bytes_tmp = ser_bytes_tmp.decode()
            if re.match('MASTER', decoded_bytes_tmp):
                mov_master = mov_tmp
                id_master = id_tmp
            if re.match('SLAVE', decoded_bytes_tmp):
                mov_slave = mov_tmp
                id_slave = id_tmp
            id_tmp += 1     # just in case
        except SerialException:
            id_tmp += 1

    print("Connected Master: -> COM" + str(id_master))
    mov_master.flushInput()
    print("Connected Slave -> COM" + str(id_slave))
    mov_slave.flushInput()

    return mov_master, mov_slave

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
slave_id = 2

main_mover, slave_mover = init_movers()

acc_list = list()
gyr_list = list()

# plots stuff

ax = plt.axes(projection='3d')
# todo can be done with lambdas

#MAIN
m_acc_x_values = list()
m_acc_y_values = list()
m_acc_z_values = list()
m_gyr_x_values = list()
m_gyr_y_values = list()
m_gyr_z_values = list()

#SLAVE
s_acc_x_values = list()
s_acc_y_values = list()
s_acc_z_values = list()
s_gyr_x_values = list()
s_gyr_y_values = list()
s_gyr_z_values = list()
while True:
    ser_bytes = main_mover.readline()
    decoded_bytes = ser_bytes.decode()

    # Read main data
    if re.match("main", decoded_bytes):
        ser_bytes = main_mover.readline()
        decoded_bytes = ser_bytes.decode()
        tmp = re.findall("(\S*),(\S*),(\S*),(\S*),(\S*),(\S*)", decoded_bytes)[0]
        acc = array.array('i', [int(tmp[0]), int(tmp[1]), int(tmp[2])])
        gyr = array.array('i', [int(tmp[3]), int(tmp[4]), int(tmp[5])])

        # for plots
        m_acc_x_values.append(int(tmp[0]))
        m_acc_y_values.append(int(tmp[1]))
        m_acc_z_values.append(int(tmp[2]))
        m_gyr_x_values.append(int(tmp[3]))
        m_gyr_y_values.append(int(tmp[4]))
        m_gyr_z_values.append(int(tmp[5]))

        acc_list.append(acc)
        gyr_list.append(gyr)
    if re.match("slave", decoded_bytes):
        ser_bytes = main_mover.readline()
        decoded_bytes = ser_bytes.decode()
        tmp = re.findall("(\S*),(\S*),(\S*),(\S*),(\S*),(\S*)", decoded_bytes)[0]

        # for plots
        s_acc_x_values.append(int(tmp[0]))
        s_acc_y_values.append(int(tmp[1]))
        s_acc_z_values.append(int(tmp[2]))
        s_gyr_x_values.append(int(tmp[3]))
        s_gyr_y_values.append(int(tmp[4]))
        s_gyr_z_values.append(int(tmp[5]))

ax.scatter(m_gyr_x_values, m_gyr_y_values, m_gyr_z_values, c=m_gyr_z_values, cmap="Greens")
ax.scatter(s_gyr_x_values, s_gyr_y_values, s_gyr_z_values, c=s_gyr_z_values, cmap="Greens")

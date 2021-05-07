import time

import serial
from serial import SerialException
import tkinter as tk
from pynput.keyboard import KeyCode, Listener
import matplotlib.pyplot as plt
import re
import array
import pandas as pd

reset_mov = False
listener = None

id_master = 0
id_slave = 0


def init_movers():
    global id_master, id_slave

    mov_master = None
    mov_slave = None
    id_tmp = 0

    while (mov_master and mov_slave) is None:

        try:
            mov_tmp = serial.Serial('COM' + str(id_tmp))
            ser_bytes_tmp = mov_tmp.readline()
            decoded_bytes_tmp = ser_bytes_tmp.decode()
            if re.match('MASTER', decoded_bytes_tmp):
                mov_master = mov_tmp
                mov_master.write(b'c')
                id_master = id_tmp
            if re.match('SLAVE', decoded_bytes_tmp):
                mov_slave = mov_tmp
                mov_slave.write(b'c')
                id_slave = id_tmp
        except SerialException:
            pass

        id_tmp += 1
        if id_tmp > 10:
            break

    if mov_master and mov_slave:
        print("Connected Master: -> COM" + str(id_master))
        mov_master.flushInput()
        print("Connected Slave -> COM" + str(id_slave))
        mov_slave.flushInput()

        return mov_master, mov_slave
    else:
        print("Couldn't find the devices!")
        exit(0)


def reinit_movers(m_mover, s_mover):
    global id_master, id_slave

    m_mover.write(b'r')
    s_mover.write(b'r')

    time.sleep(5)
    m_mover.write(b'c')
    s_mover.write(b'c')
    print("Resetted Master -> COM" + str(id_master))
    print("Resetted Slave -> COM" + str(id_slave))
    return m_mover, s_mover

# Keyboard shortcuts

def start_listener():
    global listener
    if listener is None:
        listener = Listener(on_press=on_press, on_relase=on_release)
        listener.start()


def on_press(key):
    global reset_mov

    if key == KeyCode.from_char('r'):
        reset_mov = True
        return False


def on_release(key):
    pass

'''
window = tk.Tk()
greeting = tk.Label(text="Mover Receiver")
greeting.pack()
window.mainloop()
'''
########################################################################
# Main
print("------------MOVER MANAGER------------\n")
main_mover, slave_mover = init_movers()

acc_list = list()
gyr_list = list()

# plots stuff

ax = plt.axes(projection='3d')
# todo can be done with lambdas

# MAIN
m_acc_x_values = list()
m_acc_y_values = list()
m_acc_z_values = list()
m_gyr_x_values = list()
m_gyr_y_values = list()
m_gyr_z_values = list()

# SLAVE
s_acc_x_values = list()
s_acc_y_values = list()
s_acc_z_values = list()
s_gyr_x_values = list()
s_gyr_y_values = list()
s_gyr_z_values = list()


start_listener()
main_mover.reset_input_buffer()
print("Start")
while True:
    if reset_mov:
        print("Resetting!")
        reset_mov = False
        time.sleep(3)
        main_mover, slave_mover = reinit_movers(main_mover, slave_mover)

    ser_bytes = main_mover.readline()
    decoded_bytes = ser_bytes.decode()
    print(decoded_bytes)

    # Read main data
    if re.match("main", decoded_bytes):
        ser_bytes = main_mover.readline()
        decoded_bytes = ser_bytes.decode()
        print(decoded_bytes)

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
        print(decoded_bytes)

        tmp = re.findall("(\S*),(\S*),(\S*),(\S*),(\S*),(\S*)", decoded_bytes)[0]

        # for plots
        s_acc_x_values.append(int(tmp[0]))
        s_acc_y_values.append(int(tmp[1]))
        s_acc_z_values.append(int(tmp[2]))
        s_gyr_x_values.append(int(tmp[3]))
        s_gyr_y_values.append(int(tmp[4]))
        s_gyr_z_values.append(int(tmp[5]))

    #slave_bytes = slave_mover.readline()
    #decoded_slave_bytes = slave_bytes.decode()
    #print(decoded_slave_bytes)

######################################################
ax.scatter(m_gyr_x_values, m_gyr_y_values, m_gyr_z_values, c=m_gyr_z_values, cmap="Greens")
ax.scatter(s_gyr_x_values, s_gyr_y_values, s_gyr_z_values, c=s_gyr_z_values, cmap="Reds")


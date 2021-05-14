import time
import serial
from serial import SerialException
import tkinter as tk
from pynput.keyboard import KeyCode, Listener
import re
import pickle
from collections import Counter

from controller_module import Controller



# MAPPING
# 0 = Jogging
# 1 = Walking
# 2 = Upstairs (Maybe reusable for jumping?)
# 6 = Stopped
# // = Downstairs (Useless)
# // = Standing (Useless)
# // = Sitting (Useless)

# IDs
ID_MASTER = 0
ID_SLAVE = 0


# Miscellaneous
LISTENER = None
DATA_DIVIDER = 1000
MAX_LEN_PREDICTION_LIST = 4


def init_movers():
    global ID_MASTER, ID_SLAVE

    mov_master = None
    mov_slave = None
    id_tmp = 0

    while (mov_master and mov_slave) is None:

        try:
            mov_tmp = serial.Serial('COM' + str(id_tmp))
            ser_bytes_tmp = mov_tmp.readline()
            decoded_bytes_tmp = ser_bytes_tmp.decode()\

            # todo rewrite this
            if re.match('MASTER', decoded_bytes_tmp):
                mov_master = mov_tmp
                mov_master.write(b'c')
                ID_MASTER = id_tmp
            if re.match('SLAVE', decoded_bytes_tmp):
                mov_slave = mov_tmp
                mov_slave.write(b'c')
                ID_SLAVE = id_tmp
        except SerialException:
            pass

        id_tmp += 1
        if id_tmp > 10:
            break

    if mov_master and mov_slave:
        print("Connected Master: -> COM" + str(ID_MASTER))
        mov_master.flushInput()
        print("Connected Slave -> COM" + str(ID_SLAVE))
        mov_slave.flushInput()

        return mov_master, mov_slave
    else:
        print("Couldn't find the devices!")
        exit(0)


def reinit_movers(m_mover, s_mover):
    global ID_MASTER, ID_SLAVE

    m_mover.write(b'r')
    s_mover.write(b'r')

    time.sleep(5)
    m_mover.write(b'c')
    s_mover.write(b'c')
    print("Resetted Master -> COM" + str(ID_MASTER))
    print("Resetted Slave -> COM" + str(ID_SLAVE))
    return m_mover, s_mover


def read_decode_data(mover, search_string, acc_values_list):

    ser_bytes = mover.readline()
    decoded_bytes = ser_bytes.decode()

    while re.match(search_string, decoded_bytes) is None:
        print("Waiting for data")
        ser_bytes = mover.readline()
        decoded_bytes = ser_bytes.decode()
    else:
        ser_bytes_data_line = mover.readline()
        decoded_bytes_data_line = ser_bytes_data_line.decode()
        regex_search = re.findall("(\S*),(\S*),(\S*),(\S*),(\S*),(\S*)", decoded_bytes_data_line)[0]

        acc_values_list.append([float(regex_search[0])/DATA_DIVIDER,
                                float(regex_search[1])/DATA_DIVIDER,
                                float(regex_search[2])/DATA_DIVIDER])

    return acc_values_list


# Keyboard shortcuts


def start_listener():
    global LISTENER

    if LISTENER is None:
        LISTENER = Listener(on_press=on_press, on_relase=on_release)
        LISTENER.start()


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
########################################################################

print("------------MOVER MANAGER------------\n")
main_mover, slave_mover = init_movers()
counter = 0
start_listener()

# Mover states
reset_mov = False

# Set the controller
controller = Controller()

# Saves predicted movement from 3-4 loops to guess what movement is actually going. The most of the value wins
main_prediction_list = list()
slave_prediction_list = list()

# Loading prediction model
knn = pickle.load(open('model2.bin', 'rb'))

print("Start")
main_mover.reset_input_buffer()


try:
    while True:
        if reset_mov:
            print("Resetting!")
            reset_mov = False
            time.sleep(3)
            main_mover, slave_mover = reinit_movers(main_mover, slave_mover)

        # Resetting list with current values of movement
        acc_values = list()

        if len(main_prediction_list) > MAX_LEN_PREDICTION_LIST or len(slave_prediction_list) > MAX_LEN_PREDICTION_LIST:

            # Guess movement
            main_prediction = Counter(main_prediction_list).most_common(1)[0][0]
            slave_prediction = Counter(slave_prediction_list).most_common(1)[0][0]

            # Resets lists
            main_prediction_list = list()
            slave_prediction_list = list()

        acc_values = read_decode_data(main_mover, 'main', acc_values)
        acc_values = read_decode_data(main_mover, 'slave', acc_values)

        try:
            # Predict type of movement
            predictions = knn.predict(acc_values)

            controller.set_analog(predictions)
            main_prediction_list.append(predictions[0])
            slave_prediction_list.append(predictions[1])

            #print(predictions)
            #print(acc_values)
            #print('--------------------')
        except ValueError:
            pass

        # Using values from the prediction, we guess what movement the user is doing
        counter += 1
except SerialException:
    print("Mover disconnected!")














####################################################################################################
# TEST STUFF
####################################################################################################


np.array(_acc_x_values)
normalization_x = np.linalg.norm(m_acc_x_values)
normalized_x = m_acc_x_values/normalization_x

######################################################
ax.scatter(m_gyr_x_values, m_gyr_y_values, m_gyr_z_values, c=m_gyr_z_values, cmap="Greens")
ax.scatter(s_gyr_x_values, s_gyr_y_values, s_gyr_z_values, c=s_gyr_z_values, cmap="Reds")


# need a way to let these two "datasets" collaborate 'cause they're pretty different data wise
# if we divide the data by 20? should be enough to compare them


list_to_predict = list()
for i in range(0, len(m_acc_x_values)):
   list_to_predict.append([m_acc_x_values[i], m_acc_y_values[i], m_acc_z_values[i]])

import time
import serial
from serial import SerialException
import tkinter as tk
from pynput.keyboard import KeyCode, Listener
import re
import pickle
from collections import Counter
import threading

from controller_module import Controller


# MAPPING
# 0 = Jogging
# 1 = Walking
# 2 = Upstairs (Maybe reusable for jumping?)
# 6 = Stopped
# // = Downstairs (Useless)
# // = Standing (Useless)
# // = Sitting (Useless)



# Miscellaneous
DATA_DIVIDER = 1000
MAX_LEN_PREDICTION_LIST = 4


# Globals
reset_mov = False
should_run_thread = False




########################################################################################

class MoverReceiver:

    def __init__(self):
        # One time setup
        self.thread = None
        self.reset_mov = False
        self.should_run_thread = False

        self.id_master = 0
        self.id_slave = 0
        self.main_mover, self.slave_mover = self.init_movers()

        self.controller = Controller()        # Set the controller
        self.knn = pickle.load(open('model2.bin', 'rb'))     # Loading prediction model

        self.main_prediction_list = []
        self.slave_prediction_list = []
        self.acc_values = []

    def init_movers(self):

        mov_master_tmp = None
        mov_slave_tmp = None
        id_tmp = 0

        while (mov_master_tmp and mov_slave_tmp) is None:

            try:
                mov_tmp = serial.Serial('COM' + str(id_tmp))
                ser_bytes_tmp = mov_tmp.readline()
                decoded_bytes_tmp = ser_bytes_tmp.decode()

                    # todo rewrite this
                if re.match('MASTER', decoded_bytes_tmp):
                    mov_master_tmp = mov_tmp
                    mov_master_tmp.write(b'c')
                    self.id_master = id_tmp
                if re.match('SLAVE', decoded_bytes_tmp):
                    mov_slave_tmp = mov_tmp
                    mov_slave_tmp.write(b'c')
                    self.id_slave = id_tmp
            except SerialException:
                pass

            id_tmp += 1
            if id_tmp > 10:
                break

        if mov_master_tmp and mov_slave_tmp:
            print("Connected Master: -> COM" + str(self.id_master))
            mov_master_tmp.flushInput()
            print("Connected Slave -> COM" + str(self.id_slave))
            mov_slave_tmp.flushInput()
        else:
            print("Couldn't find the devices!")
            mov_master_tmp = None
            mov_slave_tmp = None

        return mov_master_tmp, mov_slave_tmp

    def re_init_movers(self):
        self.main_mover.write(b'r')
        self.slave_mover.write(b'r')

        time.sleep(5)
        self.main_mover.write(b'c')
        self.slave_mover.write(b'c')
        print("Resetted Master -> COM" + str(self.id_master))
        print("Resetted Slave -> COM" + str(self.id_slave))

    def initial_setup(self):

        print("------------MOVER MANAGER------------\n")

        # Saves predicted movement from 3-4 loops to guess what movement is actually going. The most of the value wins
        main_prediction_list = list()
        slave_prediction_list = list()

        print("Start")
        self.main_mover.reset_input_buffer()

    # Looping stuff

    def read_decode_data(self, search_string):

        ser_bytes = self.main_mover.readline()
        decoded_bytes = ser_bytes.decode()

        while re.match(search_string, decoded_bytes) is None:
            print("Waiting for data")
            ser_bytes = self.main_mover.readline()
            decoded_bytes = ser_bytes.decode()
        else:
            ser_bytes_data_line = self.main_mover.readline()
            decoded_bytes_data_line = ser_bytes_data_line.decode()
            regex_search = re.findall("(\S*),(\S*),(\S*),(\S*),(\S*),(\S*)", decoded_bytes_data_line)[0]

            self.acc_values.append([float(regex_search[0]) / DATA_DIVIDER,
                                    float(regex_search[1]) / DATA_DIVIDER,
                                    float(regex_search[2]) / DATA_DIVIDER])

    def loop(self):
        while self.should_run_thread:
            try:

                # check status of slave. if it's disconnected, it will throw SerialException
                self.slave_mover.read()

                if self.reset_mov:
                    print("Resetting!")
                    self.reset_mov = False
                    time.sleep(3)
                    self.re_init_movers()

                # Resetting list with current values of movement
                self.acc_values = list()

                if len(self.main_prediction_list) > MAX_LEN_PREDICTION_LIST or len(
                        self.slave_prediction_list) > MAX_LEN_PREDICTION_LIST:
                    # Guess movement
                    main_prediction = Counter(self.main_prediction_list).most_common(1)[0][0]
                    slave_prediction = Counter(self.slave_prediction_list).most_common(1)[0][0]

                    # Resets lists
                    self.main_prediction_list = list()
                    self.slave_prediction_list = list()

                self.read_decode_data('main')
                self.read_decode_data('slave')

                try:
                    # Predict type of movement
                    predictions = self.knn.predict(self.acc_values)

                    self.controller.set_analog(predictions)
                    self.main_prediction_list.append(predictions[0])
                    self.slave_prediction_list.append(predictions[1])

                    print(predictions)
                    print(self.acc_values)
                    print('--------------------')
                except ValueError:
                    pass

            except SerialException:
                print("Mover disconnected! Retrying initialization")
                self.main_mover = None
                self.slave_mover = None
                while self.main_mover is None or self.slave_mover is None:
                    self.init_movers()
                self.reset_mov = True

    # External variables

    def startup_threaded_loop(self):
        self.should_run_thread = True       # to keep it looping
        self.thread = threading.Thread(target=self.loop, args=()).start()

    def stop_currently_running_thread(self):
        print("Stopping loop...")
        self.should_run_thread = False
        self.main_mover.flushInput()        # To stop completely

    def set_reset_mov(self, var):
        self.reset_mov = var




mov = MoverReceiver()


class GUI:

    def __init__(self):

        # Setup GUI
        self.window = tk.Tk()
        self.window.geometry("400x200")
        self.window.title("Mover Receiver")

        self.main_frame = tk.Frame(self.window, width=300, height=50)
        self.main_frame.pack(side=tk.LEFT)

        self.main_label = tk.Label(self.main_frame, text="Main operations")
        self.main_label.pack(side=tk.LEFT)

        self.start_button = tk.Button(self.main_frame, text='Start', command=mov.startup_threaded_loop)
        self.start_button.pack(side=tk.LEFT)

        self.stop_button = tk.Button(self.main_frame, text='Stop', command=mov.stop_currently_running_thread)
        self.stop_button.pack(side=tk.LEFT)

        self.reset_button = tk.Button(self.main_frame, text='Reset Movers', command=lambda: mov.set_reset_mov(True))
        self.reset_button.pack(side=tk.RIGHT)

        self.window.mainloop()

    def start_background_loop(self):
        global should_run_thread
        should_run_thread = True
        self.thread.start()

    def stop_background_loop(self):
        global should_run_thread
        should_run_thread = False



gui = GUI()



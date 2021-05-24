import time
import serial
from serial import SerialException
import tkinter as tk
import re
import pickle
import threading
import sensormotion as sm

from controller_module import Controller


# MAPPING
# 0 = Jogging
# 1 = Walking
# 2 = Upstairs (Maybe reusable for jumping?)
# 3 = Stopped


# Miscellaneous
DATA_DIVIDER = 500
MAX_LEN_PREDICTION_LIST = 4

DEBUG = 0

########################################################################################


class MoverReceiver:

    def __init__(self):

        self.b, self.a = sm.signal.build_filter(frequency=10, sample_rate=100, filter_type='low',  filter_order=4)

        # One time setup
        self.thread = None
        self.reset_mov = False
        self.should_run_thread = False

        self.id_master = 0
        self.id_slave = 0

        self.main_mover = None
        self.slave_mover = None

        while self.main_mover is None or self.slave_mover is None:
            self.main_mover, self.slave_mover = self.init_movers()

        self.controller = Controller()        # Set the controller
        self.knn = pickle.load(open('trained_models/model4.bin', 'rb'))     # Loading prediction model

        self.main_prediction_list = []
        self.slave_prediction_list = []

        self.predictions = [-1, -1]
        self.acc_values = [[-1, -1, -1], [-1, -1, -1]]

        self.raw_values_x = list()
        self.raw_values_y = list()
        self.raw_values_z = list()

        self.doing_prediction = False

    def init_movers(self):

        mov_master_tmp = None
        mov_slave_tmp = None
        id_tmp = 0

        while (mov_master_tmp and mov_slave_tmp) is None:

            try:
                mov_tmp = serial.Serial('COM' + str(id_tmp), baudrate=115200, timeout=0.01)
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
        self.reset_mov = False

    def initial_setup(self):

        print("------------MOVER MANAGER------------\n")

        # Saves predicted movement from 3-4 loops to guess what movement is actually going. The most of the value wins
        main_prediction_list = list()
        slave_prediction_list = list()

        print("Start")
        self.main_mover.reset_input_buffer()

    # Looping stuff

    def read_decode_data(self):
        try:

            self.main_mover.flushInput()

            data = self.main_mover.readline()
            ser_bytes_data_line = data.decode()
            regex_search = re.findall('(\S*),(\S*),(\S*),(\S*),(\S*),(\S*),', ser_bytes_data_line[:-2])[0]

            m_raw_x = float(regex_search[0]) / DATA_DIVIDER
            m_raw_y = float(regex_search[1]) / DATA_DIVIDER
            m_raw_z = float(regex_search[2]) / DATA_DIVIDER
            s_raw_x = float(regex_search[3]) / DATA_DIVIDER
            s_raw_y = float(regex_search[4]) / DATA_DIVIDER
            s_raw_z = float(regex_search[5]) / DATA_DIVIDER

            self.acc_values.append([m_raw_x, m_raw_y, m_raw_z])
            self.acc_values.append([s_raw_x, s_raw_y, s_raw_z])

        except (ValueError, IndexError) as e:
            pass
            #print(decoded_bytes_data_line)

    def loop(self):

        # Around 0.11 seconds to make a full loop
        while self.should_run_thread:
            try:

                # check status of slave. if it's disconnected, it will throw SerialException
                self.slave_mover.read()

                if self.reset_mov:
                    print("Resetting!")
                    time.sleep(3)
                    self.re_init_movers()

                # Resets lists
                self.acc_values = list()
                self.read_decode_data()

                if len(self.acc_values) == 0:
                    continue

                try:
                    # Predict type of movement
                    self.doing_prediction = True
                    self.predictions = self.knn.predict(self.acc_values)
                    self.doing_prediction = False
                    self.controller.set_analog(self.predictions)

                    if DEBUG:
                        print(self.predictions)
                        print(self.acc_values)
                        print('--------------------')

                except (ValueError, IndexError) as e:
                    print(e)

            except SerialException:
                print("Mover disconnected! Retrying initialization")
                self.main_mover = None
                self.slave_mover = None
                while self.main_mover is None or self.slave_mover is None:
                    self.init_movers()
                self.reset_mov = True

    def startup_threaded_loop(self):

        if self.should_run_thread is False:
            self.should_run_thread = True       # to keep it looping
            self.thread = threading.Thread(target=self.loop, args=()).start()

    def stop_currently_running_thread(self):

        if self.should_run_thread:
            print("Stopping loop...")
            self.should_run_thread = False
            self.main_mover.flushInput()        # To stop completely
        else:
            print("It's not running right now!")

    def set_reset_mov(self, var):
        self.reset_mov = var

    def get_current_prediction(self):
        if self.doing_prediction:
            return None

        return self.predictions

    def get_current_acceleration(self):

        return self.acc_values


class GUI:

    def __init__(self, mover):

        # Setup GUI
        self.mover = mover

        self.window = tk.Tk()
        self.window.iconbitmap(r'favicon.ico')
        self.window.minsize(250, 250)
        self.window.maxsize(250, 250)
        self.window.title("Mover Manager")

        self.main_frame = tk.Frame(self.window, relief=tk.RAISED)
        self.main_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.start_button = tk.Button(self.main_frame, text='Start', command=self.mover.startup_threaded_loop)
        self.start_button.pack(side=tk.LEFT, padx=5, pady=5, anchor=tk.N)

        self.stop_button = tk.Button(self.main_frame, text='Stop', command=self.mover.stop_currently_running_thread)
        self.stop_button.pack(side=tk.LEFT, padx=5, pady=5, anchor=tk.N)
        self.stop_button.config(fg='black')

        self.reset_button = tk.Button(self.main_frame, text='Reset Movers', command=lambda: self.mover.set_reset_mov(True))
        self.reset_button.pack(side=tk.LEFT, padx=5, pady=5, anchor=tk.N)
        self.reset_button.config(fg='black')

        # Controller related stuff
        self.controller_frame = tk.Frame(self.window, relief=tk.RAISED)
        self.controller_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.controller_label = tk.Label(self.controller_frame, text="")
        self.controller_label.pack()

        self.label_frame_main = tk.LabelFrame(self.window, text="Acc. Values")
        self.label_frame_main.config(bg='white')
        self.label_frame_main.pack(fill="both", expand="yes")

        self.label_frame_master = tk.LabelFrame(self.label_frame_main)
        self.label_frame_master.pack(fill="both", expand="yes")

        self.infos_m_label = tk.Label(self.label_frame_master, text='Master -> ')
        self.infos_m_label.config(bg='white')
        self.infos_m_label.pack(side=tk.LEFT, anchor=tk.NW, fill=tk.BOTH)
        self.infos_m = tk.Label(self.label_frame_master, text="tmp")
        self.infos_m.config(bg='white')
        self.infos_m.pack(side=tk.LEFT, anchor=tk.NW, fill=tk.BOTH, expand='yes')

        self.label_frame_slave = tk.LabelFrame(self.label_frame_main)
        self.label_frame_slave.pack(fill="both", expand="yes")

        self.infos_s_label = tk.Label(self.label_frame_slave, text='Slave -> ')
        self.infos_s_label.config(bg='white')
        self.infos_s_label.pack(side=tk.LEFT, anchor=tk.NW, fill=tk.BOTH)
        self.infos_s = tk.Label(self.label_frame_slave, text="tmp")
        self.infos_s.config(bg='white')
        self.infos_s.pack(side=tk.LEFT, anchor=tk.NW, fill=tk.BOTH, expand='yes')

        self.prediction_frame = tk.Frame(self.window)
        self.prediction_frame.pack(side=tk.BOTTOM, anchor=tk.E)

        self.prediction_label_main = tk.Label(self.prediction_frame, text='Predictions: ')
        self.prediction_label_m = tk.Label(self.prediction_frame, text='tmp')
        self.prediction_label_s = tk.Label(self.prediction_frame, text='tmp')
        self.prediction_label_main.pack(side=tk.LEFT)
        self.prediction_label_m.config(fg='red')
        self.prediction_label_m.pack(side=tk.LEFT)
        self.prediction_label_s.config(fg='red')
        self.prediction_label_s.pack(side=tk.LEFT)

        self.update_values()
        self.window.mainloop()

    def update_values(self):

        if self.mover.should_run_thread:
            self.start_button.config(fg='red')
        else:
            self.start_button.config(fg='black')

        if self.mover.reset_mov:
            self.reset_button.config(fg='red')
        else:
            self.reset_button.config(fg='black')

        controller_values = self.mover.controller.get_y_axis()
        controller_string = 'tmp'

        if -0.2 < controller_values <= 0.1:
            controller_string = '->'
        if 0.1 < controller_values <= 0.2:
            controller_string = '-->'
        if 0.2 < controller_values <= 0.3:
            controller_string = '--->'
        if 0.3 < controller_values <= 0.4:
            controller_string = '---->'
        if 0.4 < controller_values <= 0.5:
            controller_string = '----->'
        if 0.5 < controller_values <= 0.6:
            controller_string = '------>'
        if 0.6 < controller_values <= 0.7:
            controller_string = '------->'
        if 0.7 < controller_values <= 0.8:
            controller_string = '-------->'
        if 0.8 < controller_values <= 0.9:
            controller_string = '--------->'
        if 0.9 < controller_values <= 1.5:      # includes a little bit of float error
            controller_string = '---------->'

        self.controller_label['text'] = controller_string

        try:
            acc_values = self.mover.get_current_acceleration()

            self.infos_m['text'] = str(f'{acc_values[0][0]:.2f}') + ", " + str(f'{acc_values[0][1]:.2f}') + ", " \
                                   + str(f'{acc_values[0][2]:.2f}')
            self.infos_s['text'] = str(f'{acc_values[1][0]:.2f}') + ", " + str(f'{acc_values[1][1]:.2f}') + ", " \
                                   + str(f'{acc_values[1][2]:.2f}')

        except IndexError:
            pass

        try:

            preds = self.mover.get_current_prediction()

            self.prediction_label_m['text'] = preds[0]
            self.prediction_label_s['text'] = preds[1]
        except (IndexError, TypeError):
            pass
            #print("Error during prediction printing")

        self.prediction_label_main.after(1, self.update_values)

########################################################################################
# Startup


mov = MoverReceiver()
gui = GUI(mov)






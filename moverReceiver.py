import statistics
import time
import serial
from serial import SerialException
import tkinter as tk
import re
import pickle
import threading
from controller_module import Controller
from config import *

from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
from statistics import mean


class MoverReceiver:

    def __init__(self):

        # One time setup
        self.thread = None
        self.reset_mov = False
        self.should_run_thread = False

        self.id_master = 0
        self.id_slave = 0

        self.has_connection_been_estabilished = False
        self.main_mover, self.slave_mover = self.init_movers()

        self.controller = Controller()  # Set the controller
        #self.model = pickle.load(open('trained_models/model22.bin', 'rb'))  # Loading prediction model

        self.main_prediction_list = []
        self.slave_prediction_list = []

        self.predictions = [-1, -1]
        self.acc_values = [[-1, -1, -1], [-1, -1, -1]]
        self.gyr_values = list()

        self.raw_values_x = list()
        self.raw_values_y = list()
        self.raw_values_z = list()

        self.values_prediction_test = list()
        self.values_prediction = list()

        self.doing_prediction = False
        self.detected_movement = False

    def init_movers(self):

        mov_master_tmp = None
        mov_slave_tmp = None
        id_tmp = 0

        # this was janky before and it's even worse now.

        ###### Connect only master for now
        while mov_master_tmp is None or mov_slave_tmp is None:
            try:
                mov_tmp = serial.Serial('COM' + str(id_tmp), baudrate=115200, timeout=0.01)
                ser_bytes_tmp = mov_tmp.readline()
                decoded_bytes_tmp = ser_bytes_tmp.decode()

                # todo rewrite this
                if re.match('MASTER', decoded_bytes_tmp):
                    mov_master_tmp = mov_tmp
                    # mov_master_tmp.write(b'c')
                    self.id_master = id_tmp
                if re.match('SLAVE', decoded_bytes_tmp):
                    mov_slave_tmp = mov_tmp
                    # mov_master_tmp.write(b'c')
                    self.id_slave = id_tmp


            except SerialException:
                pass
            print(id_tmp)
            id_tmp += 1
            if id_tmp > 10:
                break

        if mov_master_tmp:
            print("Connected Master -> COM" + str(self.id_master))
            print("Connected Slave -> COM" + str(self.id_slave))

            mov_master_tmp.flushInput()

            self.has_connection_been_estabilished = True
        else:
            print("Couldn't find the devices!")
            self.has_connection_been_estabilished = False
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

        def rdd_base(check=True):
            is_done = False

            while is_done is False:
                try:
                    self.main_mover.flushInput()

                    # first reading. if it's considered "moving", then let's start the window to determine the type of movement
                    acc_line = self.main_mover.readline()
                    gyr_line = self.main_mover.readline()

                    decoded_acc_line = acc_line.decode()
                    decoded_gyr_line = gyr_line.decode()

                    regex_search = re.findall('a,(\S*),(\S*),(\S*)', decoded_acc_line[:-2])[0]

                    z_offset = 8000  # todo fix
                    m_raw_x = float(regex_search[0]) / data_divider
                    m_raw_y = float(regex_search[1]) / data_divider
                    m_raw_z = float(int(regex_search[2]) - z_offset) / data_divider

                    self.acc_values.append([m_raw_x, m_raw_y, m_raw_z])

                    regex_search = re.findall('g,(\S*),(\S*),(\S*)', decoded_gyr_line[:-2])[0]
                    m_gyr_y = float(regex_search[0])
                    m_gyr_z = float(regex_search[1])

                    # self.gyr_values.append([m_gyr_y, m_gyr_z])
                    # self.gyr_values.append([s_gyr_y, s_gyr_z])

                    is_done = True
                    if check is False:
                        self.values_prediction.append([m_raw_x, m_raw_y, m_raw_z, m_gyr_y, m_gyr_z])
                        self.values_prediction_test.append([m_raw_x, m_raw_y, m_raw_z, m_gyr_y, m_gyr_z])

                        # print([m_raw_x, m_raw_y, m_raw_z, m_gyr_y, m_gyr_z, current_time])

                    else:
                        return (abs(m_raw_x) + abs(m_raw_y) + abs(m_raw_z)) / 3  # as movement test

                except Exception:
                    pass

        try:
            '''PEAK DETECTION'''
            mov_test = rdd_base()
            if mov_test > 0.75:
                # print("Detected movement!")
                self.detected_movement = True
                # start prediction stuff
                while len(self.values_prediction) < 30:
                    rdd_base(False)
            else:
                self.detected_movement = False
        except (ValueError, IndexError) as e:
            self.main_mover.flushInput()

    def filtering(self, frame):
        b = [0.00066048, 0.00132097, 0.00066048]
        a = [-1, 1.92600086, -0.92864279]
        Nb = len(b)
        final_frame = []
        for axis in frame:
            single_f_frame = np.zeros(len(axis))
            for m in range(3, len(axis)):
                single_f_frame[m] = b[0] * axis[m]
                for i in range(1, Nb):
                    single_f_frame[m] += a[i] * single_f_frame[m - i] + b[i] * axis[m - i]
            # fix the first three values
            single_f_frame = single_f_frame[3:]
            final_frame.append(tuple(single_f_frame))


        return tuple(final_frame)

    def read_tmp(self):
        is_done = False

        while is_done is False:
            try:
                self.main_mover.flushInput()
                line = self.main_mover.readline().decode()
                regex_search = re.findall('m,(\S*),(\S*),(\S*)', line[:-2])[0]
                x_m = float(regex_search[0])
                y_m = float(regex_search[1])
                z_m = float(regex_search[2]) - 8100  # not really precise but hey

                line = self.main_mover.readline().decode()
                regex_search = re.findall('s,(\S*),(\S*),(\S*)', line[:-2])[0]
                x_s = float(regex_search[0])
                y_s = float(regex_search[1])
                z_s = float(regex_search[2]) - 8100  # not really precise but hey

                line = self.main_mover.readline().decode()
                regex_search = re.findall('(\S*)', line[:-2])[0]
                sec = float(regex_search) / 1000

                return x_m, y_m, z_m, x_s, y_s, z_s, sec

            except Exception:
                self.main_mover.flushInput()

    def loop(self):

        while self.should_run_thread:
            try:
                if self.reset_mov:
                    print("Resetting!")
                    time.sleep(3)
                    self.re_init_movers()

                # Resets lists
                self.acc_values = list()
                self.gyr_values = list()

                # self.read_decode_data()
                x_m, y_m, z_m, x_s, y_s, z_s, sec = self.read_tmp()

                try:
                    if first_time == -1 or first_time is None:
                        first_time = sec
                except Exception:
                    self.main_mover.flushInput()
                    first_time = -1
                    continue

                # print("First time: " + str(first_time))
                # print("Current time: " + str(ms))

                sec = sec - first_time  # convert it
                #print("Corrected time: " + str(sec))

                if sec > -1:
                    # Read until first second to make a frame
                    try:

                        x_list_m.append(x_m)
                        y_list_m.append(y_m)
                        z_list_m.append(z_m)

                        x_list_s.append(x_s)
                        y_list_s.append(y_s)
                        z_list_s.append(z_s)

                        t_list.append(sec)
                    except UnboundLocalError:
                        x_list_m = []
                        y_list_m = []
                        z_list_m = []
                        x_list_s = []
                        y_list_s = []
                        z_list_s = []
                        t_list = []

                '''try:
                    if abs(mean(x_list_m)) < 200:
                        for single_list in [x_list_m, y_list_m, z_list_m, x_list_s, y_list_s, z_list_s]:
                           single_list.clear()  # clean everything

                        first_time = -1
                except statistics.StatisticsError:
                    pass

                    first_time = -1
'''

                #203 because of the three values going missing after filtering.
                if len(x_list_m) > 203:
                    print("new frame with " + str(len(x_list_m)) + " values")
                    # all_frames.append(current_frame)
                    current_frame_m = (x_list_m, y_list_m, z_list_m)
                    #current_frame_s = (x_list_s, y_list_s, z_list_s)
                    filtered_frame_m = self.filtering(current_frame_m)
                    #filtered_frame_s = self.filtering(current_frame_s)
                    for single_list in [x_list_m, y_list_m, z_list_m, x_list_s, y_list_s, z_list_s]:
                        single_list.clear()
                    t_list = []
                    first_time = -1  # reset frame time
                    tuple_list.append(filtered_frame_m)

                    ## PREDICTION
                    '''
                    self.doing_prediction = True
                    prediction_m = self.model.predict(np.array(filtered_frame_m).reshape(1, -1))
                    prediction_s = self.model.predict(np.array(filtered_frame_s).reshape(1, -1))
                    self.values_prediction = []
                    self.doing_prediction = False
                    self.controller.set_correct_predictions(self.predictions)
                    '''


                '''
                try:
                    # Predict type of movement
                    
                    if self.detected_movement:
                        self.doing_prediction = True
                        self.predictions = self.knn.predict(np.array(self.values_prediction).reshape(1, -1))
                        self.values_prediction = []
                        self.doing_prediction = False
                        self.controller.set_correct_predictions(self.predictions)
                        if debug_printing_receiver:
                            #print(self.acc_values)
                            test_array = np.array(self.acc_values)
                            mean_x = abs(np.mean(test_array[:, [0]]))
                            mean_y = abs(np.mean(test_array[:, [0]]))
                            mean_z = abs(np.mean(test_array[:, [0]]))
                            final_mean = (mean_x + mean_y + mean_z)/3
                            #print(final_mean)
                            if final_mean > 1.1:
                                print("R")
                            print(self.predictions)

                            print('--------------------')
                    else:
                        self.controller.reduce_speed()
                        #todo reduce speed and stop eventually

                except (ValueError, IndexError) as e:
                    print(e)
                '''
            except SerialException:
                print("Mover disconnected! Retrying initialization")
                self.main_mover = None
                self.slave_mover = None
                while self.main_mover is None or self.slave_mover is None:
                    self.init_movers()
                self.reset_mov = True

    # Thread running section
    def startup_threaded_loop(self):

        if self.should_run_thread is False:
            self.should_run_thread = True  # to keep it looping
            self.thread = threading.Thread(target=self.loop, args=()).start()

    def stop_currently_running_thread(self):

        if self.should_run_thread:
            print("Stopping loop...")
            self.should_run_thread = False
            self.main_mover.flushInput()  # To stop completely
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
        self.config_window = None

        self.window = tk.Tk()
        self.window.iconbitmap(r'favicon.ico')
        self.window.minsize(250, 250)
        self.window.maxsize(250, 250)
        self.window.title("Mover Manager")

        self.main_frame = tk.Frame(self.window, relief=tk.RAISED)
        self.main_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        if mover.has_connection_been_estabilished:
            self.start_button = tk.Button(self.main_frame, text='Start', command=self.mover.startup_threaded_loop)
            self.start_button.pack(side=tk.LEFT, padx=5, pady=5, anchor=tk.N)

            self.stop_button = tk.Button(self.main_frame, text='Stop', command=self.mover.stop_currently_running_thread)
            self.stop_button.pack(side=tk.LEFT, padx=5, pady=5, anchor=tk.N)
            self.stop_button.config(fg='black')

            self.reset_button = tk.Button(self.main_frame, text='Reset Movers',
                                          command=lambda: self.mover.set_reset_mov(True))
            self.reset_button.pack(side=tk.LEFT, padx=5, pady=5, anchor=tk.N)
            self.reset_button.config(fg='black')

            self.config_button = tk.Button(self.main_frame, text='Config', command=lambda: self.open_config_window())
            self.config_button.pack(side=tk.RIGHT, padx=5, pady=5, anchor=tk.N)
            self.config_button.config(fg='black')

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
        else:
            self.retry_button = tk.Button(self.main_frame, text='Retry connection', command=self.retry_connection)
            self.retry_button.pack(side=tk.LEFT, padx=5, pady=5, anchor=tk.N)

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
        if 0.9 < controller_values <= 1.5:  # includes a little bit of float error
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
            # print("Error during prediction printing")

        self.prediction_label_main.after(1, self.update_values)

    def retry_connection(self):
        self.mover.main_mover = self.mover.init_movers()

        if self.mover.has_connection_been_estabilished:
            self.retry_button.destroy()

            self.start_button = tk.Button(self.main_frame, text='Start', command=self.mover.startup_threaded_loop)
            self.start_button.pack(side=tk.LEFT, padx=5, pady=5, anchor=tk.N)

            self.stop_button = tk.Button(self.main_frame, text='Stop',
                                         command=self.mover.stop_currently_running_thread)
            self.stop_button.pack(side=tk.LEFT, padx=5, pady=5, anchor=tk.N)
            self.stop_button.config(fg='black')

            self.reset_button = tk.Button(self.main_frame, text='Reset Movers',
                                          command=lambda: self.mover.set_reset_mov(True))
            self.reset_button.pack(side=tk.LEFT, padx=5, pady=5, anchor=tk.N)
            self.reset_button.config(fg='black')

            self.config_button = tk.Button(self.main_frame, text='Config',
                                           command=lambda: self.open_config_window())
            self.config_button.pack(side=tk.RIGHT, padx=5, pady=5, anchor=tk.N)
            self.config_button.config(fg='black')

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

    def open_config_window(self):
        global config

        if self.config_window is None or self.config_window.winfo_exists() is 0:
            self.config_window = tk.Toplevel(self.window)
            self.config_window.iconbitmap(r'favicon.ico')
            self.config_window.minsize(250, 150)
            self.config_window.maxsize(250, 150)
            self.config_window.title("Config")

            # Data divider
            tk.Label(self.config_window, text="Divider: ").grid(row=0, sticky=tk.W)
            e1 = tk.Entry(self.config_window)
            e1.insert(tk.END, int(config['Data']['data_divider']))
            e1.grid(row=0, column=1)

            # Debug
            chk_d_r = tk.IntVar(value=int(config['Debug']['debug_printing_receiver']))
            tk.Checkbutton(self.config_window, text='Debug Receiver',
                           variable=chk_d_r, onvalue=1, offvalue=0).grid(row=2, sticky=tk.W, pady=(10, 2))

            chk_d_c = tk.IntVar(value=int(config['Debug']['debug_printing_receiver']))
            tk.Checkbutton(self.config_window, text='Debug Virtual Controller',
                           variable=chk_d_c, onvalue=1, offvalue=0).grid(row=3, sticky=tk.W)
            # should close and set ini
            tk.Button(self.config_window, text='OK',
                      command=lambda: self.save_settings(e1, chk_d_r)).grid(row=4, column=1, pady=(10, 2))

    def save_settings(self, e1, c1):

        config['Data']['data_divider'] = e1.get()
        config['Debug']['debug_printing_receiver'] = str(c1.get())

        with open('settings.ini', 'w') as configfile:  # save
            config.write(configfile)

        self.config_window.destroy()


########################################################################################
# Startup
first_time = None
x_list_m = []
y_list_m = []
z_list_m = []
x_list_s = []
y_list_s = []
z_list_s = []
t_list = []
x_filtered_list = []
x_full_list = []
y_full_list = []
z_full_list = []
t_full_list = []

current_frame = []
all_frames = []
tuple_list = []

mov = MoverReceiver()
gui = GUI(mov)

################################################################################################
for x in all_frames:
    x_full_list.append([var[0] for var in x])
    # check len, must be > 500 Hz ?

y_full_list = []
for y in all_frames:
    y_full_list.append([var[1] for var in y])

z_full_list = []
for z in all_frames:
    z_full_list.append([var[2] for var in z])

t_full_list = []
for t in all_frames:
    t_full_list.append([var[3] for var in t])

for i in range(0, 12):
    x_list = x_full_list[i]
    x_list.pop(0)

    y_list = y_full_list[i]
    y_list.pop(0)

    z_list = z_full_list[i]
    z_list.pop(0)

    fixed_z_list = []
    for z in z_list:
        fixed_z_list.append(z - 8150)

    t_list = t_full_list[i]
    t_list.pop()

    plt.rcParams["figure.figsize"] = 10, 5
    plt.rcParams["font.size"] = 16
    plt.rcParams.update({"text.usetex": True, "font.family": "sans-serif", "font.sans-serif": ["Helvetica"]})
    plt.figure()
    plt.plot(t_list, x_list)
    plt.plot(t_list, y_list)
    plt.plot(t_list, fixed_z_list)
    plt.title(i)
    plt.xlabel("time")
    plt.ylabel("axis")
    plt.xlim([min(t_list), max(t_list)])
    plt.show()

plt.figure()
plt.plot(t_list, x_filtered_list)
plt.xlabel("time")
plt.ylabel("y")
plt.show()

y_fourier = np.fft.fft(x_list)
y_filtered_fourier = np.fft.fft(x_filtered_list)
cycles_fourier = np.fft.fftfreq(len(t_list), d=1.0 / 770)  # the frequencies in cycles/s

plt.figure()
plt.plot(cycles_fourier, np.absolute(y_fourier))
plt.plot(cycles_fourier, np.absolute(y_filtered_fourier))
plt.xlim([-100, 100])
plt.xlabel("$\omega$ (cycles/s)")
plt.ylabel("$|\hat{y}|$")
plt.show()
# checks if it's actually moving. if it's moving, we have to check the window of around 1 second...

w0 = 2 * np.pi * 5  # pole frequency (rad/s)
num = w0  # transfer function numerator coefficients
den = [1, w0]  # transfer function denominator coefficients
dt = 1.0 / 772
discreteLowPass = signal.cont2discrete((num, den), dt, method='gbt', alpha=0.5)

# The coefficients from the discrete form of the filter transfer function (but with a negative sign)
b = discreteLowPass[0][0]
a = [-x for x in discreteLowPass]
print("Filter coefficients b_i: " + str(b))
print("Filter coefficients a_i: " + str(a[1:]))

# Filter the signal
yfilt = np.zeros(len(x_list))
for i in range(3, len(x_list)):
    yfilt[i] = a[1] * yfilt[i - 1] + b[0] * x_list[i] + b[1] * x_list[i - 1]

plt.figure()
plt.plot(t_list, x_list)
plt.plot(t_list, yfilt)
plt.ylabel("$y(t)$")
plt.show()

yfilthat = np.fft.fft(yfilt)
fcycles = np.fft.fftfreq(len(t_list), d=1.0 / 772)

plt.figure()
plt.plot(fcycles, np.absolute(y_fourier))
plt.plot(fcycles, np.absolute(yfilthat))
plt.xlim([-100, 100])
plt.xlabel("$\omega$ (cycles/s)")
plt.ylabel("$|\hat{y}|$")

#### BUTTERWORTH LOW PASS

samplingFreq = 600  # around this
signalFreq = [2, 100]

# Butterworth filter
wc = 2 * np.pi * 5  # cutoff frequency (rad/s)
n = 2  # Filter order

# Compute the Butterworth filter coefficents
a = np.zeros(n + 1)
gamma = np.pi / (2.0 * n)
a[0] = 1  # first coef is always 1
for k in range(0, n):
    rfac = np.cos(k * gamma) / np.sin((k + 1) * gamma)
    a[k + 1] = rfac * a[k]  # Other coefficients by recursion

print("Butterworth polynomial coefficients a_i:                " + str(a))

# Adjust the cutoff frequency
c = np.zeros(n + 1)
for k in range(0, n + 1):
    c[n - k] = a[k] / pow(wc, k)

print("Butterworth coefficients with frequency adjustment c_i: " + str(c))
# Low-pass filter
w0 = 2 * np.pi * 5  # pole frequency (rad/s)
num = [1]  # transfer function numerator coefficients
den = c  # transfer function denominator coefficients
lowPass = signal.TransferFunction(num, den)  # Transfer function

# Generate the bode plot
w = np.logspace(np.log10(min(signalFreq) * 2 * np.pi / 10), np.log10(max(signalFreq) * 2 * np.pi * 10), 500)
w, mag, phase = signal.bode(lowPass, w)

# Magnitude plot
plt.figure()
plt.semilogx(w, mag)
for sf in signalFreq:
    plt.semilogx([sf * 2 * np.pi, sf * 2 * np.pi], [min(mag), max(mag)], 'k:')
plt.ylabel("Magnitude ($dB$)")
plt.xlim([min(w), max(w)])
plt.ylim([min(mag), max(mag)])

# Phase plot
plt.figure()
plt.semilogx(w, phase)  # Bode phase plot
plt.ylabel("Phase ($^\circ$)")
plt.xlabel("$\omega$ (rad/s)")
plt.xlim([min(w), max(w)])
plt.show()

# Compute the discrete low pass with delta_t = 1/samplingFrequency
dt = 1.0 / samplingFreq
discreteLowPass = lowPass.to_discrete(dt, method='gbt', alpha=0.5)
print(discreteLowPass)

# The coefficients from the discrete form of the filter transfer function (but with a negative sign)
b = discreteLowPass.num
a = -discreteLowPass.den
print("Filter coefficients b_i: " + str(b))
print("Filter coefficients a_i: " + str(a[1:]))

# Filter the signal
Nb = len(b)
xfilt = np.zeros(len(x_full_list))
for m in range(3, len(x_full_list)):
    xfilt[m] = b[0] * x_full_list[m]
    for i in range(1, Nb):
        xfilt[m] += a[i] * xfilt[m - i] + b[i] * x_full_list[m - i]
# View the result
# Plot the signal
plt.figure()
plt.plot(t_full_list, x_full_list)
plt.plot(t_full_list, xfilt)
plt.ylabel("$y(t)$")
plt.xlim([min(t_full_list), max(t_full_list)])
plt.show()

# Generate Fourier transform
yfilthat = np.fft.fft(yfilt)
fcycles = np.fft.fftfreq(len(t), d=1.0 / samplingFreq)

plt.figure()
plt.plot(fcycles, np.absolute(yhat));
plt.plot(fcycles, np.absolute(yfilthat));
plt.xlim([-100, 100]);
plt.xlabel("$\omega$ (cycles/s)");
plt.ylabel("$|\hat{y}|$");

s = time.time()
filtering(frame)
e = time.time()
print(str(e - s))

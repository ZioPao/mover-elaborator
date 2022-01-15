import time
import serial
import serial.tools.list_ports
from pynput.keyboard import Key
from serial import SerialException
import tkinter as tk
import re
import pickle
import configparser
import asyncio
from controller_module import Controller
import numpy as np
import threading
from pynput import keyboard




# Config setup

config = configparser.ConfigParser()
config.read('settings.ini')






class MoverReceiver:

    def __init__(self):
        print("------------MOVER MANAGER------------\n")

        # One time setup
        self.thread = None
        self.should_run_thread = False
        self.has_connection_been_estabilished = False
        self.left_mov, self.right_mov = self.init_movers()
        self.controller = Controller()  # Setup the controller

        self.model = pickle.load(open('trained_models/mod9.bin', 'rb'))
        self.prediction = -1
        self.x_list_l = []
        self.y_list_l = []
        self.z_list_l = []
        self.x_list_r = []
        self.y_list_r = []
        self.z_list_r = []
        self.t_list = []


        # to share data
        self.best_pred = -1
        self.current_data = [[0,0,0], [0,0,0]]


    def init_movers(self):

        left_mov_tmp = None
        right_mov_tmp = None

        right_id = '8&29C54EA8&0&2'  # right
        left_id = '8&29C54EA8&0&1'  # left

        ports = serial.tools.list_ports.comports()
        for p in ports:

            if p.serial_number == right_id:
                right_mov_tmp = serial.Serial(p.device, baudrate=38400, timeout=0.01)
            if p.serial_number == left_id:
                left_mov_tmp = serial.Serial(p.device, baudrate=38400, timeout=0.01)

        if right_mov_tmp is not None and left_mov_tmp is not None:
            print("Connected L -> " + str(left_mov_tmp.name))
            print("Connected R -> " + str(right_mov_tmp.name))

            left_mov_tmp.flushInput()
            right_mov_tmp.flushInput()

            self.has_connection_been_estabilished = True
        else:
            print("Couldn't find the devices!")
            self.has_connection_been_estabilished = True
            left_mov_tmp = 1
            right_mov_tmp = 1

        return left_mov_tmp, right_mov_tmp

    def on_press(self, key):

        if key == Key.X:
            self.controller.controller.set_value('BtnA', 1)
            time.sleep(1)
            self.controller.controller.set_value('BtnA', 0)
            print("Trying to focus on the application...")

    # Main operations
    def read_data(self):
        is_done = False
        while is_done is False:
            try:

                self.left_mov.flushInput()
                self.right_mov.flushInput()
                (x_l, y_l, z_l, t_l) = self.read_single_mov('l')
                (x_r, y_r, z_r, t_r) = self.read_single_mov('r')

                return x_l, y_l, z_l, t_l, x_r, y_r, z_r, t_r

            except RuntimeError:
                print("Runtime error in read_data")
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

    def read_single_mov(self, mov_id):
        try:
            line = '\r\n'

            if mov_id == 'l':
                while line == '\r\n':
                    line = self.left_mov.readline().decode()
                regex_search = re.findall('l,(\S*),(\S*),(\S*),(\S*)', line[:-2])[0]
            else:
                while line == '\r\n':
                    line = self.right_mov.readline().decode()
                regex_search = re.findall('r,(\S*),(\S*),(\S*),(\S*)', line[:-2])[0]

            x = float(regex_search[0])
            y = float(regex_search[1])
            z = float(regex_search[2])
            t = float(regex_search[3]) / 1000
        except IndexError:
            #print(e)
            #print("CANNOT READ DATA!!!!!")
            x = 0
            y = 0
            z = 0
            t = 0

        return x, y, z, t

    def loop(self):

        listener = keyboard.Listener(on_press=self.on_press)
        listener.start()  # start to listen on a separate thread
        #listener.join()  # remove if main thread is polling self.keys
        print("Starting loop")

        while self.should_run_thread:

            try:
                x_l, y_l, z_l, t_l, x_r, y_r, z_r, t_r = self.read_data()
                zero_check = np.array([x_l, y_l, z_l, x_r, y_r, z_r])

                # check if every value is not 0
                if np.all((zero_check != 0.)):

                    self.current_data = [[x_l,y_l,z_l], [x_r,y_r,z_r]]

                    # Setup time
                    try:
                        if first_time == -1 or first_time is None:
                            first_time = t_l
                    except Exception:
                        first_time = -1
                        continue
                    sec = t_l - first_time  # convert it

                    if sec > -1:
                        # Read until first second to make a frame
                        self.x_list_l.append(x_l)
                        self.y_list_l.append(y_l)
                        self.z_list_l.append(z_l)

                        self.x_list_r.append(x_r)
                        self.y_list_r.append(y_r)
                        self.z_list_r.append(z_r)

                        self.t_list.append(sec)

                    # sample size of 100 elements... 50 per sensor?
                    if len(self.x_list_l) > 50 or len(self.x_list_r) > 50:

                        frame = (self.x_list_l, self.y_list_l, self.z_list_l, self.x_list_r,
                                 self.y_list_r, self.z_list_r)
                        ##############################################

                        if debug_learning:
                            #print("Right: " + str(x_r) + ", " + str(y_r) + ", " + str(z_r))
                            #print("Left: " + str(x_l) + ", " + str(y_l) + ", " + str(z_l))
                            all_frames.append(frame)        # should work?
                            time.sleep(0.1)     # todo let's assume that this is 100 ms, prediction
                            print(len(all_frames))
                            if len(all_frames) > 210:
                                print("STOP")
                            ############################
                        else:
                            self.prediction = self.model.predict_proba(np.array(frame).reshape(1, -1))

                        self.x_list_l = []
                        self.y_list_l = []
                        self.z_list_l = []
                        self.x_list_r = []
                        self.y_list_r = []
                        self.z_list_r = []
                        self.t_list = []
                        first_time = -1  # reset frame time

                        best_pred_probability = np.amax(self.prediction)

                        if best_pred_probability > float(config['Data']['best_pred_probability']):        #Customizable
                            self.best_pred = np.where(self.prediction[0] == best_pred_probability)[0][0]
                            self.controller.manage_prediction(self.best_pred)

                            #self.pred_list.append(best_pred)
                        else:
                            self.controller.decrease_speed()
                else:
                    self.current_data = [[0, 0, 0], [0, 0, 0]]
                    self.controller.decrease_speed()
            except TypeError as e:
                print("Entering type error")
                # CLEANING
                for single_list in [self.x_list_l, self.y_list_l, self.z_list_l, self.x_list_r, self.y_list_r,
                                    self.z_list_r]:
                    single_list.clear()
                self.t_list = []
                first_time = -1  # reset frame time
            except SerialException:
                print("Reconnect devices")
                exit(1)

    # Thread running section
    def startup_threaded_loop(self):

        if self.should_run_thread is False:
            self.should_run_thread = True  # to keep it looping
            self.thread = threading.Thread(target=self.loop, args=()).start()

    def stop_currently_running_thread(self):

        if self.should_run_thread:
            print("Stopping loop...")
            self.should_run_thread = False
        else:
            print("It's not running right now!")

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
            self.buttons_frame = tk.LabelFrame(self.main_frame)
            self.buttons_frame.pack(fill=tk.X, expand=True)
            self.start_button = tk.Button(self.buttons_frame, text='Start', command=self.mover.startup_threaded_loop)
            self.start_button.pack(side=tk.LEFT, padx=5, pady=5, anchor=tk.N)

            self.stop_button = tk.Button(self.buttons_frame, text='Stop', command=self.mover.stop_currently_running_thread)
            self.stop_button.pack(side=tk.LEFT, padx=5, pady=5, anchor=tk.N)
            self.stop_button.config(fg='black')

            self.config_button = tk.Button(self.buttons_frame, text='Config', command=lambda: self.open_config_window())
            self.config_button.pack(side=tk.RIGHT, padx=5, pady=5, anchor=tk.N)
            self.config_button.config(fg='black')


            self.debug_frame = tk.LabelFrame(self.main_frame)
            self.debug_frame.pack(expand=True, fill=tk.BOTH)
            self.debug_left_label = tk.Label(self.debug_frame)
            self.debug_left_label.pack(side=tk.LEFT, padx=60)
            self.debug_right_label = tk.Label(self.debug_frame)
            self.debug_right_label.pack(side=tk.LEFT)

            # Controller related stuff
            self.controller_frame = tk.Frame(self.window, relief=tk.RAISED)
            self.controller_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

            self.label_frame_main = tk.LabelFrame(self.window, text="Controller Values")
            self.label_frame_main.config(bg='white')
            self.label_frame_main.pack(fill="both", expand="yes")

            self.label_frame_y = tk.LabelFrame(self.label_frame_main)
            self.label_frame_y.pack(fill="both", expand="yes")

            self.y_axis_label = tk.Label(self.label_frame_y, text='Y -> ')
            self.y_axis_label.config(bg='white')
            self.y_axis_label.pack(side=tk.LEFT, anchor=tk.NW, fill=tk.BOTH)
            self.y_axis_info = tk.Label(self.label_frame_y, text="tmp")
            self.y_axis_info.config(bg='white')
            self.y_axis_info.pack(side=tk.LEFT, anchor=tk.NW, fill=tk.BOTH, expand='yes')

            self.label_frame_x = tk.LabelFrame(self.label_frame_main)
            self.label_frame_x.pack(fill="both", expand="yes")

            self.x_axis_label = tk.Label(self.label_frame_x, text='X -> ')
            self.x_axis_label.config(bg='white')
            self.x_axis_label.pack(side=tk.LEFT, anchor=tk.NW, fill=tk.BOTH)
            self.x_axis_info = tk.Label(self.label_frame_x, text="tmp")
            self.x_axis_info.config(bg='white')
            self.x_axis_info.pack(side=tk.LEFT, anchor=tk.NW, fill=tk.BOTH, expand='yes')

            self.prediction_frame = tk.Frame(self.window)
            self.prediction_frame.pack(side=tk.BOTTOM, anchor=tk.E)

            self.prediction_label_main = tk.Label(self.prediction_frame, text='Predictions: ')
            self.prediction_label = tk.Label(self.prediction_frame, text='tmp')
            self.prediction_label_main.pack(side=tk.LEFT)
            self.prediction_label.config(fg='red')
            self.prediction_label.pack(side=tk.LEFT)

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

        self.y_axis_info['text'] = self.mover.controller.analog_values_y['current']
        self.x_axis_info['text'] = self.mover.controller.analog_values_x['current']
        self.debug_left_label['text'] = self.mover.current_data[0]
        self.debug_right_label['text'] = self.mover.current_data[1]

        self.prediction_label['text'] = self.mover.best_pred

        self.prediction_label_main.after(1, self.update_values)

    def retry_connection(self):
        self.mover.left_mov, self.mover.right_mov = self.mover.init_movers()

        if self.mover.has_connection_been_estabilished:
            self.retry_button.destroy()
            self.start_button = tk.Button(self.main_frame, text='Start', command=self.mover.startup_threaded_loop)
            self.start_button.pack(side=tk.LEFT, padx=5, pady=5, anchor=tk.N)

            self.stop_button = tk.Button(self.main_frame, text='Stop', command=self.mover.stop_currently_running_thread)
            self.stop_button.pack(side=tk.LEFT, padx=5, pady=5, anchor=tk.N)
            self.stop_button.config(fg='black')


            # Controller related stuff
            self.controller_frame = tk.Frame(self.window, relief=tk.RAISED)
            self.controller_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

            self.label_frame_main = tk.LabelFrame(self.window, text="Controller")
            self.label_frame_main.config(bg='white')
            self.label_frame_main.pack(fill="both", expand="yes")

            self.label_frame_y = tk.LabelFrame(self.label_frame_main)
            self.label_frame_y.pack(fill="both", expand="yes")

            self.y_axis_label = tk.Label(self.label_frame_y, text='Y -> ')
            self.y_axis_label.config(bg='white')
            self.y_axis_label.pack(side=tk.LEFT, anchor=tk.NW, fill=tk.BOTH)
            self.y_axis_info = tk.Label(self.label_frame_y, text="tmp")
            self.y_axis_info.config(bg='white')
            self.y_axis_info.pack(side=tk.LEFT, anchor=tk.NW, fill=tk.BOTH, expand='yes')

            self.label_frame_x = tk.LabelFrame(self.label_frame_main)
            self.label_frame_x.pack(fill="both", expand="yes")

            self.x_axis_label = tk.Label(self.label_frame_x, text='X -> ')
            self.x_axis_label.config(bg='white')
            self.x_axis_label.pack(side=tk.LEFT, anchor=tk.NW, fill=tk.BOTH)
            self.x_axis_info = tk.Label(self.label_frame_x, text="tmp")
            self.x_axis_info.config(bg='white')
            self.x_axis_info.pack(side=tk.LEFT, anchor=tk.NW, fill=tk.BOTH, expand='yes')

            self.prediction_frame = tk.Frame(self.window)
            self.prediction_frame.pack(side=tk.BOTTOM, anchor=tk.E)

            self.prediction_label_main = tk.Label(self.prediction_frame, text='Predictions: ')
            self.prediction_label = tk.Label(self.prediction_frame, text='tmp')
            self.prediction_label_main.pack(side=tk.LEFT)
            self.prediction_label.config(fg='red')
            self.prediction_label.pack(side=tk.LEFT)

            self.update_values()

    def open_config_window(self):

        if self.config_window is None or self.config_window.winfo_exists() == 0:
            self.config_window = tk.Toplevel(self.window)
            self.config_window.iconbitmap(r'favicon.ico')
            self.config_window.minsize(250, 150)
            self.config_window.maxsize(250, 150)
            self.config_window.title("Config")

            # best_pred_probability
            tk.Label(self.config_window, text="Prediction probability: ").grid(pady=10, row=0, sticky=tk.W)
            e1 = tk.Entry(self.config_window)
            e1.insert(tk.END, float(config['Data']['best_pred_probability']))
            e1.grid(row=0, column=1)

            # Debug
            #chk_d_r = tk.IntVar(value=int(config['Debug']['debug_printing_receiver']))
            #tk.Checkbutton(self.config_window, text='Debug Receiver',
            #               variable=chk_d_r, onvalue=1, offvalue=0).grid(row=2, sticky=tk.W, pady=(10, 2))

            #chk_d_c = tk.IntVar(value=int(config['Debug']['debug_printing_receiver']))
            #tk.Checkbutton(self.config_window, text='Debug Virtual Controller',
            #               variable=chk_d_c, onvalue=1, offvalue=0).grid(row=3, sticky=tk.W)
            # should close and set ini
            tk.Button(self.config_window, text='OK',
                      command=lambda: self.save_settings(e1)).grid(row=4, column=1, pady=(10, 2))

    def save_settings(self, e1):
        global config
        config['Data']['best_pred_probability'] = e1.get()

        with open('settings.ini', 'w') as configfile:
            config.write(configfile)

        # Reload
        config.read('settings.ini')
        self.config_window.destroy()


########################################################################################
# Startup

all_frames = []
debug_learning = False
mov = MoverReceiver()
gui = GUI(mov)

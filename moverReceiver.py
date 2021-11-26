import statistics
import serial
import serial.tools.list_ports
from serial import SerialException
import tkinter as tk
import re
import pickle
from config import *
import asyncio
from controller_module import Controller
import numpy as np
from filtering import filtering_pass


class MoverReceiver:

    def __init__(self):

        # One time setup
        self.thread = None
        self.reset_mov = False
        self.should_run_thread = False

        self.id_left = 0
        self.id_right = 0

        self.has_connection_been_estabilished = False
        self.left_mov, self.right_mov = self.init_movers()

        self.controller = Controller()  # Set the controller

        # Loading prediction models
        self.model = pickle.load(open('trained_models/mod.bin', 'rb'))
        #self.model_right = pickle.load(open('trained_models/left/l_mod1.bin', 'rb'))

        self.current_x_l = 0
        self.current_y_l = 0
        self.current_z_l = 0
        self.current_t_l = 0

        self.current_x_r = 0
        self.current_y_r = 0
        self.current_z_r = 0
        self.current_t_r = 0


        self.values_prediction_test = list()
        self.values_prediction = list()

        self.x_list_l = []
        self.y_list_l = []
        self.z_list_l = []
        self.x_list_r = []
        self.y_list_r = []
        self.z_list_r = []
        self.t_list = []

        self.prediction_l = -1
        self.prediction_r = -1


        self.x_plot_list = []
        self.y_plot_list = []

        self.doing_prediction = False
        self.detected_movement = False

    def init_movers(self):

        left_mov_tmp = None
        right_mov_tmp = None

        right_id ='8&29C54EA8&0&1'        # right
        left_id = '8&29C54EA8&0&2'        # left

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
            self.has_connection_been_estabilished = False
            left_mov_tmp = None
            right_mov_tmp = None

        return left_mov_tmp, right_mov_tmp

    def re_init_movers(self):
        self.left_mov.write(b'r')
        self.right_mov.write(b'r')

        time.sleep(5)
        self.left_mov.write(b'c')
        self.right_mov.write(b'c')
        print("Resetted L -> COM" + str(self.id_left))
        print("Resetted R -> COM" + str(self.id_right))
        self.reset_mov = False

    def initial_setup(self):

        print("------------MOVER MANAGER------------\n")

        print("Start")
        self.left_mov.reset_input_buffer()
        self.right_mov.reset_input_buffer()

        # Start reading threads

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
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            except Exception as e:
                #print(e)
                pass

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
            z = float(regex_search[2])  # not really precise but hey
            t = float(regex_search[3]) / 1000
        except Exception as e:
            #print(e)
            x = 0
            y = 0
            z = 0
            t = 0

        return x, y, z, t

    def loop(self):

        while self.should_run_thread:

            try:
                if self.reset_mov:
                    print("Resetting!")
                    time.sleep(3)
                    self.re_init_movers()

                x_l, y_l, z_l, t_l, x_r, y_r, z_r, t_r = self.read_data()
                print("Right: " + str(x_r) + ", " + str(y_r) + ", " + str(z_r))
                print("Left: " + str(x_l) + ", " + str(y_l) + ", " + str(z_l))


                zero_check = np.array([x_l, y_l, z_l, x_r, y_r, z_r])
                is_valid = np.all((zero_check != 0.))

                if is_valid:
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

                    # sample size of 50 elements... 25 per sensor?
                    if len(self.x_list_l) > 25 and len(self.x_list_r) > 25:
                        print("Prediction")
                        frame = (self.x_list_l, self.y_list_l, self.z_list_l, self.x_list_r, self.y_list_r, self.z_list_r)


                        ##############################################
                        all_frames.append(frame)        # should work?
                        if len(all_frames) > 50:
                            print("STOP")
                        ############################

                        # PREDICTION
                        self.doing_prediction = True
                        #self.prediction = self.model.predict(np.array(frame).reshape(1, -1))
                        #print(self.prediction)
                        time.sleep(0.1)     # todo let's assume that this is 100 ms
                        self.doing_prediction = False


                        self.x_list_l = []
                        self.y_list_l = []
                        self.z_list_l = []
                        self.x_list_r = []
                        self.y_list_r = []
                        self.z_list_r = []


                        self.t_list = []
                        first_time = -1  # reset frame time

            except TypeError:
                # CLEANING
                for single_list in [self.x_list_l, self.y_list_l, self.z_list_l, self.x_list_r, self.y_list_r,
                                    self.z_list_r]:
                    single_list.clear()
                self.t_list = []
                first_time = -1  # reset frame time
                # tuple_list.append(filtered_frame_r)
            except SerialException:
                print("Mover disconnected! Retrying initialization")
                self.left_mov = None
                self.right_mov = None
                while self.left_mov is None or self.right_mov is None:
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
            self.left_mov.flushInput()  # To stop completely
            self.right_mov.flushInput()  # To stop completely

        else:
            print("It's not running right now!")

    def set_reset_mov(self, var):
        self.reset_mov = var


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
            self.prediction_label_l = tk.Label(self.prediction_frame, text='tmp')
            self.prediction_label_r = tk.Label(self.prediction_frame, text='tmp')
            self.prediction_label_main.pack(side=tk.LEFT)
            self.prediction_label_l.config(fg='red')
            self.prediction_label_l.pack(side=tk.LEFT)
            self.prediction_label_r.config(fg='red')
            self.prediction_label_r.pack(side=tk.LEFT)

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



        controller_values = self.mover.controller.analog_values_y['current']
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

        '''
        try:
            acc_values = self.mover.get_current_acceleration()

            self.infos_m['text'] = str(f'{acc_values[0][0]:.2f}') + ", " + str(f'{acc_values[0][1]:.2f}') + ", " \
                                   + str(f'{acc_values[0][2]:.2f}')
            self.infos_s['text'] = str(f'{acc_values[1][0]:.2f}') + ", " + str(f'{acc_values[1][1]:.2f}') + ", " \
                                   + str(f'{acc_values[1][2]:.2f}')

        except IndexError:
            pass
        '''

        try:
            pass
            if self.mover.prediction_l[0] != -1:
                self.prediction_label_l['text'] = str(self.mover.prediction_l[0])
            if self.mover.prediction_r[0] != -1:
                self.prediction_label_r['text'] = str(self.mover.prediction_r[0])
        except (IndexError, TypeError):
            pass
            #print("Error during prediction printing")

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

current_frame = []
all_frames = []
tuple_list = []

mov = MoverReceiver()
gui = GUI(mov)

#import matplotlib.pyplot as plt
#plt.plot()
#plt.scatter(mov.x_plot_list, mov.y_plot_list)
#plt.show()

'''
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
print(str(e - s))'''


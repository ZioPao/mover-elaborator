import pyxinput
from config import *


LEFT_AXIS_X = 'AxisLx'
LEFT_AXIS_Y = 'AxisLy'

OLD_ANALOG_KEY = 'old'
CURR_ANALOG_KEY = 'current'


NORMAL_MOV = 0.
RUN_MOV = 1.
LEFT_MOV = 2.
RIGHT_MOV = 3.

LEFT = 0
RIGHT = 1

class Controller:
    # 1) Get predictions from main
    # 2) Decide which to chose
    # 3) Set analog value for x and y depending on current value
    # 4) Backup current value for x and y

    def __init__(self):
        # Setup virtual controller
        self.controller = pyxinput.vController()
        self.analog_values_y = {'old': 0., 'current': 0.}
        self.analog_values_x = {'old': 0., 'current': 0.}

        self.prev_prediction = None

    def get_correct_y(self, array, increment, top_value):
        if array[OLD_ANALOG_KEY] <= top_value:
            new_y_value = (array[OLD_ANALOG_KEY] + increment)/2     # smooth it out
        else:
            new_y_value = array[OLD_ANALOG_KEY]     # current

        return new_y_value

    def get_correct_x(self, array, increment, top_value, direction):

        # negative values
        if direction == LEFT:
            if array[OLD_ANALOG_KEY] >= top_value:
                x = (array[OLD_ANALOG_KEY] - increment) / 2  # smooth it out
            else:
                x = array[OLD_ANALOG_KEY]  # current
        else:

            if array[OLD_ANALOG_KEY] <= top_value:
                x = (array[OLD_ANALOG_KEY] + increment) / 2  # smooth it out
            else:
                x = array[OLD_ANALOG_KEY]  # current

        return x
    def manage_prediction(self, pred):

        # startup
        y_value = self.analog_values_y[OLD_ANALOG_KEY]
        x_value = self.analog_values_x[OLD_ANALOG_KEY]

        self.prev_prediction = pred

        # WALK
        if pred == 0.:
            y_value = self.get_correct_y(self.analog_values_y,
                                                 walking_increment, walking_top)
        # RUN
        if pred == 1.:
            y_value = self.get_correct_y(self.analog_values_y,
                                                 jogging_increment, jogging_top)

        # SIDE LEFT
        if pred == 2.:
            x_value = self.get_correct_x(self.analog_values_x, side_increment, -side_top, LEFT)

        # SIDE RIGHT
        if pred == 3.:
            x_value = self.get_correct_x(self.analog_values_x, side_increment, side_top, RIGHT)


        # back them up
        self.analog_values_y[OLD_ANALOG_KEY] = self.analog_values_y[CURR_ANALOG_KEY]
        self.analog_values_y[CURR_ANALOG_KEY] = y_value
        self.analog_values_x[OLD_ANALOG_KEY] = self.analog_values_x[CURR_ANALOG_KEY]
        self.analog_values_x[CURR_ANALOG_KEY] = x_value

        # Applies current values
        self.controller.set_value(LEFT_AXIS_Y, y_value)
        self.controller.set_value(LEFT_AXIS_X, x_value)

    def decrease_speed(self):
        # STOPPING
        y_value = self.analog_values_y[CURR_ANALOG_KEY]
        x_value = self.analog_values_x[CURR_ANALOG_KEY]


        y_value = y_value - 0.01


        # to stop it from jittering
        if x_value == 0 or abs(x_value) < 0.005:
            x_value = 0
        else:
            if x_value < 0:
                x_value = x_value + 0.01
            else:
                x_value = x_value - 0.01

        if y_value < 0:
            y_value = 0

        # back them up
        self.analog_values_y[OLD_ANALOG_KEY] = self.analog_values_y[CURR_ANALOG_KEY]
        self.analog_values_y[CURR_ANALOG_KEY] = y_value
        self.analog_values_x[OLD_ANALOG_KEY] = self.analog_values_x[CURR_ANALOG_KEY]
        self.analog_values_x[CURR_ANALOG_KEY] = x_value

        # Applies current values
        self.controller.set_value(LEFT_AXIS_Y, y_value)
        self.controller.set_value(LEFT_AXIS_X, x_value)


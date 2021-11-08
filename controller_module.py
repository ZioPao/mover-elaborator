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

class Controller:
    # 1) Get predictions from main
    # 2) Decide which to chose
    # 3) Set analog value for x and y depending on current value
    # 4) Backup current value for x and y

    def __init__(self):
        # Setup virtual controller
        self.controller = pyxinput.vController()
        self.analog_values_y = {'old': 0, 'current': 0}
        self.analog_values_x = {'old': 0, 'current': 0}

        self.prev_prediction = None

    def get_new_analog_values(self, array, increment, decrease, top_value):
        if array[OLD_ANALOG_KEY] < top_value:
            new_y_value = array[OLD_ANALOG_KEY] + increment
        else:
            new_y_value = array[OLD_ANALOG_KEY] - decrease

        return new_y_value


    def manage_predictions(self, pred_left, pred_right):
        y_value = 0
        x_value = 0

        # WALK MOVEMENT
        if pred_left == 0. and pred_right == 0.:
            y_value = 0 if -0.1 < self.analog_values_y[OLD_ANALOG_KEY] < stopped_range else \
                self.analog_values_y[OLD_ANALOG_KEY] - stopped_decrement

        # RUN MOVEMENT
        # or because we only need one leg to sprint or something
        if pred_left == 1. or pred_right == 1.:
            y_value = self.get_new_analog_values(self.analog_values_y,
                                                 walking_increment, walking_decrement, walking_top)

        # SIDE MOVEMENT
        if pred_left == 2. or pred_right == 2.:
            if self.analog_values_x[OLD_ANALOG_KEY] != 0:
                x_value = self.get_new_analog_values(self.analog_values_x,
                                                     side_increment, side_decrement, side_top)

                if pred_left == 2.:
                    x_value = -x_value      # left
            else:
                x_value = -0.1 if pred_left == 2. else 0.1      # small steps?

        # Finally applies the y_value to the controller
        y_value = y_value if y_value > 0. else 0.
        y_value = y_value if y_value < 1. else 1.

        self.controller.set_value(LEFT_AXIS_Y, y_value)
        self.controller.set_value(LEFT_AXIS_X, x_value)

        # back them up
        self.analog_values_y[OLD_ANALOG_KEY] = self.analog_values_y[CURR_ANALOG_KEY]
        self.analog_values_y[CURR_ANALOG_KEY] = y_value

        self.analog_values_x[OLD_ANALOG_KEY] = self.analog_values_x[CURR_ANALOG_KEY]
        self.analog_values_x[CURR_ANALOG_KEY] = x_value

        if debug_printing_controller:
            print("Pred Left -> " + str(pred_left) + ", Pred Right -> " + str(pred_right))
            print("Y -> " + str(f'{self.analog_values_y[CURR_ANALOG_KEY]:.2f}'))
            print("X -> " + str(f'{self.analog_values_x[CURR_ANALOG_KEY]:.2f}'))
            print("___________________________")




    def reduce_speed(self):
        y_value = self.get_y_axis()
        y_value -= walking_decrement   # todo fix it later, maybe not walking
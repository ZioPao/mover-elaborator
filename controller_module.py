import pyxinput
from config import *

LEFT_AXIS_X = 'AxisLx'
LEFT_AXIS_Y = 'AxisLy'

OLD_ANALOG_KEY = 'old'
CURR_ANALOG_KEY = 'current'


class Controller:

    def __init__(self):
        # Setup virtual controller
        self.controller = pyxinput.vController()
        self.analog_values = {'old': 0, 'current': 0}
        self.prev_prediction = None

    def get_new_analog_values(self, increment, decrease, top_value, ):
        if self.analog_values[OLD_ANALOG_KEY] < top_value:
            new_y_value = self.analog_values[OLD_ANALOG_KEY] + increment
        else:
            new_y_value = self.analog_values[OLD_ANALOG_KEY] - decrease

        return new_y_value

    def set_analog(self, preds):

        # if there is only one 3, then we have to ignore it and choose the latter

        try:
            first_prediction = preds[0]
            second_prediction = preds[1]

            if first_prediction == 0. and second_prediction == 0.:
                self.choose_prediction(0.)

            else:
                # only one is stopped
                if (first_prediction == 0.) ^ (second_prediction == 0.):

                    if first_prediction == 0.:
                        self.choose_prediction(second_prediction)
                    else:
                        self.choose_prediction(first_prediction)
                else:
                    if self.prev_prediction == first_prediction:
                        self.choose_prediction(first_prediction)
                    else:
                        self.choose_prediction(second_prediction)

        except IndexError:
            pass

    def choose_prediction(self, prediction):
        # todo better if ints and not floats

        # at least 2 same prediction in a row to do something.
        new_y_value = 0

        if prediction == self.prev_prediction:

            if prediction == 0.:
                # stopped
                if -0.1 < self.analog_values[OLD_ANALOG_KEY] < stopped_range:
                    new_y_value = 0
                else:
                    new_y_value = self.analog_values[OLD_ANALOG_KEY] - stopped_decrement
            if prediction == 1.:
                # walking
                new_y_value = self.get_new_analog_values(walking_increment, walking_decrement, walking_top)
            if prediction == 2.:
                # jogging
                new_y_value = self.get_new_analog_values(jogging_increment, jogging_decrement, jogging_top)

                pass
        else:
            new_y_value = self.analog_values[OLD_ANALOG_KEY] - stopped_decrement

        # check overflow and underflow
        if new_y_value < 0.:
            new_y_value = 0.

        if new_y_value > 1.:
            new_y_value = 1.        # cap it off to max value

        self.controller.set_value(LEFT_AXIS_Y, new_y_value)
        # back them up
        self.analog_values[OLD_ANALOG_KEY] = self.analog_values[CURR_ANALOG_KEY]
        self.analog_values[CURR_ANALOG_KEY] = new_y_value
        self.prev_prediction = prediction       # salves old prediction

        if debug_printing_controller:
            print("Pred -> " + str(prediction))
            print("Old Y -> " + str(f'{self.analog_values[OLD_ANALOG_KEY]:.2f}'))
            print("New Y -> " + str(f'{self.analog_values[CURR_ANALOG_KEY]:.2f}'))
            print("___________________________")

    def get_y_axis(self):
        return self.analog_values[CURR_ANALOG_KEY]

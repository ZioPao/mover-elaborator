import pyxinput
from config import *

LEFT_AXIS_X = 'AxisLx'
LEFT_AXIS_Y = 'AxisLy'

OLD_ANALOG_KEY = 'old'
CURR_ANALOG_KEY = 'current'


NORMAL_MOV = 0.
LEFT_MOV = 1.
RIGHT_MOV = 2.
JUMP_MOV = 3.

class Controller:
    # RECAP VALUES
    # 0 -> moving (running determined by something different maybe
    # 1 -> left mov
    # 2 -> right mov
    # 3 -> jump




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

    def get_new_analog_values(self, increment, decrease, top_value, ):
        if self.analog_values_y[OLD_ANALOG_KEY] < top_value:
            new_y_value = self.analog_values_y[OLD_ANALOG_KEY] + increment
        else:
            new_y_value = self.analog_values_y[OLD_ANALOG_KEY] - decrease

        return new_y_value

    def set_correct_predictions(self, preds):
        # will enter only if there's movement.
        # if there is only one 3, then we have to ignore it and choose the latter

        try:
            first_prediction = preds[0]
            second_prediction = preds[1]

            # NORMAL MOVEMENT
            if first_prediction == 0. and second_prediction == 0.:
                self.choose_prediction(NORMAL_MOV)

            # LEFT MOVEMENT
            if first_prediction == 1.:
                self.choose_prediction(LEFT_MOV)

            # RIGHT MOVEMENT
            if second_prediction == 2.:
                self.choose_prediction(RIGHT_MOV)

            # JUMP
            if first_prediction == 3. and second_prediction == 3.:
                self.choose_prediction(JUMP_MOV)
        except IndexError:
            pass    # should never happen

    def reduce_speed(self):
        y_value = self.get_y_axis()
        y_value -= walking_decrement   # todo fix it later, maybe not walking


    def choose_prediction(self, prediction):
        # todo better if ints and not floats

        # at least 2 same prediction in a row to do something.
        new_y_value = 0

        if prediction == self.prev_prediction:

            if prediction == 0.:
                # stopped
                if -0.1 < self.analog_values_y[OLD_ANALOG_KEY] < stopped_range:
                    new_y_value = 0
                else:
                    new_y_value = self.analog_values_y[OLD_ANALOG_KEY] - stopped_decrement
            if prediction == 1.:
                # walking
                new_y_value = self.get_new_analog_values(walking_increment, walking_decrement, walking_top)
            if prediction == 2.:
                # jogging
                new_y_value = self.get_new_analog_values(jogging_increment, jogging_decrement, jogging_top)

                pass
        else:
            new_y_value = self.analog_values_y[OLD_ANALOG_KEY] - stopped_decrement

        # check overflow and underflow
        if new_y_value < 0.:
            new_y_value = 0.

        if new_y_value > 1.:
            new_y_value = 1.        # cap it off to max value

        self.controller.set_value(LEFT_AXIS_Y, new_y_value)


        # back them up
        self.analog_values_y[OLD_ANALOG_KEY] = self.analog_values_y[CURR_ANALOG_KEY]
        self.analog_values_y[CURR_ANALOG_KEY] = new_y_value

        #self.analog_values_x[OLD_ANALOG_KEY] = self.analog_values_x[CURR_ANALOG_KEY]
        #self.analog_values_x[CURR_ANALOG_KEY] = new_x_value







        self.prev_prediction = prediction       # salves old prediction

        if debug_printing_controller:
            print("Pred -> " + str(prediction))
            print("Old Y -> " + str(f'{self.analog_values_y[OLD_ANALOG_KEY]:.2f}'))
            print("New Y -> " + str(f'{self.analog_values_y[CURR_ANALOG_KEY]:.2f}'))
            print("___________________________")





    def get_y_axis(self):
        return self.analog_values_y[CURR_ANALOG_KEY]

    def get_x_axis(self):
        return self.analog_values_x[CURR_ANALOG_KEY]
import pyxinput

LEFT_AXIS_X = 'AxisLx'
LEFT_AXIS_Y = 'AxisLy'

OLD_ANALOG_KEY = 'old'
CURR_ANALOG_KEY = 'current'

# 0 = Jogging
# 1 = Walking
# 2 = Upstairs (Maybe reusable for jumping?)
# 6 = Stopped
JOGGING_INCREMENT = 10
WALKING_INCREMENT = 2
STOPPED_INCREMENT = 0
#UPSTAIRS_INCREMENT = ..


class Controller:

    def __init__(self):
        # Setup virtual controller
        self.controller = pyxinput.vController()
        self.data_reading = pyxinput.rController(1)     # we cannot read back the analog stick values normally

        self.analog_values = {'old': [], 'current': []}
        self.prev_prediction = None

    def choose_prediction(self, prediction):
        # todo better if ints and not floats


        new_x_value = 0
        new_y_value = 0

        if prediction == 0.:

            # jogging
            new_x_value = self.analog_values[OLD_ANALOG_KEY][0] + JOGGING_INCREMENT
            new_y_value = self.analog_values[OLD_ANALOG_KEY][1] + JOGGING_INCREMENT

        if prediction == 1.:
            # walking
            new_x_value = self.analog_values[OLD_ANALOG_KEY][0] + WALKING_INCREMENT
            new_y_value = self.analog_values[OLD_ANALOG_KEY][1] + WALKING_INCREMENT
        if prediction == 2.:
            # upstairs
            pass
        if prediction == 6.:        # todo change it to 3
            new_x_value = self.analog_values[OLD_ANALOG_KEY][0] + STOPPED_INCREMENT
            new_y_value = self.analog_values[OLD_ANALOG_KEY][1] + STOPPED_INCREMENT

        self.controller.set_value(LEFT_AXIS_X, new_x_value)
        self.controller.set_value(LEFT_AXIS_Y, new_y_value)
        # back them up
        self.analog_values[OLD_ANALOG_KEY] = self.analog_values[CURR_ANALOG_KEY]
        self.analog_values[CURR_ANALOG_KEY] = [new_x_value, new_y_value]

    def set_analog(self, preds, mov_list):

        # todo how do we use Z? as a multiplier?
        first_prediction = preds[0]
        second_prediction = preds[1]

        if first_prediction != second_prediction:
            # uses old prediction to guess which one to use
            if self.prev_prediction == first_prediction:
                self.choose_prediction(first_prediction)
            else:
                self.choose_prediction(second_prediction)
        else:
            self.choose_prediction(first_prediction)



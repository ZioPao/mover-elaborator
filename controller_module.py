import pyxinput

#####################
DEBUG = 1
#####################

LEFT_AXIS_X = 'AxisLx'
LEFT_AXIS_Y = 'AxisLy'

OLD_ANALOG_KEY = 'old'
CURR_ANALOG_KEY = 'current'

# 0 = Jogging
# 1 = Walking
# 2 = Upstairs (Maybe reusable for jumping?)
# 3 = Stopped
JOGGING_INCREMENT = 0.1
JOGGING_DECREASE = 0.015
JOGGING_TOP = 1

WALKING_INCREMENT = 0.1
WALKING_DECREASE = 0.005
WALKING_TOP = 0.2

STOPPED_DECREMENT = 0.05        # todo needs some sort of weightning system
STOPPED_MIN = 0.05
STOPPED_RANGE = 0.05


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

            if first_prediction == 3. and second_prediction == 3.:
                self.choose_prediction(3.)

            else:
                # only one is stopped
                if (first_prediction == 3.) ^ (second_prediction == 3.):

                    if first_prediction == 3.:
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
        new_y_value = 0

        if prediction == 0.:
            # jogging
            new_y_value = self.get_new_analog_values(JOGGING_INCREMENT, JOGGING_DECREASE, JOGGING_TOP)
        if prediction == 1.:
            # walking
            new_y_value = self.get_new_analog_values(WALKING_INCREMENT, WALKING_DECREASE, WALKING_TOP)
        if prediction == 2.:
            # upstairs
            new_y_value = self.analog_values[OLD_ANALOG_KEY]
            # should be jumping

            pass
        if prediction == 3.:
            # stopped
            if -0.1 < self.analog_values[OLD_ANALOG_KEY] < STOPPED_RANGE:
                new_y_value = 0
            else:
                new_y_value = self.analog_values[OLD_ANALOG_KEY] - STOPPED_DECREMENT

        # check overflow
        if new_y_value > 1.:
            new_y_value = 1.        # cap it off to max value

        self.controller.set_value(LEFT_AXIS_Y, new_y_value)
        # back them up
        self.analog_values[OLD_ANALOG_KEY] = self.analog_values[CURR_ANALOG_KEY]
        self.analog_values[CURR_ANALOG_KEY] = new_y_value

        if DEBUG:
            print("Pred -> " + str(prediction))
            print("Old Y -> " + str(f'{self.analog_values[OLD_ANALOG_KEY]:.2f}'))
            print("New Y -> " + str(f'{self.analog_values[CURR_ANALOG_KEY]:.2f}'))
            print("___________________________")

    def get_y_axis(self):
        return self.analog_values[CURR_ANALOG_KEY]

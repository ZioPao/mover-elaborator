import pyxinput

#####################
DEBUG = 0
#####################

LEFT_AXIS_X = 'AxisLx'
LEFT_AXIS_Y = 'AxisLy'

OLD_ANALOG_KEY = 'old'
CURR_ANALOG_KEY = 'current'

# 0 = Jogging
# 1 = Walking
# 2 = Upstairs (Maybe reusable for jumping?)
# 6 = Stopped
JOGGING_INCREMENT = 0.1
JOGGING_DECREASE = 0.015
JOGGING_TOP = 1

WALKING_INCREMENT = 0.1
WALKING_DECREASE = 0.005
WALKING_TOP = 0.2

STOPPED_DECREMENT = 0.1        # needs some sort of weightning system
STOPPED_MIN = 0.05


STOPPED_RANGE = 0.15
#UPSTAIRS_INCREMENT = ..


class Controller:

    def __init__(self):
        # Setup virtual controller
        self.controller = pyxinput.vController()
        #self.data_reading = pyxinput.rController(1)     # we cannot read back the analog stick values normally

        self.analog_values = {'old': [0, 0], 'current': [0, 0]}
        self.prev_prediction = None

    def get_new_analog_values(self, increment, decrease, top_value, ):
        if self.analog_values[OLD_ANALOG_KEY][1] < top_value:
            #self.analog_values[OLD_ANALOG_KEY][0] < top_value and \


            #new_x_value = self.analog_values[OLD_ANALOG_KEY][0] + increment
            new_y_value = self.analog_values[OLD_ANALOG_KEY][1] + increment
        else:
            #new_x_value = self.analog_values[OLD_ANALOG_KEY][0]
            new_y_value = self.analog_values[OLD_ANALOG_KEY][1] - decrease     # decreases

        new_x_value = 0     # delete me
        return new_x_value, new_y_value

    def set_analog(self, preds):

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

    def choose_prediction(self, prediction):
        # todo better if ints and not floats

        # todo fix this, just for test
        new_x_value = 0
        new_y_value = 0

        if prediction == 0.:
            # jogging
            new_x_value, new_y_value = self.get_new_analog_values(JOGGING_INCREMENT, JOGGING_DECREASE, JOGGING_TOP)
        if prediction == 1.:
            # walking
            new_x_value, new_y_value = self.get_new_analog_values(WALKING_INCREMENT, WALKING_DECREASE, WALKING_TOP)
        if prediction == 2.:
            # upstairs
            pass
        if prediction == 6.:        # todo change it to 3
            # stopped
            if STOPPED_RANGE > self.analog_values[OLD_ANALOG_KEY][1] > -STOPPED_RANGE: #and \
                    #STOPPED_RANGE > self.analog_values[OLD_ANALOG_KEY][0] > -STOPPED_RANGE:
                new_x_value = 0
                new_y_value = 0
            else:
                #new_x_value = self.analog_values[OLD_ANALOG_KEY][0] - STOPPED_DECREMENT
                new_y_value = self.analog_values[OLD_ANALOG_KEY][1] - STOPPED_DECREMENT

        if DEBUG:
            print("Pred -> " + str(prediction))
            #print("Old X ->" + str(self.analog_values[OLD_ANALOG_KEY][0]))
            print("Old Y -> " + str(self.analog_values[OLD_ANALOG_KEY][1]))
            #print("New X ->" + str(new_x_value))
            print("New Y -> " + str(new_y_value))
            print("___________________________")

        #self.controller.set_value(LEFT_AXIS_X, new_x_value)
        self.controller.set_value(LEFT_AXIS_Y, new_y_value)
        # back them up
        self.analog_values[OLD_ANALOG_KEY] = self.analog_values[CURR_ANALOG_KEY]
        self.analog_values[CURR_ANALOG_KEY] = [new_x_value, new_y_value]

    def get_y_axis(self):
        return self.analog_values[CURR_ANALOG_KEY][1]

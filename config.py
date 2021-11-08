import configparser
import threading
import time

# MAPPING
# 0 = Jogging
# 1 = Walking
# 2 = Upstairs (Maybe reusable for jumping?)
# 3 = Stopped


# Miscellaneous
config = configparser.ConfigParser()
config.read('settings.ini')


data_divider = int(config['Data']['data_divider'])


#####################
debug_printing_controller = bool(int(config['Debug']['debug_printing_controller']))
debug_printing_receiver = bool(int(config['Debug']['debug_printing_receiver']))
#####################

# 0 = Jogging
# 1 = Walking
# 2 = Upstairs (Maybe reusable for jumping?)
# 3 = Stopped

jogging_increment = float(config['Movement']['jogging_increment'])
jogging_decrement = float(config['Movement']['jogging_decrement'])
jogging_top = float(config['Movement']['jogging_top'])

walking_increment = float(config['Movement']['walking_increment'])
walking_decrement = float(config['Movement']['walking_decrement'])
walking_top = float(config['Movement']['walking_top'])

side_increment = float(config['Movement']['side_increment'])
side_decrement = float(config['Movement']['side_decrement'])
side_top = float(config['Movement']['side_top'])

stopped_decrement = float(config['Movement']['stopped_decrement'])
stopped_min = float(config['Movement']['stopped_min'])
stopped_range = float(config['Movement']['stopped_range'])


def read_config():
    config.read('settings.ini')
    time.sleep(5)


threading.Thread(target=read_config).start()



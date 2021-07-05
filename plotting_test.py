from collections import deque
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.ticker import FuncFormatter

def init():
    line.set_ydata([np.nan] * len(x))
    return line,

def animate(i):
    # Add next value
    data.append(np.random.randint(0, max_rand))
    line.set_ydata(data)
    plt.savefig('e:\\python temp\\fig_{:02}'.format(i))
    print(i)
    return line,

max_x = 10
max_rand = 5

data = deque(np.zeros(max_x), maxlen=max_x)  # hold the last 10 values
x = np.arange(0, max_x)

fig, ax = plt.subplots()
ax.set_ylim(0, max_rand)
ax.set_xlim(0, max_x-1)
line, = ax.plot(x, np.random.randint(0, max_rand, max_x))
ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: '{:.0f}s'.format(max_x - x - 1)))
plt.xlabel('Seconds ago')

ani = animation.FuncAnimation(
    fig, animate, init_func=init, interval=1000, blit=True, save_count=10)

plt.show()





import numpy as np
s_time = mov.values_prediction_test[0][5]

act = 1.0
prediction_list = []
temp_list = []
container_list = [act]
counter = 1


for p_l in mov.values_prediction_test:

    # clamp to 30 values...?
    if counter < 31:
        temp_val = np.array([p_l[0], p_l[1], p_l[2],
                    p_l[3], p_l[4]], dtype=float)
        temp_list.append(temp_val)
        # add in list
    else:
        print(len(prediction_list))
        #container_list.append(temp_list)
        prediction_list.append(temp_list)
        container_list = []  # activity at the start I guess
        temp_list = []
        counter = 0
        # create new series/array to store stuff and append to the main one
    counter += 1

prediction_list_np = np.array(prediction_list)


pickle.dump(prediction_list, open('values_predictions_mov1.bin', 'wb'))
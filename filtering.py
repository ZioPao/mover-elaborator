import numpy as np


def filtering_pass(frame):
    b = [0.00066048, 0.00132097, 0.00066048]
    a = [-1, 1.92600086, -0.92864279]
    nb = len(b)
    final_frame = []
    for axis in frame:
        single_f_frame = np.zeros(len(axis))
        for m in range(3, len(axis)):
            single_f_frame[m] = b[0] * axis[m]
            for i in range(1, nb):
                single_f_frame[m] += a[i] * single_f_frame[m - i] + b[i] * axis[m - i]

        # fix the first three values
        single_f_frame = single_f_frame[3:]
        final_frame.append(tuple(single_f_frame))

    return tuple(final_frame)

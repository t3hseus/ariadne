import numpy as np


def PtFromHelix(rows):

    h1, h2, h3 = rows.iloc[0], rows.iloc[1], rows.iloc[2]
    B = 1  # magnetic    field[T]
    tmp = 2 * ((h1.x - h3.x) * (h2.y - h3.y) - (h1.y - h3.y) * (h2.x - h3.x))

    if tmp == 0:  # three points on the same line
        return 0

    y0 = (h2.x - h1.x) * (h2.x - h3.x) * (h1.x - h3.x)
    y0 = y0 + (h2.y * h2.y - h3.y * h3.y) * (h1.x - h3.x)
    y0 = y0 - (h1.y * h1.y - h3.y * h3.y) * (h2.x - h3.x)
    y0 /= tmp

    x0 = (h1.x + h2.x) / 2
    if h1.x != h2.x:
        x0 += (h1.y - h2.y) * (h1.y + h2.y - 2 * y0) / (h1.x - h2.x) / 2
    else:
        x0 += (h1.y - h3.y) * (h1.y + h2.y - 2 * y0) / (h1.x - h3.x) / 2

    # // helix radius
    r02 = (h1.x - x0) * (h1.x - x0) + (h1.y - y0) * (h1.y - y0)

    pt = 0.3 * B * np.sqrt(r02)  # // Pt in [MeV / c]
    return pt
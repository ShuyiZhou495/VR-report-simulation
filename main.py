import cv2
import numpy as np

img = cv2.imread("images/example.png") // 2
height = 1000
img = cv2.resize(img, (height, int(img.shape[0] / (img.shape[1] / height))))
last = np.copy(img)

drawing = False
ix, iy = -1, -1
big = True


def draw_grid(lx, ly, big=True):
    flag = -1 if big else 1
    block = 6
    if lx <= ly:
        ax = lx // block
        ay = int(ly / (ly // ax))
    else:
        ay = ly // block
        ax = int(lx / (lx // ay))

    grid = np.zeros((ly * 2, lx * 2))
    coy = (lx ** 2) * ly / 100  # how pixels much to curve
    cox = (ly ** 2) * lx / 80

    # calculate peaks
    y1, y2 = [], []
    x1, x2 = [], []

    for w in range(lx // 2, ((lx // 2 + lx) // ax + 1) * ax, ax):
        h = ly // 2
        y = int(flag * ((h - ly) * ((w - lx) ** 2) - (h - ly) * ((lx // 2 - lx) ** 2)) / coy + h)
        y1.append(y)
        h = ((ly // 2 + ly) // ay) * ay
        y = int(flag * ((h - ly) * ((w - lx) ** 2) - (h - ly) * ((lx // 2 - lx) ** 2)) / coy + h)
        y2.append(y)

    for h in range(ly // 2, ((ly // 2 + ly) // ay + 1) * ay, ay):
        w = lx // 2
        x = int(flag * ((w - lx) * ((h - ly) ** 2) - (w - lx) * ((ly // 2 - ly) ** 2)) / cox + w)
        x1.append(x)
        w = ((lx // 2 + lx) // ax) * ax
        x = int(flag * ((w - lx) * ((h - ly) ** 2) - (w - lx) * ((ly // 2 - ly) ** 2)) / cox + w)
        x2.append(x)

    # calculate lines
    for h, xs, xe in zip(range(ly // 2, ly // 2 + ay * len(x1) + 1, ay), x1, x2):
        for w in range(xs, xe + 1):
            y = int(flag * ((h - ly) * ((w - lx) ** 2) - (h - ly) * ((lx // 2 - lx) ** 2)) / coy + h)
            if 2 * ly > y > 0:
                grid[y, w] = 255

    for w, ys, ye in zip(range(lx // 2, lx // 2 + ax * len(y1) + 1, ax), y1, y2):
        for h in range(ys, ye + 1):
            x = int(flag * ((w - lx) * ((h - ly) ** 2) - (w - lx) * ((ly // 2 - ly) ** 2)) / cox + w)
            if 2 * lx > x > 0:
                grid[h, x] = 255
        for h in list(range(0, ys)) + list(range(ye, 2 * ly)):
            x = int(-flag * ((w - lx) * ((h - ly) ** 2) - (w - lx) * ((ly // 2 - ly) ** 2)) / (cox * 10) + w)
            if 2 * lx > x > 0:
                grid[h, x] = 255

    grid[0:y1[0] - ay:ay, lx // 2: lx // 2 + lx] = 255
    grid[y2[0] + ay:2 * ly:ay, lx // 2: lx // 2 + lx] = 255
    grid[-1, lx // 2: lx // 2 + lx] = 255
    return np.repeat(grid[..., np.newaxis], 3, axis=-1)


def draw(event, x, y, flags, param):
    global ix, iy, drawing, big, last, img
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = not drawing
        if drawing:
            ix, iy = x, y
            cv2.circle(img, (ix, iy), 4, (0, 255, 255), thickness=-1)
        if not drawing:
            img = last.copy()
            sx, sy = min(ix, x), min(iy, y)
            ex, ey = max(ix, x), max(iy, y)
            lx, ly = ex - sx, ey - sy
            grid = draw_grid(lx, ly, big)

            gsx, gsy = 0, 0
            gex, gey = lx * 2, ly * 2

            imsx, imsy = sx - lx // 2, sy - ly // 2
            imex, imey = ex + lx - lx // 2, ey + ly - ly // 2

            if imsy < 0:
                gsy = -imsy
                imsy = 0
            if imsx < 0:
                gsx = -imsx
                imsx = 0
            if imex > img.shape[1]:
                gex -= imex - img.shape[1]
                imex = img.shape[1]
            if imey > img.shape[0]:
                gey -= imey - img.shape[0]
                imey = img.shape[0]

            grid = grid[gsy:gey, gsx:gex]
            last = img.copy()
            img[imsy:imey, imsx:imex][grid != 0] = \
                (img[imsy:imey, imsx:imex][grid != 0] + grid[grid != 0]) // 2

    elif drawing and event == cv2.EVENT_MOUSEMOVE:
        img = last.copy()
        cv2.rectangle(img, (ix, iy), (x, y), (0, 255, 255), 1)


if __name__ == '__main__':
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", draw)
    while (True):
        cv2.imshow('image', img)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('z'):
            # command z
            img = last.copy()
        elif k == ord("m"):
            big = not big
        elif k == 27:
            break
    cv2.destroyAllWindows()

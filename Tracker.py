# Simple bayesian correlation filter
# Programmed by Olac Fuentes
# Modified by Miguel Nunez
# Last modified Dec 2, 2018

import numpy as np
import cv2
import time


def get_coordinates(event, x, y, flags, param):
    global x0, x1, y0, y1
    if event == cv2.EVENT_LBUTTONDOWN:
        if x0 == -1:
            x0 = x
            y0 = y
            print('Entered x0 =', x0, 'y0 =', y0)
        elif x1 == -1:
            x1 = x + 1
            y1 = y + 1
            print('Entered x1 =', x1, 'y1 =', y1)


def get_pattern():
    global x0, x1, y0, y1
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', get_coordinates)
    while x1 == -1:
        # Grabbing footage
        ret, img = cap.read()
        img = cv2.GaussianBlur(img, (5, 5), 0)
        img = np.mean(img, axis=2) / 255
        cv2.imshow('image', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyWindow('image')
    return (img[y0:y1, x0:x1] / 255).astype(np.float)


x0 = x1 = y0 = y1 = -1
cap = cv2.VideoCapture(0)
pattern = get_pattern()
mean_filt = np.ones((pattern.shape[0], pattern.shape[1]))
mean_filt = mean_filt / np.sum(mean_filt)
py = pattern.shape[0]
px = pattern.shape[1]
cv2.imshow('pattern', pattern)
pattern[:, :] = pattern[:, :] - np.mean(pattern[:, :])

start = time.time()
count = 0
rows = 480
cols = 640
fm = np.zeros((rows, cols))
col_mat = np.tile(np.arange(cols), (rows, 1))
row_mat = np.tile(np.arange(rows), (cols, 1)).T
k = 1000
everyFrame = 1
while True:
    count += 1
    ret, frame = cap.read()
    if everyFrame == 1:
        frame = cv2.GaussianBlur(frame, (5, 5), 0)
        frame = np.mean(frame, axis=2)
        # Turn the frame black and white
        frame = (frame / 255).astype(np.float32)
        # Took out for loop looping through channels
        fm[:, :] = cv2.filter2D(frame[:, :], -1, mean_filt, anchor=(0, 0))
        centered_frame = frame - fm

        match = cv2.filter2D(centered_frame[:, :], -1, pattern[:, :], anchor=(0, 0))
        match = match / np.max(match)

        prior_x = col_mat - x0
        prior_y = row_mat - y0
        prior = prior_x * prior_x + prior_y * prior_y
        prior = np.exp(-prior / k)
        combined = match * prior
        cv2.imshow('prior probability', prior)
        # cv2.imwrite('Prior Probability.jpg', prior)
        cv2.imshow('match probability', match)
        # cv2.imwrite('Match Probability.jpg', match)
        cv2.imshow('combined probability', combined)
        # cv2.imwrite('Combined Probability.jpg', combined)
        y0, x0 = np.unravel_index(combined.argmax(), match.shape)
        # Black rectangle to show object detected
        cv2.rectangle(frame, (x0, y0), (x0 + px, y0 + py), (0, 255, 0), 1)
        cv2.imshow('frame', frame)
        # cv2.imwrite('Frame Probability.jpg', frame)
        everyFrame = 0
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        cv2.rectangle(frame, (x0, y0), (x0 + px, y0 + py), (0, 255, 0), 1)
        cv2.imshow('frame', frame)
        everyFrame += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


elapsed_time = time.time() - start
print('Capture speed: {0:.2f} frames per second'.format(count / elapsed_time))

cap.release()
cv2.destroyAllWindows()

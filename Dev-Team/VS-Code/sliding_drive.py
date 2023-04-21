import cv2, time
import numpy as np
import random

image_width, image_height = 640, 480

warp_image_width = 320
warp_image_height = 240

# sliding window height = warp_image_height / num_sliding_window
# sliding window width = width_sliding_window * 2

num_sliding_window = 20
width_sliding_window = 20
min_points = 5

warp_x_margin = 20
warp_y_margin = 3

lane_bin_th = 145

pts1_x, pts1_y = 290 - warp_x_margin, 290 - warp_y_margin
pts2_x, pts2_y = 100 - warp_x_margin, 410 + warp_y_margin
pts3_x, pts3_y = 440 + warp_x_margin, 290 - warp_y_margin
pts4_x, pts4_y = 580 + warp_x_margin, 400 + warp_y_margin

'''
pts1_x, pts1_y = 290, 290
pts2_x, pts2_y = 100, 410
pts3_x, pts3_y = 440, 290
pts4_x, pts4_y = 580, 400
'''

warp_src = np.array([
    [pts1_x, pts1_y],
    [pts2_x, pts2_y],
    [pts3_x, pts3_y],
    [pts4_x, pts4_y],
], dtype = np.float32)

warp_dist = np.array([
[0               , 0],
[0               , warp_image_height],
[warp_image_width, 0],
[warp_image_width, warp_image_height],
], dtype = np.float32)

capture = cv2.VideoCapture("/Users/1001l1000/Documents/Dev-Course/Dev-Team/VS-Code/track2.avi")
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

def onChange(pos):
    pass

'''
--- TrackBar ---

cv2.namedWindow("Trackbar Windows")
cv2.createTrackbar("threshold", "Trackbar Windows", 0, 255, onChange)
cv2.createTrackbar("minimum_points", "Trackbar Windows", 0, 24 * 24, onChange)
cv2.setTrackbarPos("threshold", "Trackbar Windows", lane_bin_th)
cv2.setTrackbarPos("minimum_points", "Trackbar Windows", min_points)
'''

def calibrate_image(src):
    global image_width, image_height

    calibrate_mtx = np.array([
        [ 350.354184, 0.0, 328.104147],
        [0.0, 350.652653, 236.540676],
        [0.0, 0.0, 1.0]
    ])
    dist = np.array([-0.289296, 0.061035, 0.001786, 0.015238, 0.0])

    cal_mtx, cal_roi = cv2.getOptimalNewCameraMatrix(calibrate_mtx, dist,
                            (image_width, image_height), 1, (image_width, image_height))

    dst = cv2.undistort(src, calibrate_mtx, dist, None, cal_mtx)
    x, y, w, h = cal_roi
    calibrated_image = dst[y : y + h, x : x + w]

    return cv2.resize(calibrated_image, (image_width, image_height))

def warp_image(img, src, dst, size):
    src_to_dst_mtx = cv2.getPerspectiveTransform(src, dst)
    dst_to_src_mtx = cv2.getPerspectiveTransform(dst, src)
    warp_img = cv2.warpPerspective(img, src_to_dst_mtx, size, flags = cv2.INTER_LINEAR)
    return warp_img, src_to_dst_mtx, dst_to_src_mtx

def draw_lines(img,x,y):
    global Offset
    x1 = x[0]
    x2 = x[4]
    y1 = y[0]
    y2 = y[4]
    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    img = cv2.line(img, (x1, y1), (x2, y2), color, 2)
    return img

def warp_process_image(image):
    global num_sliding_window
    global width_sliding_window
    global min_points
    global lane_bin_th

    # Gaussian Blurring
    blur = cv2.GaussianBlur(image, (5, 5), 0) 

    # detect white
    _, L, _ = cv2.split(cv2.cvtColor(blur, cv2.COLOR_BGR2HLS))  

    # lane_bin_th: threshold
    _, lane = cv2.threshold(L, lane_bin_th, 255, cv2.THRESH_BINARY)

    out_img = np.dstack((lane, lane, lane)) * 255

    dividen_lane = np.split(lane, num_sliding_window, axis = 0) # lane 이미지를 각 sliding window마다 적용 가능하도록 10개로 자름

    window_margin = 20  # 아래 윈도우의 x좌표보다 +- 50픽셀 이내의 다음 점을 찾기 위한 마진값

    lane_width = 200    # 차선 평균 픽셀 너비 = 240

    lx, ly, rx, ry = [], [], [], []
    l_box_center, r_box_center = [], []
    before_l_detected, before_r_detected = True, True

    for window in range(num_sliding_window - 1, 0, -1):
        left_lane_inds = []
        right_lane_inds = []
        histogram = np.sum(dividen_lane[window], axis = 0)
        midpoint = np.int32(histogram.shape[0] / 2)
        r_min_points, l_min_points = min_points, min_points

        if window == num_sliding_window - 1:        # 첫 window를 기반으로 그리기 위해 첫 window는 화면 중앙을 기준으로 좌, 우 차선을 나눔
            leftx_current = np.argmax(histogram[:midpoint])
            rightx_current = np.argmax(histogram[midpoint:]) + midpoint
            
        elif before_l_detected == True and before_r_detected == True:   # 이전 window에서 차선을 둘 다 인식했을 때 이번 window에서는 이전 윈도우의 x값 +- margin값 이내의 윈도우를 찾아냅니다.
            leftx_current = np.argmax(histogram[: lx[-1] + window_margin])
            rightx_current = np.argmax(histogram[rx[-1] - window_margin : rx[-1] + window_margin]) + rx[-1] - window_margin

        elif before_l_detected == False and before_r_detected == True:
            if rx[-1]  - lane_width > 0:
                leftx_current = np.argmax(histogram[:rx[-1] - lane_width])
                rightx_current = np.argmax(histogram[rx[-1] - window_margin :]) + histogram[rx[-1] - window_margin]
                if abs(leftx_current - rightx_current) < 100:
                    l_min_points = (width_sliding_window * 2) * (warp_image_height / num_sliding_window)

        elif before_l_detected == True and before_r_detected == False:   # 이전 window에서 차선을 하나만 인식했을 때, 있던 차선은 margin값 이내에서, 다른 차선은 lane width만큼 이동 후 찾아냅니다.
            leftx_current = np.argmax(histogram[max(0, lx[-1] - window_margin):lx[-1] + window_margin]) + max(0, lx[-1] - window_margin)
            rightx_current = np.argmax(histogram[min(lx[-1] + lane_width, histogram.shape[0] - 1):]) + min(lx[-1] + lane_width, histogram.shape[0] - 1)

            if rightx_current - leftx_current < 100:
                    r_min_points = (width_sliding_window * 2) * (warp_image_height / num_sliding_window)

        elif before_l_detected == False and before_r_detected == False:
            leftx_current = np.argmax(histogram[:midpoint])
            rightx_current = np.argmax(histogram[midpoint:]) + midpoint
        
        window_height = np.int32(lane.shape[0] / num_sliding_window)    # window_height = 24

        nz = dividen_lane[window].nonzero()

        win_yl = (window + 1) * window_height
        win_yh = window * window_height

        win_xll = leftx_current - width_sliding_window
        win_xlh = leftx_current + width_sliding_window
        win_xrl = rightx_current - width_sliding_window
        win_xrh = rightx_current + width_sliding_window

        # cv2.rectangle(out_img, (win_xll, win_yl), (win_xlh, win_yh), (0, 255, 0), 2)
        # cv2.rectangle(out_img, (win_xrl, win_yl), (win_xrh, win_yh), (0, 255, 0), 2)

        good_left_inds = ((nz[0] >= 0) & (nz[0] < window_height) & (nz[1] >= win_xll) & (nz[1] < win_xlh)).nonzero()[0]
        good_right_inds = ((nz[0] >= 0) & (nz[0] < window_height) & (nz[1] >= win_xrl) & (nz[1] < win_xrh)).nonzero()[0]

        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        if len(good_left_inds) > l_min_points:
            leftx_current = np.int32(np.mean(nz[1][good_left_inds]))
            cv2.rectangle(out_img, (win_xll, win_yl), (win_xlh, win_yh), (0, 255, 0), 2)
            l_box_center.append([(win_xll + win_xlh) // 2, (win_yl + win_yh) // 2])
            before_l_detected = True
        else:
            before_l_detected = False

        if len(good_right_inds) > r_min_points:
            rightx_current = np.int32(np.mean(nz[1][good_right_inds]))
            cv2.rectangle(out_img, (win_xrl, win_yl), (win_xrh, win_yh), (0, 255, 0), 2)
            r_box_center.append([(win_xrl + win_xrh) // 2, (win_yl + win_yh) // 2])
            before_r_detected = True
        else:
            before_r_detected = False

        lx.append(leftx_current)
        ly.append((win_yl + win_yh) / 2)
        rx.append(rightx_current)
        ry.append((win_yl + win_yh) / 2)
    
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        out_img[(window * window_height) + nz[0][left_lane_inds], nz[1][left_lane_inds]] = [255, 0, 0]
        out_img[(window * window_height) + nz[0][right_lane_inds], nz[1][right_lane_inds]] = [0, 0, 255]

    lfit = np.polyfit(np.array(ly), np.array(lx), 2)
    rfit = np.polyfit(np.array(ry), np.array(rx), 2)
        

    return lfit, rfit, np.mean(lx), np.mean(rx), out_img, l_box_center, r_box_center, lane

def get_line_pos(lst):
    global image_width
    low_slope_threshold = 0
    high_slope_threshold = 100

    slopes = []
    new_lines = []
    x1 = lst[0][0]
    x2 = lst[-1][0]
    y1 = lst[0][1]
    y2 = lst[-1][1]

    dx = x2 - x1

    if dx == 0:
        dx = 0.01

    slope = float(y2-y1) / float(dx)
    if abs(slope) > low_slope_threshold and abs(slope) < high_slope_threshold:
        return slope
    else:
        return float("inf")

while True:

    ''' 
    Trackbar

    lane_bin_th = cv2.getTrackbarPos("threshold", "Trackbar Windows")
    '''

    _, src = capture.read()
    cal = calibrate_image(src)
    warp, src_mtx, dst_mtx = warp_image(cal, warp_src, warp_dist, (warp_image_width, warp_image_height))
    l_fit, r_rit, l_m, r_m, out_img, l_box_center, r_box_center, lane = warp_process_image(warp)

    cv2.circle(cal, (pts1_x, pts1_y), 20, (255, 0, 0), -1)
    cv2.circle(cal, (pts2_x, pts2_y), 20, (255, 0, 0), -1)
    cv2.circle(cal, (pts3_x, pts3_y), 20, (255, 0, 0), -1)
    cv2.circle(cal, (pts4_x, pts4_y), 20, (255, 0, 0), -1)

    cv2.imshow("calibrated image 1", cal)
    cv2.imshow("bin", lane)
    cv2.imshow("warp", warp)
    cv2.imshow("box_img", out_img)

    if cv2.waitKey(10) == 27:
        break


'''
--- 수정사항 ---

--- 4.15---
* sliding window의 개수를 10개로 변경하였습니다.

--- 4.16 ---
* window를 그리는 순서를 위 -> 아래에서 아래 -> 위로 그리도록 변경하였습니다.
'''
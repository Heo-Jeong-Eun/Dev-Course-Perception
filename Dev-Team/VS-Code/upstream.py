# 2023-04-23 (J)

import cv2
import random
import time

import numpy as np

# 이미지 사이즈 = 640 * 480
image_width, image_height = 640, 480

# bird eye view로 변환한 이미지 사이즈 
warp_image_width = 320
warp_image_height = 240

# sliding window height = warp_image_height / num_sliding_window
# sliding window width = width_sliding_window * 2

# 슬라이딩 윈도우 갯수
num_sliding_window = 20
# 슬라이딩 윈도우 넓이
width_sliding_window = 20
# 선을 그리기 위해 최소한으로 있어야 하는 점의 갯수 
min_points = 5

# warp을 위해 좌표를 구할 때 사용될 마진값 
warp_x_margin = 20
warp_y_margin = 3

lane_bin_th = 145

roi_offset = 420

# warp 이전 4개 점의 좌표를 구하기 위해 마진값을 빼준다. 
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

# warp 이전 4개 점의 좌표 
warp_src = np.array([
    [pts1_x, pts1_y],
    [pts2_x, pts2_y],
    [pts3_x, pts3_y],
    [pts4_x, pts4_y],
], dtype = np.float32)

# warp 이후 4개 점의 좌표 
warp_dist = np.array([
[0, 0],
[0, warp_image_height],
[warp_image_width, 0],
[warp_image_width, warp_image_height],
], dtype = np.float32)


# track2 영상으로 테스트 하기 위해 영상 경로, 사이즈 작성 
capture = cv2.VideoCapture("/Users/1001l1000/Documents/Dev-Course/Dev-Team/VS-Code/track2.avi")
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# ?
def onChange(pos):
    pass

# calibration 
def calibrate_image(src):
    # 이미지 사이즈 = 640 * 480
    global image_width, image_height

    # 자이카 카메라의 calibration 보정값 
    # warp 변환 시 필요한 3x3 변환 행렬값 
    calibrate_mtx = np.array([
        [ 350.354184, 0.0, 328.104147],
        [0.0, 350.652653, 236.540676],
        [0.0, 0.0, 1.0]
    ])

    dist = np.array([-0.289296, 0.061035, 0.001786, 0.015238, 0.0])

    # getOptimalNewCameraMatrix는 calibaration에 필요한 mtx와 roi를 구한다. 
    cal_mtx, cal_roi = cv2.getOptimalNewCameraMatrix(calibrate_mtx, dist,
                            (image_width, image_height), 1, (image_width, image_height))

    # 위에서 구한 보정 행렬값을 적용하여 이미지를 반듯하게 수정-> undistort() 호출해서 이미지 수정
    # undistort = 어안렌즈인 경우 사용하는 함수 
    dst = cv2.undistort(src, calibrate_mtx, dist, None, cal_mtx)
    x, y, w, h = cal_roi
    calibrated_image = dst[y : y + h, x : x + w]

    # 반듯하게 펴진 image를 return 한다. 
    return cv2.resize(calibrated_image, (image_width, image_height))

# 변환전과 후의 4개 점 좌표를 전달해서 이미지를 원근 변환 처리하여 새로운 이미지로 만든다.
def warp_image(image, src, dst, size):
    src_to_dst_mtx = cv2.getPerspectiveTransform(src, dst)
    dst_to_src_mtx = cv2.getPerspectiveTransform(dst, src)
    warp_img = cv2.warpPerspective(image, src_to_dst_mtx, size, flags = cv2.INTER_LINEAR)
    
    return warp_img, src_to_dst_mtx, dst_to_src_mtx

# 선분 그리기
# 허프 변환 함수로 검출된 모든 선분을 다양한 색깔로 출력한다. 
def draw_lines(image, x, y):
    global roi_offset

    x1 = x[0]
    x2 = x[4]
    y1 = y[0]
    y2 = y[4]
    
    color = (random.randint(0, 255), 
             random.randint(0, 255), 
             random.randint(0, 255))
    
    # 허프 변환에서 사용하는 관심 영역에 넣기 위해 offset을 더해준다. 
    image = cv2.line(image, (x1, y1 + roi_offset), (x2, y2 + roi_offset), color, 2)
    
    return image

def warp_process_image(image):
    global num_sliding_window # 20
    global width_sliding_window # 20
    global min_points # 5
    global lane_bin_th # 145

    # 이미지에서 가우시안 블러링으로 노이즈 제거 
    blur = cv2.GaussianBlur(image, (5, 5), 0) 

    # detect white
    # 블러링 처리된 image의 흰선 구분을 위한 코드
    # HLS 포맷에서 L 채널을 이용하면 흰색 선을 쉽게 구분할 수 있다.
    # LAB 포맷에서는 B 채널을 이용하면 노란색 선을 쉽게 구분할 수 있다.
    _, L, _ = cv2.split(cv2.cvtColor(blur, cv2.COLOR_BGR2HLS))  

    # lane_bin_th: threshold
    # L 채널 이미지의 분할부를 확실하게 만들기 위해 바이너리화 한다. 
    # 임계값은 현재 이미지의 상태에 따라 낮추거나 올린다. 
    # 실습실의 경우 검은 차선이므로 THRESH_BINARY_INV를 사용해 검은색을 -> 흰색으로 바꿔준다. 
    _, lane = cv2.threshold(L, lane_bin_th, 255, cv2.THRESH_BINARY)

    # lane 이미지를 각 sliding window마다 적용 가능하도록 10개로 자른다. 
    dividen_lane = np.split(lane, num_sliding_window, axis = 0) 

    # 아래 윈도우의 x좌표보다 +- 50픽셀 이내의 다음 점을 찾기 위한 마진값
    window_margin = 20  

    # 차선 평균 픽셀 너비 = 240
    lane_width = 200  

    lx, ly, rx, ry = [], [], [], []
    l_box_center, r_box_center = [], []

    # ?
    before_l_detected, before_r_detected = True, True

    # ? 
    out_image = np.dstack((lane, lane, lane)) * 255

    # 윈도우 10개 그리기 
    for window in range((num_sliding_window - 1), 0, -1):
        left_lane_inds = []
        right_lane_inds = []

        # 히스토그램이란 이미지를 구성하는 픽셀값 분포에 대한 그래프이다. 
        # 히스토그램의 X와 Y축의 정의
        # x축 : 픽셀의 x 좌표값
        # y축 : 특정 x 좌표값을 갖는 모든 흰색 픽셀의 갯수  
        histogram = np.sum(dividen_lane[window], axis = 0)

        # x축, x좌표를 반으로 나누어 왼쪽 차선과 오른쪽 차선을 구분한다.
        midpoint = np.int32(histogram.shape[0] / 2)
        
        # r_min_points, l_min_points에 5를 저장한다. 
        r_min_points, l_min_points = min_points, min_points

        # 첫 window를 기반으로 그리기 위해 첫 window는 화면 중앙을 기준으로 좌, 우 차선을 나눈다. 
        if window == num_sliding_window - 1:      
            leftx_current = np.argmax(histogram[:midpoint])
            rightx_current = np.argmax(histogram[midpoint:]) + midpoint
            
        # 이전 window에서 차선을 둘 다 인식했을 때 이번 window에서는 이전 윈도우의 x값 +- margin값 이내의 윈도우를 찾아낸다. 
        elif before_l_detected == True and before_r_detected == True:   
            leftx_current = np.argmax(histogram[: lx[-1] + window_margin])
            rightx_current = np.argmax(histogram[rx[-1] - window_margin : rx[-1] + window_margin]) + rx[-1] - window_margin

        elif before_l_detected == False and before_r_detected == True:
            if rx[-1]  - lane_width > 0:
                leftx_current = np.argmax(histogram[:rx[-1] - lane_width])
                rightx_current = np.argmax(histogram[rx[-1] - window_margin :]) + histogram[rx[-1] - window_margin]
                if abs(leftx_current - rightx_current) < 100:
                    l_min_points = (width_sliding_window * 2) * (warp_image_height / num_sliding_window)

        # 이전 window에서 차선을 하나만 인식했을 때, 있던 차선은 margin값 이내에서, 다른 차선은 lane width만큼 이동 후 찾아낸다. 
        elif before_l_detected == True and before_r_detected == False:   
            leftx_current = np.argmax(histogram[max(0, lx[-1] - window_margin):lx[-1] + window_margin]) + max(0, lx[-1] - window_margin)
            rightx_current = np.argmax(histogram[min(lx[-1] + lane_width, histogram.shape[0] - 1):]) + min(lx[-1] + lane_width, histogram.shape[0] - 1)

            if rightx_current - leftx_current < 100:
                    r_min_points = (width_sliding_window * 2) * (warp_image_height / num_sliding_window)

        elif before_l_detected == False and before_r_detected == False:
            leftx_current = np.argmax(histogram[:midpoint])
            rightx_current = np.argmax(histogram[midpoint:]) + midpoint
        
        # window_height = 24
        window_height = np.int32(lane.shape[0] / num_sliding_window)    

        nz = dividen_lane[window].nonzero()

        win_yl = (window + 1) * window_height
        win_yh = window * window_height

        # width_sliding_window = 20
        win_xll = leftx_current - width_sliding_window
        win_xlh = leftx_current + width_sliding_window
        win_xrl = rightx_current - width_sliding_window
        win_xrh = rightx_current + width_sliding_window

        # cv2.rectangle(out_image, (win_xll, win_yl), (win_xlh, win_yh), (0, 255, 0), 2)
        # cv2.rectangle(out_image, (win_xrl, win_yl), (win_xrh, win_yh), (0, 255, 0), 2)

        # 슬라이딩 윈도우 박스 하나 안에 있는 흰색 픽셀의 x좌표를 모두 수집
        # 왼쪽과 오른쪽 슬라이딩 박스를 따로 작업한다. 
        # 이때 1번 window는 히스토그램으로 확보가 되어있는 상태이다. 
        good_left_inds = ((nz[0] >= 0) & (nz[0] < window_height) & (nz[1] >= win_xll) & (nz[1] < win_xlh)).nonzero()[0]
        good_right_inds = ((nz[0] >= 0) & (nz[0] < window_height) & (nz[1] >= win_xrl) & (nz[1] < win_xrh)).nonzero()[0]

        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # 위에서 구한 x좌표 리스트에서 흰색 점이 5개 이상인 경우에 한해서 x좌표의 평균값을 구한다.
        # 이 값을 위에 쌓을 슬라이딩 윈도우의 중심점으로 사용한다. -> for 반복, 10번
        # if 조건은 확실히 차선인 경우를 의미한다. 
        # 이 코드 이전에는 흰점 x 좌표를 수집하기만 하고 이후 평균값을 구한 뒤 다음에 쌓아야 하는 window의 위치가 정해진다. 
        if len(good_left_inds) > l_min_points:
            leftx_current = np.int32(np.mean(nz[1][good_left_inds]))
            cv2.rectangle(out_image, (win_xll, win_yl), (win_xlh, win_yh), (0, 255, 0), 2)
            l_box_center.append([(win_xll + win_xlh) // 2, (win_yl + win_yh) // 2])
            before_l_detected = True
        else:
            before_l_detected = False

        if len(good_right_inds) > r_min_points:
            rightx_current = np.int32(np.mean(nz[1][good_right_inds]))
            cv2.rectangle(out_image, (win_xrl, win_yl), (win_xrh, win_yh), (0, 255, 0), 2)
            r_box_center.append([(win_xrl + win_xrh) // 2, (win_yl + win_yh) // 2])
            before_r_detected = True
        else:
            before_r_detected = False

         # 슬라이딩 윈도우의 중심점(x좌표)을 lx / ly, rx / ry에 담아둔다. 
        lx.append(leftx_current)
        ly.append((win_yl + win_yh) / 2)
        rx.append(rightx_current)
        ry.append((win_yl + win_yh) / 2)
    
        # 10번의 loop가 끝나면 그동안 모은 점들을 저장한다. 
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # 기존 흰색 차선 픽셀을 왼쪽과 오른쪽 각각 파란색과 빨간색으로 색 변경
        out_image[(window * window_height) + nz[0][left_lane_inds], nz[1][left_lane_inds]] = [255, 0, 0]
        out_image[(window * window_height) + nz[0][right_lane_inds], nz[1][right_lane_inds]] = [0, 0, 255]

    # 슬라이딩 윈도우의 중심점(x좌표) 9개를 가지고 2차 함수를 만들어낸다. 
    # 2차 함수 -> x = ay^2 + by + c
    lfit = np.polyfit(np.array(ly), np.array(lx), 2)
    rfit = np.polyfit(np.array(ry), np.array(rx), 2)

    return lfit, rfit, np.mean(lx), np.mean(rx), out_image, l_box_center, r_box_center, lane

# 왼쪽 선분, 오른쪽 선분
def get_line_pos(list):
    global image_width
    low_slope_threshold = 0
    high_slope_threshold = 100

    x1 = list[0][0]
    x2 = list[-1][0]
    y1 = list[0][1]
    y2 = list[-1][1]

    dx = x2 - x1

    if dx == 0:
        dx = 0.01

    slope = float(y2-y1) / float(dx)
    
    # 선분의 기울기를 구해, 기울기 절대값이 0 초과, 100 미만인 것만 추출 
    if abs(slope) > low_slope_threshold and abs(slope) < high_slope_threshold:
        return slope
    else:
        return float("inf")

'''
# hough_drive code 
# 왼쪽 선분, 오른쪽 선분
def divide_left_right(lines):
    global VIDEO_WIDTH
    low_slope_threshold = 0
    high_slope_threshold = 10

    slopes = []
    new_lines = []

    for line in lines:
        x1, y1, x2, y2 = line[0]

        if x2 - x1 == 0:
            slope = 0
        else:
            slope = float(y2 - y1) / float(x2 - x1)
        
        # 선분의 기울기를 구해, 기울기 절대값이 10 이하인 것만 추출 
        if (abs(slope) > low_slope_threshold) and (abs(slope) < high_slope_threshold):
            slopes.append(slope)
            new_lines.append(line[0])

        # 허프 변환 함수로 검출한 선분들의 기울기를 비교해 왼쪽 차선과 오른쪽 차선을 구분한다. 
        left_lines = []
        right_lines = []

        # OpenCV 좌표계에서는 아래 방향으로 y가 증가하므로 기울기 계산법이 다르다. 
        # 화면의 왼쪽에 있는 선분 중 기울기가 음수인 것들만 수집
        # 화면의 오른쪽에 있는 선분 중에서 기울기가 양수인 것들만 수집
        for j in range(len(slopes)):
            line = new_lines[j]
            slope = slopes[j]
            x1, y1, x2, y2 = line

            if (slope < 0) and (x2 < VIDEO_WIDTH / 2 - 90):
                left_lines.append([line.tolist()])
            elif(slope > 0) and (x1 > VIDEO_WIDTH / 2 + 90):
                right_lines.append([line.tolist()])
        
        return left_lines, right_lines
'''

while True:
    _, src = capture.read()

    # calibration 작업 
    cal = calibrate_image(src)

    warp, src_mtx, dst_mtx = warp_image(cal, warp_src, warp_dist, (warp_image_width, warp_image_height))
    
    # ? -> l_m, r_m, out_image, l_box_center, r_box_center
    l_fit, r_rit, l_m, r_m, out_image, l_box_center, r_box_center, lane = warp_process_image(warp)

    # warp 작업 후 4개의 좌표 점 출력
    cv2.circle(cal, (pts1_x, pts1_y), 20, (255, 0, 0), -1)
    cv2.circle(cal, (pts2_x, pts2_y), 20, (255, 0, 0), -1)
    cv2.circle(cal, (pts3_x, pts3_y), 20, (255, 0, 0), -1)
    cv2.circle(cal, (pts4_x, pts4_y), 20, (255, 0, 0), -1)

    cv2.imshow("calibrated image 1", cal)
    cv2.imshow("bin", lane)
    cv2.imshow("warp", warp)
    cv2.imshow("box_img", out_image)

    # 종료 
    if cv2.waitKey(10) == 27:
        break

'''
--- 수정사항 ---

--- 4.15---
* sliding window의 개수를 10개로 변경하였습니다.

--- 4.16 ---
* window를 그리는 순서를 위 -> 아래에서 아래 -> 위로 그리도록 변경하였습니다.
'''
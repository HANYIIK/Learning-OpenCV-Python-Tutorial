# 车道检测
import cv2
import numpy as np

# 高斯滤波核大小
blur_ksize = 5
# Canny 边缘检测的 min、max 阈值
canny_min = 50
canny_max = 150

# 霍夫变换参数
rho = 1
theta = np.pi / 180
threshold = 15
min_line_len = 40
max_line_gap = 20


# 显示图像函数
def ShowImage(name_of_image, image, rate):
    img_min = cv2.resize(image, None, fx=rate, fy=rate, interpolation=cv2.INTER_CUBIC)
    cv2.namedWindow(name_of_image, cv2.WINDOW_NORMAL)
    cv2.imshow(name_of_image, img_min)
    if cv2.waitKey(0) == 27:  # wait for ESC to exit
        print('Not saved!')
        cv2.destroyAllWindows()
    elif cv2.waitKey(0) == ord('s'):  # wait for 's' to save and exit
        cv2.imwrite(name_of_image + '.jpg', image)  # save
        print('Saved successfully!')
        cv2.destroyAllWindows()


# 图像处理
def process_an_image(image):
    # 1、灰度化/高斯滤波/Canny 边缘检测
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_blur = cv2.GaussianBlur(image_gray, (blur_ksize, blur_ksize), 1)
    edges = cv2.Canny(image_blur, canny_min, canny_max)

    # 2、标记四个坐标点用于 ROI 截取
    rows, cols = edges.shape    # 540, 960
    print('edges shape: ', edges.shape)
    points = np.array([[(160, rows-100), (460, 335), (520, 335), (cols-160, rows-100)]])
    roi_edges = roi_mask(edges, points)

    # 3、霍夫直线提取
    drawing, lines = hough_lines(roi_edges, rho, theta, threshold, min_line_len, max_line_gap)
    draw_lines(image, lines)

    # 4、车道拟合计算
    draw_lanes(drawing, lines)

    result = cv2.addWeighted(image, 0.9, drawing, 0.2, 0)

    return result


def draw_lanes(image, lines):
    # 划分左右车道
    left_lines, right_lines = [], []
    for line in lines:
        for x1, y1, x2, y2 in line:
            k = (y2 - y1) / (x2 - x1)
            if k < 0:
                left_lines.append(line)
            else:
                right_lines.append(line)
    if len(left_lines) <= 0 or len(right_lines) <= 0:
        return

    # 清理异常数据
    clean_lines(left_lines, 0.1)
    clean_lines(right_lines, 0.1)

    # 得到左右车道线点的集合，拟合直线
    left_points = [(x1, y1) for line in left_lines for x1, y1, x2, y2 in line]
    left_points = left_points + [(x2, y2) for line in left_lines for x1, y1, x2, y2 in line]
    right_points = [(x1, y1) for line in right_lines for x1, y1, x2, y2 in line]
    right_points = right_points + [(x2, y2) for line in right_lines for x1, y1, x2, y2 in line]

    left_results = least_squares_fit(left_points, 325, img.shape[0])
    right_results = least_squares_fit(right_points, 325, img.shape[0])

    # 注意这里点的顺序
    vtxs = np.array([[left_results[1], left_results[0], right_results[0], right_results[1]]])

    # 填充车道区域
    cv2.fillPoly(image, vtxs, (0, 255, 0))


# 迭代计算斜率均值，排除掉与差值差异较大的数据
def clean_lines(lines, threshold_):
    slope = [(y2 - y1) / (x2 - x1) for line in lines for x1, y1, x2, y2 in line]
    while len(lines) > 0:
        mean = np.mean(slope)
        diff = [abs(s - mean) for s in slope]
        idx = np.argmax(diff)
        if diff[idx] > threshold_:
            slope.pop(idx)
            lines.pop(idx)
        else:
            break


# 最小二乘法拟合
def least_squares_fit(point_list, ymin, ymax):
    # 最小二乘法拟合
    x = [p[0] for p in point_list]
    y = [p[1] for p in point_list]

    # polyfit第三个参数为拟合多项式的阶数，所以1代表线性
    fit = np.polyfit(y, x, 1)
    fit_fn = np.poly1d(fit)  # 获取拟合的结果

    xmin = int(fit_fn(ymin))
    xmax = int(fit_fn(ymax))

    return [(xmin, ymin), (xmax, ymax)]


def hough_lines(image_gray, rho_, theta_, thresh, min_line_len_, max_line_gap_):
    # 统计概率霍夫直线变换
    lines = cv2.HoughLinesP(image_gray, rho_, theta_, thresh, minLineLength=min_line_len_, maxLineGap=max_line_gap_)

    # 新建白画布一张
    drawing = np.ones((image_gray.shape[0], image_gray.shape[1], 3), dtype=np.uint8)
    return drawing, lines


def draw_lines(image_bgr, lines):
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(image_bgr, (x1, y1), (x2, y2), [0, 0, 255], 1)


def roi_mask(image, corner_points):
    # 创建掩模
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, corner_points, 255)
    img_masked = cv2.bitwise_and(image, mask)
    ShowImage('mask', img_masked, 1)
    return img_masked

img_1 = cv2.imread('test6.jpg')
img = cv2.resize(img_1, (960, 540), interpolation=cv2.INTER_CUBIC)
res = process_an_image(img)

ShowImage('res___04', res, 1)
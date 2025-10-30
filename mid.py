import cv2
import numpy as np

# 讀取影片
cap = cv2.VideoCapture('LaneVideo.mp4')  # 修改這裡為你的影片路徑

ww, hh, rh, r = 640, 400, 0.6, 3

# === 新增：時間平滑與缺漏容忍的狀態 ===
left_s, left_b = None, None
right_s, right_b = None, None
miss_left = miss_right = 0
ALPHA = 0.206          # EMA 權重：越小越穩定（0.2~0.35之間調）
MISS_MAX = 20          # 可容忍連續遺失幀數

# === 新增：避免被外側線拉寬（最小改動） ===
WINDOW_PX = 26         # 搜尋窗半寬(px)，只收集落在上一幀左右邊界附近的線
MAX_HALF_GROWTH = 6    # 底部半寬每幀最多變化(px)
HALF_MIN, HALF_MAX = 100, 120  # 底部半寬全域上下限（避免變兩線道）

last_center_bottom = None    # 上一幀底部中心x
last_half_bottom   = None    # 上一幀底部半寬
last_center_top    = None    # top 端中心
last_half_top      = None    # top 端半寬

# === 漸層外觀參數（保持你的設定） ===
LANE_ALPHA_TOP    = 0.12   # 上緣透明度（0~1）
LANE_ALPHA_BOTTOM = 0.45   # 下緣透明度（0~1）
FEATHER_KSIZE     = 21     # 羽化(模糊)核大小，需為奇數

def ema(prev, curr, alpha):
    if prev is None:
        return curr
    return alpha * curr + (1 - alpha) * prev

def draw_lane(img1, p1, p2, p3, p4, y_top, y_bottom):
    # 1) 建立單通道多段 alpha map（上淡下濃）
    alpha = np.zeros((hh, ww), dtype=np.float32)
    h = y_bottom - y_top + 1
    if h > 0:
        grad_col = np.linspace(LANE_ALPHA_TOP, LANE_ALPHA_BOTTOM, h, dtype=np.float32)[:, None]
        grad = np.repeat(grad_col, ww, axis=1)
        alpha[y_top:y_bottom+1, :] = grad

    # 2) 多邊形區域遮罩
    poly = np.array([p1, p2, p4, p3], dtype=np.int32)
    mask = np.zeros((hh, ww), dtype=np.uint8)
    cv2.fillPoly(mask, [poly], 255)

    # 3) 限制在多邊形內並羽化
    alpha = alpha * (mask.astype(np.float32) / 255.0)
    if FEATHER_KSIZE >= 3 and FEATHER_KSIZE % 2 == 1:
        alpha = cv2.GaussianBlur(alpha, (FEATHER_KSIZE, FEATHER_KSIZE), 0)

    # 4) per-pixel alpha 混合綠色
    base = img1.astype(np.float32)
    out = base.copy()
    out[..., 0] = base[..., 0] * (1.0 - alpha) + 0.0 * alpha
    out[..., 1] = base[..., 1] * (1.0 - alpha) + 255.0 * alpha
    out[..., 2] = base[..., 2] * (1.0 - alpha) + 0.0 * alpha
    img2 = np.clip(out, 0, 255).astype(np.uint8)

    # 5) 邊界與中線
    cv2.polylines(img2, [poly], isClosed=True, color=(60, 220, 60), thickness=2, lineType=cv2.LINE_AA)
    mid_bottom = ((p1[0] + p3[0]) // 2, y_bottom)
    mid_top    = ((p2[0] + p4[0]) // 2, y_top)
    cv2.line(img2, mid_bottom, mid_top, (0, 0, 255), 2, cv2.LINE_AA)
    return img2

# === 延後建立 VideoWriter：等拿到第一張 img2 再決定 ===
out = None
writer_ready = False

# 嘗試順序：mp4(H.264) → mp4(mp4v) → avi(XVID) → avi(MJPG)
def try_open_writer(frame_size, fps_guess):
    trials = [
        ("lane_out.mp4", "avc1"),  # H.264 (需 ffmpeg/x264)
        ("lane_out.mp4", "mp4v"),  # MPEG-4 part 2
        ("lane_out.avi", "XVID"),  # Xvid AVI（很通用）
        ("lane_out.avi", "MJPG"),  # Motion-JPEG（最保險）
    ]
    for path, four in trials:
        fourcc = cv2.VideoWriter_fourcc(*four)
        wr = cv2.VideoWriter(path, fourcc, fps_guess, frame_size)
        if wr.isOpened():
            print(f"[writer] using {path} / fourcc={four}")
            return wr, path, four
    return None, None, None

# 讀 FPS，抓不到就預設 30
fps = cap.get(cv2.CAP_PROP_FPS)
if not fps or fps < 1:
    fps = 30.0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # 影片結束時退出

    img1 = cv2.resize(frame, (ww, hh))
    
    # 影像預處理
    gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    gray = cv2.dilate(gray, kernel)
    gray = cv2.GaussianBlur(gray, (5,5), 0)
    gray = cv2.erode(gray, kernel)
    edges = cv2.Canny(gray, 150, 200, L2gradient=True)
    
    # 設定 ROI 區域（可適度收窄上底，降低撿到隔壁車道）
    zero = np.zeros((hh, ww, 1), dtype='uint8')
    p1_roi, p2_roi = [r, hh-r], [ww-r, hh-r]
    p3_roi, p4_roi = [int(ww*0.55), int(hh*rh)], [int(ww*0.45), int(hh*rh)]  # 原本0.6/0.4 → 改0.55/0.45更集中
    pts = np.array([p1_roi, p2_roi, p3_roi, p4_roi])
    zone = cv2.fillPoly(zero, [pts], 255)
    edges_roi = cv2.bitwise_and(edges, zone)

    # 霍夫變換檢測車道線
    HOUGH_THRESHOLD, HOUGH_MIN_LINE_LENGTH, HOUGH_MAX_LINE_GAP = 40, 20, 70
    lines = cv2.HoughLinesP(edges_roi, 1, np.pi / 180, HOUGH_THRESHOLD, None, HOUGH_MIN_LINE_LENGTH, HOUGH_MAX_LINE_GAP)
    
    img2 = img1.copy()

    # === 改良的挑線：依底部交點靠近中心的左右線 + 時間平滑 ===
    center = ww // 2
    margin = 50                    # 中心邊界，避免挑到錯線（可調）
    y_bottom = hh - r
    y_top = int(hh - hh * 0.175)

    left_best = None   # (x_bottom, s, b)
    right_best = None  # (x_bottom, s, b)

    # 先嘗試找當幀的最佳左右線
    if lines is not None:
        use_window = (last_center_bottom is not None) and (last_half_bottom is not None)
        if use_window:
            exp_left_xb  = last_center_bottom - last_half_bottom
            exp_right_xb = last_center_bottom + last_half_bottom

        for line in lines:
            x1, y1, x2, y2 = line[0]
            dx = x2 - x1
            if abs(dx) < 1:
                continue  # 避免除以0

            length = np.hypot(dx, y2 - y1)
            if length < 25:
                continue  # 過短的碎線先丟

            s = (y2 - y1) / dx
            b = y1 - s * x1

            # 斜率篩選（依鏡頭視角調）：太平或太陡先過濾
            if not (-5.0 < s < -0.2 or 0.2 < s < 5.0):
                continue

            x_bottom_line = int((y_bottom - b) / s)
            if x_bottom_line < 30 or x_bottom_line > ww - 30:
                continue  # 太靠邊緣通常不是當前車道

            if use_window:
                # 只收靠近上一幀左右邊界的位置，避免外側新線
                if s < 0:
                    if abs(x_bottom_line - exp_left_xb) <= WINDOW_PX:
                        if (left_best is None) or (abs(x_bottom_line - exp_left_xb) < abs(left_best[0] - exp_left_xb)):
                            left_best = (x_bottom_line, s, b)
                else:
                    if abs(x_bottom_line - exp_right_xb) <= WINDOW_PX:
                        if (right_best is None) or (abs(x_bottom_line - exp_right_xb) < abs(right_best[0] - exp_right_xb)):
                            right_best = (x_bottom_line, s, b)
            else:
                # 初始化階段：用 margin 來分左右
                if s < 0 and x_bottom_line < center - margin:
                    if left_best is None or x_bottom_line > left_best[0]:
                        left_best = (x_bottom_line, s, b)
                if s > 0 and x_bottom_line > center + margin:
                    if right_best is None or x_bottom_line < right_best[0]:
                        right_best = (x_bottom_line, s, b)

    # 將當幀偵測結果用 EMA 與上一幀平滑，若缺漏則沿用舊值（有限度）
    if left_best is not None:
        _, s_now, b_now = left_best
        left_s = ema(left_s, s_now, ALPHA)
        left_b = ema(left_b, b_now, ALPHA)
        miss_left = 0
    else:
        miss_left += 1

    if right_best is not None:
        _, s_now, b_now = right_best
        right_s = ema(right_s, s_now, ALPHA)
        right_b = ema(right_b, b_now, ALPHA)
        miss_right = 0
    else:
        miss_right += 1

    # 若短暫缺漏（<= MISS_MAX）就沿用上一幀平滑後的線，避免閃爍
    have_left = left_s is not None and miss_left <= MISS_MAX
    have_right = right_s is not None and miss_right <= MISS_MAX

    if have_left and have_right:
        # === 兩邊都在：正常更新 ===
        x1b = int((y_bottom - left_b)  / left_s);   x1t = int((y_top - left_b)  / left_s)
        x2b = int((y_bottom - right_b) / right_s);  x2t = int((y_top - right_b) / right_s)

        # 寬度限制，避免突然變兩線道
        center_bottom_now = (x1b + x2b) // 2
        half_bottom_now   = int(np.clip((x2b - x1b) // 2, HALF_MIN, HALF_MAX))
        if last_half_bottom is not None:
            delta = half_bottom_now - last_half_bottom
            delta = np.clip(delta, -MAX_HALF_GROWTH, MAX_HALF_GROWTH)
            half_bottom_now = int(last_half_bottom + delta)

        width_bottom = max(1, (x2b - x1b))
        width_top    = max(1, (x2t - x1t))
        ratio_top    = width_top / width_bottom
        half_top_now = int(max(1, ratio_top * half_bottom_now))

        # 用限制後的中心+半寬重建四點
        x1b = int(np.clip(center_bottom_now - half_bottom_now, 0, ww-1))
        x2b = int(np.clip(center_bottom_now + half_bottom_now, 0, ww-1))
        center_top_now = (x1t + x2t) // 2
        x1t = int(np.clip(center_top_now - half_top_now, 0, ww-1))
        x2t = int(np.clip(center_top_now + half_top_now, 0, ww-1))

        # 更新穩定幾何（含 top 端）
        last_center_bottom = center_bottom_now
        last_half_bottom   = half_bottom_now
        last_center_top    = center_top_now
        last_half_top      = half_top_now

        img2 = draw_lane(img1, [x1b, y_bottom], [x1t, y_top],
                               [x2b, y_bottom], [x2t, y_top],
                               y_top, y_bottom)

    elif (have_left ^ have_right) and (last_center_bottom is not None) and (last_half_bottom is not None):
        # === 單邊暫失：沿用「上一幀中心＋寬度」重建四點（不更新狀態）
        cx_b = last_center_bottom
        hw_b = last_half_bottom
        cx_t = last_center_top if last_center_top is not None else cx_b
        hw_t = last_half_top   if last_half_top   is not None else int(hw_b * 0.7)  # 沒記錄就用比率近似

        x1b = int(np.clip(cx_b - hw_b, 0, ww-1))
        x2b = int(np.clip(cx_b + hw_b, 0, ww-1))
        x1t = int(np.clip(cx_t - hw_t, 0, ww-1))
        x2t = int(np.clip(cx_t + hw_t, 0, ww-1))

        img2 = draw_lane(img1, [x1b, y_bottom], [x1t, y_top],
                               [x2b, y_bottom], [x2t, y_top],
                               y_top, y_bottom)

    elif (last_center_bottom is not None) and (last_half_bottom is not None):
        # === 兩邊都暫失：同樣沿用最後穩定幾何（不更新狀態）
        cx_b = last_center_bottom
        hw_b = last_half_bottom
        cx_t = last_center_top if last_center_top is not None else cx_b
        hw_t = last_half_top   if last_half_top   is not None else int(hw_b * 0.7)

        x1b = int(np.clip(cx_b - hw_b, 0, ww-1))
        x2b = int(np.clip(cx_b + hw_b, 0, ww-1))
        x1t = int(np.clip(cx_t - hw_t, 0, ww-1))
        x2t = int(np.clip(cx_t + hw_t, 0, ww-1))

        img2 = draw_lane(img1, [x1b, y_bottom], [x1t, y_top],
                               [x2b, y_bottom], [x2t, y_top],
                               y_top, y_bottom)

    # === 建立/寫入輸出影片 ===
    if not writer_ready:
        out, out_path, out_fourcc = try_open_writer((ww, hh), fps)
        writer_ready = out is not None
        if not writer_ready:
            print("[writer] 無法建立任何影片寫入器，將只顯示不存檔")
    if writer_ready:
        out.write(img2)

    # 顯示
    cv2.imshow('Lane Detection', img2)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# 收尾
cap.release()
if out is not None:
    out.release()
cv2.destroyAllWindows()

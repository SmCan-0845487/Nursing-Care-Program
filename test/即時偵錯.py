import cv2
import time
import mediapipe as mp
import numpy as np
import pandas as pd
from datetime import datetime
from ultralytics import YOLO

mp_pose = mp.solutions.pose # mediapipe 姿勢偵測
mp_drawing = mp.solutions.drawing_utils # mediapipe 繪圖方法
mp_drawing_styles = mp.solutions.drawing_styles # mediapipe 繪圖樣式
yolo_model = YOLO('yolov8n.pt')  # YOLO 初始化，使用輕量版模型，也可用 yolov8s.pt, yolov8m.pt

#================= 各個函示 ====================
"""計算手臂彎曲角度（相對於垂直線）"""
def calculate_angle(p1, p2, p3):
    # 計算手肘處的彎曲角度的函數(肩膀->手肘->手腕)
    v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])# 從手肘指向肩膀的向量(x，y)
    v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])# 從手肘指向手腕的向量
    
    # 等等要用v1 · v2 = ||v1|| × ||v2|| × cos(θ)，內積跟長度推出cos theta值，再回推角度
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    cos_angle = np.clip(cos_angle, -1.0, 1.0)# 限制餘弦值範圍，避免數值誤差
    angle = np.arccos(cos_angle)# 是餘弦函數的反函數，可以回推角度
    return np.degrees(angle)

"""計算軀幹角度（相對於垂直線）"""
def calculate_trunk_angle(left_shoulder, right_shoulder, left_hip, right_hip):
    # 計算軀幹向量與垂直線之間的角度，小角度(輕微前傾或後仰)，0角度(軀幹完全垂直)，大角度(明顯的身體傾斜)
    # 計算肩部中點和髖部中點
    shoulder_mid = [(left_shoulder[0] + right_shoulder[0])/2, (left_shoulder[1] + right_shoulder[1])/2]
    hip_mid = [(left_hip[0] + right_hip[0])/2, (left_hip[1] + right_hip[1])/2]
    
    # 計算軀幹向量與垂直線的角度
    trunk_vector = np.array([shoulder_mid[0] - hip_mid[0], shoulder_mid[1] - hip_mid[1]])
    vertical_vector = np.array([0, -1])  # 向上為負（因為y軸向下增長）
    
    cos_angle = np.dot(trunk_vector, vertical_vector) / (np.linalg.norm(trunk_vector) * np.linalg.norm(vertical_vector))
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle = np.arccos(cos_angle)
    return np.degrees(angle)

"""計算頭部傾斜角度"""
def calculate_head_angle(nose, left_ear, right_ear):
    # 計算耳朵連線與水平線的角度
    ear_vector = np.array([right_ear[0] - left_ear[0], right_ear[1] - left_ear[1]])
    horizontal_vector = np.array([1, 0])  # 水平向右
    
    cos_angle = np.dot(ear_vector, horizontal_vector) / (np.linalg.norm(ear_vector) * np.linalg.norm(horizontal_vector))
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle = np.arccos(cos_angle)
    return np.degrees(angle)

"""計算肩部角度（手臂舉高角度）- 以軀幹為基準"""
def calculate_shoulder_angle(shoulder, elbow, hip):
    # 肩部角度：0°為手臂垂直向下，90°為水平，180°為舉手過頭
    # 計算手臂向量（肩膀到手肘）
    arm_vector = np.array([elbow[0] - shoulder[0], elbow[1] - shoulder[1]])
    # 計算軀幹向量（肩膀到髖部，向下為基準）
    trunk_vector = np.array([hip[0] - shoulder[0], hip[1] - shoulder[1]])
    
    cos_angle = np.dot(arm_vector, trunk_vector) / (np.linalg.norm(arm_vector) * np.linalg.norm(trunk_vector))
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle = np.arccos(cos_angle)
    return np.degrees(angle)

"""計算膝部彎曲角度"""
def calculate_knee_angle(hip, knee, ankle):
    # 膝部角度：180°為腿部完全伸直，角度越小表示膝蓋彎曲越多
    # 計算大腿向量（髖部到膝蓋）
    thigh_vector = np.array([knee[0] - hip[0], knee[1] - hip[1]])
    # 計算小腿向量（膝蓋到腳踝）
    calf_vector = np.array([ankle[0] - knee[0], ankle[1] - knee[1]])

    cos_angle = np.dot(thigh_vector, calf_vector) / (np.linalg.norm(thigh_vector) * np.linalg.norm(calf_vector))
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle = np.arccos(cos_angle)
    return np.degrees(angle)

"""從MediaPipe landmarks提取所有角度"""
def extract_pose_angles(landmarks):
    if len(landmarks) < 33:
        return None
    
    # 提取關鍵點座標
    left_shoulder = [landmarks[11].x, landmarks[11].y] # 肩膀
    right_shoulder = [landmarks[12].x, landmarks[12].y]
    left_elbow = [landmarks[13].x, landmarks[13].y] # 手肘
    right_elbow = [landmarks[14].x, landmarks[14].y]
    left_wrist = [landmarks[15].x, landmarks[15].y] # 手腕
    right_wrist = [landmarks[16].x, landmarks[16].y]
    left_hip = [landmarks[23].x, landmarks[23].y] # 臀部
    right_hip = [landmarks[24].x, landmarks[24].y]
    nose = [landmarks[0].x, landmarks[0].y]
    left_ear = [landmarks[7].x, landmarks[7].y]
    right_ear = [landmarks[8].x, landmarks[8].y]
    left_knee = [landmarks[25].x, landmarks[25].y] # 膝蓋
    right_knee = [landmarks[26].x, landmarks[26].y]
    left_ankle = [landmarks[27].x, landmarks[27].y] # 腳踝
    right_ankle = [landmarks[28].x, landmarks[28].y]
    
    try:
        angles = {
            'left_arm_angle': calculate_angle(left_shoulder, left_elbow, left_wrist),
            'right_arm_angle': calculate_angle(right_shoulder, right_elbow, right_wrist),
            'trunk_angle': calculate_trunk_angle(left_shoulder, right_shoulder, left_hip, right_hip),
            'head_tilt_angle': calculate_head_angle(nose, left_ear, right_ear),
            'left_shoulder_angle': calculate_shoulder_angle(left_shoulder, left_elbow, left_hip),
            'right_shoulder_angle': calculate_shoulder_angle(right_shoulder, right_elbow, right_hip),
            'left_knee_angle': calculate_knee_angle(left_hip, left_knee, left_ankle),
            'right_knee_angle': calculate_knee_angle(right_hip, right_knee, right_ankle)
        }
        return angles
    except:
        return None

"""找出時間戳記最接近的教練資料"""
def find_closest_frame(coach_data, target_timestamp, tolerance=0.05):
    if coach_data.empty:
        return None
    
    # 計算時間差異
    time_diff = abs(coach_data['timestamp'] - target_timestamp)
    
    # 找出最小差異的索引
    closest_idx = time_diff.idxmin()
    min_diff = time_diff.iloc[closest_idx]
    
    # 檢查是否在容忍範圍內
    if min_diff <= tolerance:
        return coach_data.iloc[closest_idx]
    else:
        return None

"""比對動作方向的一致性（基於前一幀比較），利用得到的角度來推斷"""
def calculate_direction_similarity(user_angles, user_previous_angles, coach_data, current_timestamp, previous_timestamp):
    # 如果沒有前一幀資料，就無法計算方向
    if user_previous_angles is None:
        return {}, "No previous frame data"
    
    coach_current = find_closest_frame(coach_data, current_timestamp)# 教練目前的角度
    coach_previous = find_closest_frame(coach_data, previous_timestamp)# 教練先前的角度
    
    if coach_current is None or coach_previous is None:
        return {}, "No coach reference data found"
    
    direction_scores = {}
    angle_names = ['left_arm_angle', 'right_arm_angle', 'trunk_angle', 'head_tilt_angle',
                    'left_shoulder_angle', 'right_shoulder_angle', 'left_knee_angle', 'right_knee_angle']
    
    for angle_name in angle_names:
        if angle_name not in user_angles or angle_name not in user_previous_angles:
            continue

        # 教練的角度變化方向
        coach_direction = coach_current[angle_name] - coach_previous[angle_name]
        # 用戶的角度變化方向  
        user_direction = user_angles[angle_name] - user_previous_angles[angle_name]
        
        # 比較方向一致性
        if abs(coach_direction) < 1 and abs(user_direction) < 1:
            direction_score = 90   # 都沒有明顯變化，給基本分
        elif (coach_direction > 0 and user_direction > 0) or (coach_direction < 0 and user_direction < 0):
            direction_score = 100 # 方向一致給 100分
        else:
            direction_score = 30    # 方向相反，但給予基礎分數
            
        direction_scores[f'{angle_name}_direction'] = direction_score
    
    return direction_scores, "Success"

"""比對動作變化幅度的相似性（基於前一幀比較）"""
def calculate_change_magnitude_similarity(user_angles, user_previous_angles, coach_data, current_timestamp, previous_timestamp):
    if user_previous_angles is None:
        return {}, "No previous frame data"
    
    coach_current = find_closest_frame(coach_data, current_timestamp)# 教練目前的角度
    coach_previous = find_closest_frame(coach_data, previous_timestamp)# 教練先前的角度
    
    if coach_current is None or coach_previous is None:
        return {}, "No coach reference data found"
    
    magnitude_scores = {}
    angle_names = ['left_arm_angle', 'right_arm_angle', 'trunk_angle', 'head_tilt_angle',
                    'left_shoulder_angle', 'right_shoulder_angle', 'left_knee_angle', 'right_knee_angle']
    
    for angle_name in angle_names:
        if angle_name not in user_angles or angle_name not in user_previous_angles:
            continue
            
        # 計算變化幅度
        coach_change_gap = abs(coach_current[angle_name] - coach_previous[angle_name])
        user_change_gap = abs(user_angles[angle_name] - user_previous_angles[angle_name])
        
        # 比較變化幅度的相似性
        if coach_change_gap < 1 and user_change_gap < 1:
            magnitude_score = 90  # 都沒有明顯變化
        else:
            magnitude_diff = abs(coach_change_gap - user_change_gap)
            # 角度是arm或是shoulder時，容忍範圍是5，否則其他為3，因為活動範圍通常比其他部位大
            tolerance = 5 if 'arm' in angle_name or 'shoulder' in angle_name else 3
            similarity = max(0, 1 - magnitude_diff / tolerance)
            magnitude_score = 50 + similarity * 50 # 範圍：50-100分
            
        magnitude_scores[f'{angle_name}_magnitude'] = magnitude_score
    
    return magnitude_scores, "Success"

"""綜合評分系統（基於前一幀比較）"""
def calculate_comprehensive_similarity(user_angles, user_previous_angles, coach_data, current_timestamp):
    # 計算前一個時間點（假設幀率相對穩定，往前推0.033秒）
    previous_timestamp = current_timestamp - 0.033  
    
    # 動作方向一致性
    direction_scores, direction_msg = calculate_direction_similarity(
        user_angles, user_previous_angles, coach_data, current_timestamp, previous_timestamp
    )
    # 動作變化幅度相似性
    magnitude_scores, magnitude_msg = calculate_change_magnitude_similarity(
        user_angles, user_previous_angles, coach_data, current_timestamp, previous_timestamp
    )
    # 如果無法計算，返回None
    if not direction_scores or not magnitude_scores:
        return None, f"Direction: {direction_msg}, Magnitude: {magnitude_msg}"
    
    # 綜合計算
    all_scores = {}
    all_scores.update(direction_scores)
    all_scores.update(magnitude_scores)
    
    # 計算總分（各角度的平均）
    total_score = 0
    score_count = 0
    
    # 定義權重
    angle_weights = {
        'left_arm_angle': 0.15,'right_arm_angle': 0.15,
        'trunk_angle': 0.2,'head_tilt_angle': 0.1,
        'left_shoulder_angle': 0.15,'right_shoulder_angle': 0.15,
        'left_knee_angle': 0.05,'right_knee_angle': 0.05,
    }
    
    for angle_name, weight in angle_weights.items():
        direction_key = f'{angle_name}_direction'
        magnitude_key = f'{angle_name}_magnitude'
        
        if direction_key in all_scores and magnitude_key in all_scores:
            # 方向和幅度各佔30以及70%
            combined_score = (all_scores[direction_key] * 0.3 + all_scores[magnitude_key] * 0.7)
            total_score += combined_score * weight # 得出來的分數再加權
            score_count += weight # 計算加權平均
    
    final_score = total_score / score_count if score_count > 0 else 0

    # ========== 反饋偵測邏輯 ==========
    track_id = user_angles.get('track_id', 'default')  # 假設track_id在user_angles中
    
    # 初始化該用戶的記錄
    if track_id not in angle_status_tracker:
        angle_status_tracker[track_id] = {}
    
    feedback_messages = []
    LOW_SCORE_THRESHOLD = 60
    CONSECUTIVE_FRAMES = 6
    
    for angle_name in angle_weights.keys():
        direction_key = f'{angle_name}_direction'
        magnitude_key = f'{angle_name}_magnitude'
        
        if direction_key in all_scores and magnitude_key in all_scores:
            # 計算該角度的分數
            angle_score = (all_scores[direction_key] * 0.3 + all_scores[magnitude_key] * 0.7)
            
            # 初始化該角度的記錄
            if angle_name not in angle_status_tracker[track_id]:
                angle_status_tracker[track_id][angle_name] = []
            
            # 記錄此幀是否低於閾值
            is_low_score = angle_score < LOW_SCORE_THRESHOLD
            angle_status_tracker[track_id][angle_name].append(is_low_score)
            
            # 只保留最近的幀數記錄
            if len(angle_status_tracker[track_id][angle_name]) > CONSECUTIVE_FRAMES:
                angle_status_tracker[track_id][angle_name].pop(0)
            
            # 檢查是否連續5次都是低分
            recent_status = angle_status_tracker[track_id][angle_name]
            if len(recent_status) == CONSECUTIVE_FRAMES and all(recent_status):
                # 連續5次低分，加入提醒訊息
                angle_chinese = {
                    'left_arm_angle': 'Left Arm', 'right_arm_angle': 'Right Arm',
                    'trunk_angle': 'Trunk', 'head_tilt_angle': 'Head',
                    'left_shoulder_angle': 'Left Shoulder', 'right_shoulder_angle': 'Right Shoulder',
                    'left_knee_angle': 'Left Knee', 'right_knee_angle': 'Right Knee'
                }
                chinese_name = angle_chinese.get(angle_name, angle_name)
                feedback_messages.append(f"{chinese_name}")
    # 將反饋訊息加入回傳結果
    all_scores['feedback_messages'] = feedback_messages
    return final_score, all_scores

#==================== 主要程式開始 =====================
cap = cv2.VideoCapture(0) # 鏡頭捕捉影片
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960) # 設定解析度
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
cap.set(cv2.CAP_PROP_FPS, 30) # 設定幀率

angle_status_tracker = {} # 全域變數 - 追蹤各角度的連續狀態
user_previous_angles = {}  # {track_id: previous_angles}
all_people_data = [] # 存儲所有幀的姿勢數據
feedback_display_timer = {}  # 操控訊息停留時間

SYSTEM_STATE = "PREVIEW"  # 有分正式開始記錄跟預錄
recording_started = False
coach_video_start_time = None
SHOW_VISUAL = True  # 是否顯示視覺化窗口
SAVE_SAMPLE_FRAMES = True  # 是否保存部分分析結果圖片

# 教練影片初始化（但不開始播放）
coach_data = pd.read_csv(r"C:\Users\e6797\OneDrive\Desktop\VR虛擬教練\第一週-分析\手臂伸展分析.csv") # 教練的角度資料
coach_video_path = r"C:\Users\e6797\OneDrive\Desktop\VR虛擬教練\第一週\手臂伸展_已剪輯.mp4"
coach_cap = cv2.VideoCapture(coach_video_path) 
coach_fps = coach_cap.get(cv2.CAP_PROP_FPS)

# 讀取第一幀作為預覽用
coach_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # 設定到第一幀
ret, coach_first_frame = coach_cap.read()
if ret:
    coach_preview_frame = cv2.resize(coach_first_frame, (1920, 1080))
else:
    coach_preview_frame = None

# min_detection_confidence 小時容易檢測到人，但可能有誤判，大則相反(最大為1)
# min_tracking_confidence 當已經鎖定人物後，用來判斷是否繼續追蹤的信心值。數值小追蹤較不穩定，容易重新檢測，反之
# 穩定環境之坐姿運動(建議 0.7，0.5) 。光線不佳或多人環境(建議0.4，0.4)
with mp_pose.Pose(min_detection_confidence = 0.7 , min_tracking_confidence = 0.8) as pose:
    if not cap.isOpened(): # 無法開啟影片檔案
        print("Cannot open camera")
        exit()
    
    frame_count = 0
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret: 
            break
        frame = cv2.flip(frame, 1)  # 水平翻轉，讓動作更直觀
        frame_count += 1
        display_frame = frame

        if SYSTEM_STATE == "PREVIEW":
            # 只做即時顯示，不記錄數據
            current_timestamp = None
            # 顯示靜態教練影片（第一幀）
            if coach_preview_frame is not None:
                cv2.imshow('Coach Video', coach_preview_frame)

            # 仍然要做YOLO偵測和MediaPipe處理來顯示
            yolo_results = yolo_model.track(display_frame, tracker="bytetrack.yaml")
            person_boxes = []
            for result in yolo_results:
                for box in result.boxes:
                    if box.cls == 0:  # class 0 是 'person'
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        confidence = box.conf[0].cpu().numpy()
                        track_id = int(box.id[0].cpu().numpy()) if hasattr(box, 'id') and box.id is not None else -1
                        if confidence > 0.5:
                            person_boxes.append((x1, y1, x2, y2, confidence, track_id))
            
            # 視覺化處理（PREVIEW模式也要顯示）
            for x1, y1, x2, y2, conf, track_id in person_boxes:
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)  # 藍色框表示預覽模式
                label = f"PREVIEW ID:{track_id}"
                cv2.putText(display_frame, label, (x1, y1-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        elif SYSTEM_STATE == "RECORDING":
            # 開始記錄並播放教練影片
            if not recording_started:
                recording_started = True
                coach_video_start_time = time.time()
                record_start_time = time.time()
                coach_cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # 重置影片到開頭
            current_timestamp = time.time() - start_time

            # 計算教練影片應該播放到第幾幀
            coach_elapsed_time = time.time() - coach_video_start_time
            coach_frame_number = int(coach_elapsed_time * coach_fps)
            # 設定教練影片播放位置
            coach_cap.set(cv2.CAP_PROP_POS_FRAMES, coach_frame_number)
            coach_ret, coach_frame = coach_cap.read()
            if coach_ret:# 調整教練影片大小後顯示
                coach_display = cv2.resize(coach_frame, (1920, 1080))
                cv2.imshow('Coach Video', coach_display)
            
            # 第一步：用YOLO偵測人物(檢測+追蹤)
            yolo_results = yolo_model.track(display_frame, tracker="bytetrack.yaml")

            # 提取人物的bounding boxes  
            person_boxes = []
            for result in yolo_results:
                for box in result.boxes:
                    if box.cls == 0:  # class 0 是 'person'
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        confidence = box.conf[0].cpu().numpy()
                        # 重點：正確提取 track_id
                        if hasattr(box, 'id') and box.id is not None:
                            track_id = int(box.id[0].cpu().numpy())
                        else:
                            track_id = -1  # 如果沒有追蹤ID，使用-1表示
                        
                        if confidence > 0.5:  # 信心度閾值
                            person_boxes.append((x1, y1, x2, y2, confidence, track_id))

            # 先繪製YOLO偵測框
            if SHOW_VISUAL or SAVE_SAMPLE_FRAMES: 
                for x1, y1, x2, y2, conf, track_id in person_boxes:
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"ID:{track_id} ({conf:.2f})"
                    cv2.putText(display_frame, label, (x1, y1-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # 第二步：對每個人分別進行MediaPipe姿勢分析
            frame_people_data = []
            detailed_scores = {}
            for x1, y1, x2, y2, conf, track_id in person_boxes:
                # 擴展bounding box以確保完整包含人物
                padding = 20
                x1_crop = max(0, x1 - padding)
                y1_crop = max(0, y1 - padding)
                x2_crop = min(display_frame.shape[1], x2 + padding)
                y2_crop = min(display_frame.shape[0], y2 + padding)
                
                # 裁切人物區域
                person_crop = display_frame[y1_crop:y2_crop, x1_crop:x2_crop]
                
                if person_crop.size > 0:
                    # 轉換顏色空間並進行姿勢偵測
                    rgb_crop = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
                    pose_results = pose.process(rgb_crop)

                    # 視覺化處理
                    if SHOW_VISUAL or SAVE_SAMPLE_FRAMES:
                        if pose_results.pose_landmarks:
                            # 在裁切區域繪製骨架
                            mp_drawing.draw_landmarks(
                                person_crop,
                                pose_results.pose_landmarks,
                                mp_pose.POSE_CONNECTIONS,
                                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                            )
                            # 將處理後的區域貼回原圖
                            display_frame[y1_crop:y2_crop, x1_crop:x2_crop] = person_crop
                    
                    if pose_results.pose_landmarks: # 將MediaPipe結果繪製到原始display_frame上
                        # 計算角度（使用原始相對座標）
                        angles = extract_pose_angles(pose_results.pose_landmarks.landmark)
                        if angles:
                            prev_angles = user_previous_angles.get(track_id, None)
                            # 計算綜合相似度（只有在有前一幀資料時才計算）
                            similarity_score = None
                            
                            if prev_angles is not None:  # 有前一幀資料才能計算
                                similarity_score, detailed_scores = calculate_comprehensive_similarity(
                                    angles, prev_angles, coach_data, current_timestamp
                                )
                            
                            person_data = {
                                'frame': frame_count,
                                'timestamp': current_timestamp,
                                'track_id': track_id,
                                **angles,
                                'pose_detected': True
                            }
                            
                            # 將相似度分數加入資料
                            if similarity_score is not None:
                                person_data['similarity_score'] = similarity_score
                                if isinstance(detailed_scores, dict):
                                    person_data.update(detailed_scores)
                            
                            frame_people_data.append(person_data)
                            # 儲存當前幀作為下一幀的"前一幀"
                            user_previous_angles[track_id] = angles.copy()

        # 只在 RECORDING 狀態才記錄數據
        if SYSTEM_STATE == "RECORDING":
            all_people_data.extend(frame_people_data)

        # 顯示整體資訊
        if SHOW_VISUAL or SAVE_SAMPLE_FRAMES:
            cv2.putText(display_frame, f"Time: {current_timestamp:.1f}s", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            # 如果沒有偵測到人，顯示提示
            if len(person_boxes) == 0:
                cv2.putText(display_frame, "No Person Detected", (10, 120), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                cv2.putText(display_frame, f"People detected: {len(person_boxes)}", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
            if 'feedback_messages' in detailed_scores and detailed_scores['feedback_messages']:
                # 如果是單一字串用逗號分隔
                if isinstance(detailed_scores['feedback_messages'], str):
                    messages = detailed_scores['feedback_messages'].split(',')
                else:
                    feedback_messages = detailed_scores['feedback_messages']

                # 將新訊息加入計時器（顯示5秒）
                for message in feedback_messages:
                    feedback_display_timer[message] = current_time + 5.0

                # 顯示所有還在計時器內的訊息
                current_time = time.time()
                active_messages = []
                for message, end_time in list(feedback_display_timer.items()):
                    if current_time < end_time: 
                        active_messages.append(message)
                    else:
                        del feedback_display_timer[message]  # 清除過期訊息
                for i, message in enumerate(feedback_messages):
                    cv2.putText(display_frame, f"{message}", (10, 90 + i * 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        # 顯示即時視窗 needs improvement
        if SHOW_VISUAL:
            cv2.imshow('Multi-Person Pose Analysis', display_frame)
            # 檢查視窗是否被關閉
            key = cv2.waitKey(1) & 0xFF
            if key == ord('r'):  # 按 'r' 開始記錄
                if SYSTEM_STATE == "PREVIEW":
                    SYSTEM_STATE = "RECORDING"
                    recording_started = False
            elif key == ord('x') or key == 27:
                break
            
cap.release()
coach_cap.release()  # 釋放教練影片
cv2.destroyAllWindows()
cv2.waitKey(1)

# 保存為CSV
df = pd.DataFrame(all_people_data)
csv_filename = f"person_pose_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
df.to_csv(csv_filename, index=False, encoding='utf-8-sig')
print(f"CSV 文件已保存: {csv_filename}")

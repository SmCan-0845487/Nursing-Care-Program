import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from datetime import datetime
from ultralytics import YOLO

# mediapipe 初始化
mp_pose = mp.solutions.pose # mediapipe 姿勢偵測
mp_drawing = mp.solutions.drawing_utils # mediapipe 繪圖方法
mp_drawing_styles = mp.solutions.drawing_styles # mediapipe 繪圖樣式

# YOLO 初始化
yolo_model = YOLO('yolov8n.pt')  # 使用輕量版模型，也可用 yolov8s.pt, yolov8m.pt

# 影片路徑，真實老人運動情形之影片
video_path = r"C:\Users\e6797\OneDrive\Desktop\VR虛擬教練\第一週\手臂伸展.mp4"
cap = cv2.VideoCapture(video_path) # 將原本的鏡頭捕捉改成捕捉影片

# 設定視覺顯示參數
SHOW_VISUAL = True  # 是否顯示視覺化窗口
SAVE_SAMPLE_FRAMES = True  # 是否保存部分分析結果圖片

# 存儲所有幀的姿勢數據
all_people_data = []
frame_data = []

# 計算手肘處的彎曲角度的函數(肩膀->手肘->手腕)
def calculate_angle(p1, p2, p3):
    v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])# 從手肘指向肩膀的向量(x，y)
    v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])# 從手肘指向手腕的向量
    
    # 等等要用v1 · v2 = ||v1|| × ||v2|| × cos(θ)，內積跟長度推出cos theta值，再回推角度
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    cos_angle = np.clip(cos_angle, -1.0, 1.0)# 限制餘弦值範圍，避免數值誤差
    angle = np.arccos(cos_angle)# 是餘弦函數的反函數，可以回推角度
    return np.degrees(angle)

# 計算軀幹向量與垂直線之間的角度，小角度(輕微前傾或後仰)，0角度(軀幹完全垂直)，大角度(明顯的身體傾斜)
def calculate_trunk_angle(left_shoulder, right_shoulder, left_hip, right_hip):
    """計算軀幹角度（相對於垂直線）"""
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

def calculate_head_angle(nose, left_ear, right_ear):
    """計算頭部傾斜角度"""
    # 計算耳朵連線與水平線的角度
    ear_vector = np.array([right_ear[0] - left_ear[0], right_ear[1] - left_ear[1]])
    horizontal_vector = np.array([1, 0])  # 水平向右
    
    cos_angle = np.dot(ear_vector, horizontal_vector) / (np.linalg.norm(ear_vector) * np.linalg.norm(horizontal_vector))
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle = np.arccos(cos_angle)
    return np.degrees(angle)

# 肩部角度：0°為手臂垂直向下，90°為水平，180°為舉手過頭
def calculate_shoulder_angle(shoulder, elbow, hip):
    """計算肩部角度（手臂舉高角度）- 以軀幹為基準"""
    # 計算手臂向量（肩膀到手肘）
    arm_vector = np.array([elbow[0] - shoulder[0], elbow[1] - shoulder[1]])
    # 計算軀幹向量（肩膀到髖部，向下為基準）
    trunk_vector = np.array([hip[0] - shoulder[0], hip[1] - shoulder[1]])
    
    cos_angle = np.dot(arm_vector, trunk_vector) / (np.linalg.norm(arm_vector) * np.linalg.norm(trunk_vector))
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle = np.arccos(cos_angle)
    return np.degrees(angle)

# 膝部角度：180°為腿部完全伸直，角度越小表示膝蓋彎曲越多
def calculate_knee_angle(hip, knee, ankle):
    """計算膝部彎曲角度"""
    # 計算大腿向量（髖部到膝蓋）
    thigh_vector = np.array([knee[0] - hip[0], knee[1] - hip[1]])
    # 計算小腿向量（膝蓋到腳踝）
    calf_vector = np.array([ankle[0] - knee[0], ankle[1] - knee[1]])

    cos_angle = np.dot(thigh_vector, calf_vector) / (np.linalg.norm(thigh_vector) * np.linalg.norm(calf_vector))
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle = np.arccos(cos_angle)
    return np.degrees(angle)

def extract_pose_angles(landmarks):
    """從MediaPipe landmarks提取所有角度"""
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

PROCESS_EVERY_N_FRAMES = 2  # 處理每2幀
# min_detection_confidence 小時容易檢測到人，但可能有誤判，大則相反(最大為1)
# min_tracking_confidence 當已經鎖定人物後，用來判斷是否繼續追蹤的信心值。數值小追蹤較不穩定，容易重新檢測，反之
# 穩定環境之坐姿運動(建議 0.7，0.5) 。光線不佳或多人環境(建議0.4，0.4)
with mp_pose.Pose(min_detection_confidence = 0.5 , min_tracking_confidence = 0.8) as pose:
    
    if not cap.isOpened(): # 無法開啟影片檔案
        print("Cannot open camera")
        exit()
    
    # 影片的總幀數跟每秒幀數
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret: 
            break
        
        frame_count += 1
        # 只處理指定的幀
        if frame_count % PROCESS_EVERY_N_FRAMES != 0:
            continue
        print(f"處理進度: {frame_count}/{total_frames} ({frame_count/total_frames*100:.1f}%)", end='\r')
        
        # 可以調整影像大小以提高處理速度，但大小會影響捕捉的精準度
        # [y1:y2, x1:x2]，可選要哪部分
        '''display_frame_lu = frame[0:360, 0:600] # 左上角
        display_frame_ru = frame[0:540, 960:1920] # 右上角
        display_frame_ld = frame[540:1080, 0:960] # 左下角
        display_frame_rd = frame[540:1080, 960:1920] # 右下角 '''
        display_frame = cv2.resize(frame, (360, 640))

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
        
        # 第二步：對每個人分別進行MediaPipe姿勢分析
        frame_people_data = []
        
        for x1, y1, x2, y2, conf, track_id in person_boxes:
            # 擴展bounding box以確保完整包含人物
            padding = 20
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(display_frame.shape[1], x2 + padding)
            y2 = min(display_frame.shape[0], y2 + padding)
            
            # 裁切人物區域
            person_crop = display_frame[y1:y2, x1:x2]
            
            if person_crop.size > 0:
                # 轉換顏色空間並進行姿勢偵測
                rgb_crop = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
                pose_results = pose.process(rgb_crop)
                
                if pose_results.pose_landmarks:
                    # 計算角度（使用原始相對座標）
                    angles = extract_pose_angles(pose_results.pose_landmarks.landmark)
                    
                    if angles:
                        person_data = {
                            'frame': frame_count,
                            'timestamp': frame_count / fps,
                            'track_id': track_id,
                            **angles, # 字典解包，這裡會展開 angles 字典的所有內容
                            'pose_detected': True
                        }
                        frame_people_data.append(person_data)
                        
                        # 視覺化：在裁切區域上繪製姿勢，然後貼回原圖
                        if SHOW_VISUAL or SAVE_SAMPLE_FRAMES:
                            # 在裁切的人物區域上繪製姿勢
                            person_crop_with_pose = person_crop.copy()
                            mp_drawing.draw_landmarks(
                                person_crop_with_pose,
                                pose_results.pose_landmarks,
                                mp_pose.POSE_CONNECTIONS,
                                landmark_drawing_spec = mp_drawing_styles.get_default_pose_landmarks_style()
                            )
                            
                            # 將繪製結果貼回原圖
                            display_frame[y1:y2, x1:x2] = person_crop_with_pose
                            
                            # 繪製bounding box和person ID
                            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(display_frame, f"Track ID: {track_id}", (x1, y1-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                            
                            # 顯示角度資訊（在bounding box內）
                            text_y = y1 + 30
                            cv2.putText(display_frame, f"L Arm: {angles['left_arm_angle']:.1f}°", 
                                (x1 + 10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
                            cv2.putText(display_frame, f"R Arm: {angles['right_arm_angle']:.1f}°", 
                                (x1 + 10, text_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
        
        # 將此幀的所有人物數據加入總數據
        all_people_data.extend(frame_people_data)
        
        # 顯示整體資訊
        if SHOW_VISUAL or SAVE_SAMPLE_FRAMES:
            cv2.putText(display_frame, f"Frame: {frame_count}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_frame, f"Time: {frame_count/fps:.1f}s", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_frame, f"People detected: {len(person_boxes)}", (10, 90), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 顯示即時視窗
        if SHOW_VISUAL:
            cv2.imshow('Multi-Person Pose Analysis', display_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('x'):
                break
            elif key == ord('s'):
                SHOW_VISUAL = not SHOW_VISUAL

cap.release()
cv2.destroyAllWindows()

# 保存數據
print(f"\n分析完成！共處理 {frame_count} 幀")
print(f"總共偵測到 {len(all_people_data)} 筆人物姿勢數據")

# 保存為CSV
df = pd.DataFrame(all_people_data)
csv_filename = f"multi_person_pose_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
df.to_csv(csv_filename, index=False, encoding='utf-8-sig')
print(f"CSV 文件已保存: {csv_filename}")

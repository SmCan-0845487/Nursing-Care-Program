import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import json
from datetime import datetime
from pathlib import Path

# mediapipe 初始化
mp_pose = mp.solutions.pose # mediapipe 姿勢偵測
mp_drawing = mp.solutions.drawing_utils # mediapipe 繪圖方法
mp_drawing_styles = mp.solutions.drawing_styles # mediapipe 繪圖樣式

# 影片路徑，分別是教練影片，以及真實老人運動情形之影片
coach_video_path = r"C:\Users\e6797\OneDrive\Desktop\VR虛擬教練\第一週\手臂伸展配合.mp4"
cap = cv2.VideoCapture(coach_video_path) # 將原本的鏡頭捕捉改成捕捉影片

# 設定視覺顯示參數
SHOW_VISUAL = True  # 是否顯示視覺化窗口
SAVE_SAMPLE_FRAMES = True  # 是否保存部分分析結果圖片
SAMPLE_INTERVAL = 960  # 每480幀保存一張圖片

# 存儲所有幀的姿勢數據
pose_data = []
frame_data = []

# 計算手臂彎曲角度的函數
def calculate_angle(p1, p2, p3):
    """計算三點形成的角度"""
    v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
    v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
    
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle = np.arccos(cos_angle)
    return np.degrees(angle)

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

# min_detection_confidence 小時容易檢測到人，但可能有誤判，大則相反(最大為1)
# min_tracking_confidence 當已經鎖定人物後，用來判斷是否繼續追蹤的信心值。數值小追蹤較不穩定，容易重新檢測，反之
# 穩定環境之坐姿運動(建議 0.7，0.5) 。光線不佳或多人環境(建議0.4，0.4)
with mp_pose.Pose(min_detection_confidence = 0.7 , min_tracking_confidence = 0.7) as pose:
    
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
        print(f"處理進度: {frame_count}/{total_frames} ({frame_count/total_frames*100:.1f}%)", end='\r')
        
        # 可以調整影像大小以提高處理速度，但大小會影響捕捉的精準度
        frame = frame[0:1080, 960:1600] # 右上角
        display_frame = cv2.resize(frame, (360, 640))
        
        # 轉換顏色空間(將 BGR 轉換成 RGB)
        rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)
        
        # 視覺化處理(根據姿勢偵測結果，標記身體節點和骨架)
        if SHOW_VISUAL or SAVE_SAMPLE_FRAMES:
            # 繪製姿勢標記，有偵測到的話
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                mp_drawing.draw_landmarks(
                    display_frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                )
                
                # 計算並顯示角度
                if len(landmarks) >= 33:  # MediaPipe Pose 有33個關鍵點
                    # 手臂角度（你已有的）
                    left_shoulder = [landmarks[11].x, landmarks[11].y]
                    left_elbow = [landmarks[13].x, landmarks[13].y]
                    left_wrist = [landmarks[15].x, landmarks[15].y]
                    
                    right_shoulder = [landmarks[12].x, landmarks[12].y]
                    right_elbow = [landmarks[14].x, landmarks[14].y]
                    right_wrist = [landmarks[16].x, landmarks[16].y]
                    
                    # 軀幹相關關鍵點
                    left_hip = [landmarks[23].x, landmarks[23].y]
                    right_hip = [landmarks[24].x, landmarks[24].y]
                    
                    # 頭部相關關鍵點
                    nose = [landmarks[0].x, landmarks[0].y]
                    left_ear = [landmarks[7].x, landmarks[7].y]
                    right_ear = [landmarks[8].x, landmarks[8].y]
                    
                    # 腿部相關關鍵點
                    left_knee = [landmarks[25].x, landmarks[25].y]
                    right_knee = [landmarks[26].x, landmarks[26].y]
                    left_ankle = [landmarks[27].x, landmarks[27].y]
                    right_ankle = [landmarks[28].x, landmarks[28].y]
                    
                    try:
                        left_arm_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
                        right_arm_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
                        trunk_angle = calculate_trunk_angle(left_shoulder, right_shoulder, left_hip, right_hip)
                        head_tilt_angle = calculate_head_angle(nose, left_ear, right_ear)

                        # 新增：肩部角度計算
                        left_shoulder_angle = calculate_shoulder_angle(left_shoulder, left_elbow, left_hip)
                        right_shoulder_angle = calculate_shoulder_angle(right_shoulder, right_elbow, right_hip)
                        
                        # 新增：膝部角度計算
                        left_knee_angle = calculate_knee_angle(left_hip, left_knee, left_ankle)
                        right_knee_angle = calculate_knee_angle(right_hip, right_knee, right_ankle)
                        
                        # 保存到數據列表
                        pose_data.append({
                            'frame': frame_count,
                            'timestamp': frame_count / fps,
                            'left_arm_angle': left_arm_angle,
                            'right_arm_angle': right_arm_angle,
                            'trunk_angle': trunk_angle,
                            'head_tilt_angle': head_tilt_angle,
                            'left_shoulder_angle': left_shoulder_angle,
                            'right_shoulder_angle': right_shoulder_angle,
                            'left_knee_angle': left_knee_angle,
                            'right_knee_angle': right_knee_angle,
                            'pose_detected': True
                        })
                    except Exception as e:
                        print(f"角度計算錯誤: {e}")
                        # 錯誤時保存0值
                        pose_data.append({
                            'frame': frame_count,
                            'timestamp': frame_count / fps,
                            'left_arm_angle': 0,
                            'right_arm_angle': 0,
                            'trunk_angle': 0,
                            'head_tilt_angle': 0,
                            'left_shoulder_angle': 0,
                            'right_shoulder_angle': 0,
                            'left_knee_angle': 0,
                            'right_knee_angle': 0,
                            'pose_detected': False
                        })
            else:# 關鍵點不足時
                pose_data.append({
                    'frame': frame_count,
                    'timestamp': frame_count / fps,
                    'left_arm_angle': 0,
                    'right_arm_angle': 0,
                    'trunk_angle': 0,
                    'head_tilt_angle': 0,
                    'left_shoulder_angle': 0,
                    'right_shoulder_angle': 0,
                    'left_knee_angle': 0,
                    'right_knee_angle': 0,
                    'pose_detected': False
                })
            
            # 保存樣本圖片
            if SAVE_SAMPLE_FRAMES and frame_count % SAMPLE_INTERVAL == 0:
                sample_filename = f"sample_frame_{frame_count:05d}.jpg"
                cv2.imwrite(sample_filename, display_frame)
        else:
            # 沒有偵測到姿勢時
            cv2.putText(display_frame, "No Pose Detected", (10, 150), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            pose_data.append({
                'frame': frame_count,
                'timestamp': frame_count / fps,
                'left_arm_angle': 0,
                'right_arm_angle': 0,
                'trunk_angle': 0,
                'head_tilt_angle': 0,
                'left_shoulder_angle': 0,
                'right_shoulder_angle': 0,
                'left_knee_angle': 0,
                'right_knee_angle': 0,
                'pose_detected': False
            })

        # 顯示即時視窗
        if SHOW_VISUAL:
            cv2.imshow('Pose Analysis', display_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('x'):  # 按 x 提前退出
                break
            elif key == ord('s'):  # 按 s 暫時關閉
                SHOW_VISUAL = not SHOW_VISUAL
        
        # 如果檢測到姿勢，保存詳細數據
        if results.pose_landmarks:
            landmarks = []
            for idx, landmark in enumerate(results.pose_landmarks.landmark):
                landmarks.append({
                    'landmark_id': idx,
                    'x': landmark.x,
                    'y': landmark.y,
                    'z': landmark.z,
                    'visibility': landmark.visibility
                })
            
            frame_data.append({
                'frame_number': frame_count,
                'timestamp': frame_count / fps,
                'landmarks': landmarks
            })
        
        # 計算手臂彎曲角度的函數
        def calculate_angle(p1, p2, p3):
            """計算三點形成的角度"""
            v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
            v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
            
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = np.arccos(cos_angle)
            return np.degrees(angle)

cap.release()
cv2.destroyAllWindows()  # 關閉所有 OpenCV 視窗

# 保存數據，指定儲存路徑
save_path = Path(r"C:\Users\e6797\OneDrive\Desktop\VR虛擬教練\第一週-分析")

# 保存為 CSV 格式（適合 Excel 分析）
df = pd.DataFrame(pose_data)
csv_filename = save_path / f"pose_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
df.to_csv(csv_filename, index=False, encoding='utf-8-sig')
print(f"CSV 文件已保存: {csv_filename}")

print("\n=== 基本分析結果 ===")
valid_poses = df[df['pose_detected'] == True]
print(f"成功檢測姿勢的幀數: {len(valid_poses)}/{len(df)} ({len(valid_poses)/len(df)*100:.1f}%)")
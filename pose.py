import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# تعريف المتغيرات خارج الحلقة
counter = 0
stage = None

# فتح الكاميرا
cap = cv2.VideoCapture(0)

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        
        # تحويل الصورة إلى RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        
        # معالجة الصورة باستخدام Mediapipe
        results = pose.process(image)
        
        # تحويل الصورة إلى BGR للعرض باستخدام OpenCV
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # حساب الزوايا
        def calculate_angle(a, b, c):
            a = np.array(a)
            b = np.array(b)
            c = np.array(c)
            
            radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
            angle = np.abs(radians * 180.0 / np.pi)
            
            if angle > 180.0:
                angle = 360 - angle
            
            return angle
        
        try:
            landmarks = results.pose_landmarks.landmark
            
            # استخراج إحداثيات المفاصل
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            
            # حساب زاوية الكوع
            angle = calculate_angle(shoulder, elbow, wrist)
            
            # عرض الزاوية على الصورة
            cv2.putText(image, str(angle), tuple(np.multiply(elbow, [640, 480]).astype(int)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            
            # منطق العد
            if angle > 130:
                stage = "down"
            if angle < 60 and stage == "down":
                stage = 'up'
                counter += 1
                print(counter)
        
        except:
            pass
        
        # عرض العد والمرحلة
        cv2.rectangle(image, (0, 0), (235, 73), (245, 117, 16), -1)
        
        cv2.putText(image, 'REPS', (15, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, str(counter), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1, cv2.LINE_AA)
        
        cv2.putText(image, 'STAGE', (75, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, stage, (80, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)
        
        # رسم نقاط الهيكل العظمي
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(145, 117, 66), thickness=4),
                                  mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=4))
        
        # عرض الصورة
        cv2.imshow('Mediapipe Feed', image)
        
        # إنهاء العرض عند الضغط على 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# تحرير الكاميرا وإغلاق النوافذ
cap.release()
cv2.destroyAllWindows()

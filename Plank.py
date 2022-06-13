import pyttsx3

engine = pyttsx3.init("sapi5")
voice = engine.getProperty("voices")
engine.setProperty('voice',voice[0].id)

def speak(msg):
    engine.say(msg)
    engine.runAndWait()

import cv2
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
cap = cv2.VideoCapture(0)

def calculate_angle(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle >180.0:
        angle = 360-angle
        
    return angle 

# Plank stage variables
stage = None
flag = 0

## Setup mediapipe instance
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        
        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
      
        # Make detection
        results = pose.process(image)
    
        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Extract landmarks
        try:
            landmarks = results.pose_landmarks.landmark
            
            # Get coordinates
            shoulder = [landmarks[11].x,landmarks[11].y]
            elbow = [landmarks[13].x,landmarks[13].y]
            wrist = [landmarks[15].x,landmarks[15].y]
            hip = [landmarks[23].x,landmarks[23].y]
            knee = [landmarks[25].x,landmarks[25].y]
            
            # Calculate angle
            angle1 = calculate_angle(shoulder, elbow, wrist)
            angle2 = calculate_angle(shoulder, hip, knee)
            
            # Visualize angle
            cv2.putText(image, str(angle1), 
                           tuple(np.multiply(elbow, [640, 480]).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                )
            cv2.putText(image, str(angle2), 
                           tuple(np.multiply(hip, [640, 480]).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                )
            
            # Plank pose logic
            if angle1>80 and angle1<100 and angle2>170 and angle2<190:
                stage = "perfect"
            else:
                stage = "wrong"
                
            if flag == 0 and stage == "perfect" :
                speak("perfect")
                flag = 1
            elif flag == 1 and stage == "wrong" :
                speak("wrong")
                flag = 0
        except:
            pass
        
        # Render plank pose detector
        # Setup status box
        cv2.rectangle(image, (0,0), (225,73), (245,117,16), -1)
        

        
        # Stage data
        cv2.putText(image, 'STAGE', (65,12), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(image, stage, 
                    (60,60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
        
        
        # Render detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                 )               
        
        cv2.imshow('Mediapipe Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    
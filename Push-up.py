import cv2

import mediapipe as mp

import os

mp_drawing = mp.solutions.drawing_utils

mp_pose = mp.solutions.pose

counter = 0

stage = None

create = None

opname = "output.avi"

def findPosition(image, draw=True):

  lmList = []

  if results.pose_landmarks:

      mp_drawing.draw_landmarks(

         image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

      for id, lm in enumerate(results.pose_landmarks.landmark):

          h, w, c = image.shape

          cx, cy = int(lm.x * w), int(lm.y * h)

          lmList.append([id, cx, cy])

          #cv2.circle(image, (cx, cy), 5, (255, 0, 0), cv2.FILLED)

  return lmList

cap = cv2.VideoCapture(0)

with mp_pose.Pose(

    min_detection_confidence=0.7,

    min_tracking_confidence=0.7) as pose:

  while cap.isOpened():

    success, image = cap.read()

    image = cv2.resize(image, (640,480))

    if not success:

      print("Ignoring empty camera frame.")

      # If loading a video, use 'break' instead of 'continue'.

      continue

    # Flip the image horizontally for a later selfie-view display, and convert

    # the BGR image to RGB.

    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

    # To improve performance, optionally mark the image as not writeable to

    # pass by reference.

    results = pose.process(image)

    # Draw the pose annotation on the image.

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    lmList = findPosition(image, draw=True)

    if len(lmList) != 0:

      cv2.circle(image, (lmList[12][1], lmList[12][2]), 20, (0, 0, 255), cv2.FILLED)

      cv2.circle(image, (lmList[11][1], lmList[11][2]), 20, (0, 0, 255), cv2.FILLED)

      cv2.circle(image, (lmList[12][1], lmList[12][2]), 20, (0, 0, 255), cv2.FILLED)

      cv2.circle(image, (lmList[11][1], lmList[11][2]), 20, (0, 0, 255), cv2.FILLED)

      if (lmList[12][2] and lmList[11][2] >= lmList[14][2] and lmList[13][2]):

        cv2.circle(image, (lmList[12][1], lmList[12][2]), 20, (0, 255, 0), cv2.FILLED)

        cv2.circle(image, (lmList[11][1], lmList[11][2]), 20, (0, 255, 0), cv2.FILLED)

        stage = "down"

      if (lmList[12][2] and lmList[11][2] <= lmList[14][2] and lmList[13][2]) and stage == "down":

        stage = "up"

        counter += 1

        counter2 = str(int(counter))

        print(counter)

        os.system("echo '" + counter2 + "' | festival --tts")

    text = "{}:{}".format("Push Ups", counter)

    cv2.putText(image, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX,

                1, (255, 0, 0), 2)

    cv2.imshow('MediaPipe Pose', image)

    if create is None:

      fourcc = cv2.VideoWriter_fourcc(*'XVID')

      create = cv2.VideoWriter(opname, fourcc, 30, (image.shape[1], image.shape[0]), True)

    create.write(image)

    key = cv2.waitKey(1) & 0xFF

    # if the q key was pressed, break from the loop

    if key == ord("q"):

      break

    # do a bit of cleanup

cv2.destroyAllWindows()

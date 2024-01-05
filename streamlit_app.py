from streamlit_webrtc import webrtc_streamer, RTCConfiguration
import av
import streamlit as st
import mediapipe as mp
import cv2
import time
import math
import numpy as np

from sample_utils.download import download_file
from sample_utils.turn import get_ice_servers

# # face bounder indices
# FACE_OVAL=[ 10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103,67, 109]
# variables
frame_counter = 0
CEF_COUNTER = 0
TOTAL_BLINKS = 0
# constants
CLOSED_EYES_FRAME = 3
FONTS = cv2.FONT_HERSHEY_COMPLEX

# face bounder indices
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378,
             400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]

# lips indices for Landmarks
LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178,
        88, 95, 185, 40, 39, 37, 0, 267, 269, 270, 409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78]
LOWER_LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321,
              375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
UPPER_LIPS = [185, 40, 39, 37, 0, 267, 269, 270,
              409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78]

# Left eyes indices
LEFT_EYE = [362, 382, 381, 380, 374, 373, 390,
            249, 263, 466, 388, 387, 386, 385, 384, 398]
LEFT_EYEBROW = [336, 296, 334, 293, 300, 276, 283, 282, 295, 285]

# right eyes indices
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154,
             155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYEBROW = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]

UPPER_EYE_LEFT = [246, 161, 160, 159, 158, 157, 173, 133]
UPPER_EYE_RIGHT = [7, 33, 161, 160, 159, 158, 157, 173]

counter = [0, 0, 0, 0, 0, 0, 0, 0]

full_counter = 0


def landmarksDetection(img, results, draw=False):
    img_height, img_width = img.shape[:2]
    # list[(x,y), (x,y)....]
    mesh_coord = [(int(point.x * img_width), int(point.y * img_height))
                  for point in results.face_landmarks.landmark]
    if draw:
        [cv2.circle(img, p, 2, (0, 255, 0), -1) for p in mesh_coord]

    # returning the list of tuples for each landmarks
    return mesh_coord

# Euclaidean distance


def euclaideanDistance(point, point1):
    x, y = point
    x1, y1 = point1
    distance = math.sqrt((x1 - x)**2 + (y1 - y)**2)
    return distance

# Blinking Ratio


def blinkRatio(img, landmarks, right_indices, left_indices):

    rh_right = landmarks[right_indices[0]]
    rh_left = landmarks[right_indices[8]]
    # vertical line
    rv_top = landmarks[right_indices[12]]
    rv_bottom = landmarks[right_indices[4]]

    lh_right = landmarks[left_indices[0]]
    lh_left = landmarks[left_indices[8]]

    # vertical line
    lv_top = landmarks[left_indices[12]]
    lv_bottom = landmarks[left_indices[4]]

    rhDistance = euclaideanDistance(rh_right, rh_left)
    rvDistance = euclaideanDistance(rv_top, rv_bottom)

    lvDistance = euclaideanDistance(lv_top, lv_bottom)
    lhDistance = euclaideanDistance(lh_right, lh_left)

    reRatio = rhDistance/rvDistance
    leRatio = lhDistance/lvDistance

    ratio = (reRatio+leRatio)/2
    return ratio

# Eyes Extrctor function,


def eyesExtractor(img, right_eye_coords, left_eye_coords):
    # converting color image to  scale image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # getting the dimension of image
    dim = gray.shape

    # creating mask from gray scale dim
    mask = np.zeros(dim, dtype=np.uint8)

    # drawing Eyes Shape on mask with white color
    cv2.fillPoly(mask, [np.array(right_eye_coords, dtype=np.int32)], 255)
    cv2.fillPoly(mask, [np.array(left_eye_coords, dtype=np.int32)], 255)

    # showing the mask
    # cv.imshow('mask', mask)

    # draw eyes image on mask, where white shape is
    eyes = cv2.bitwise_and(gray, gray, mask=mask)
    # change black color to gray other than eys
    # cv.imshow('eyes draw', eyes)
    eyes[mask == 0] = 155

    # getting minium and maximum x and y  for right and left eyes
    # For Right Eye
    r_max_x = (max(right_eye_coords, key=lambda item: item[0]))[0]
    r_min_x = (min(right_eye_coords, key=lambda item: item[0]))[0]
    r_max_y = (max(right_eye_coords, key=lambda item: item[1]))[1]
    r_min_y = (min(right_eye_coords, key=lambda item: item[1]))[1]

    # For LEFT Eye
    l_max_x = (max(left_eye_coords, key=lambda item: item[0]))[0]
    l_min_x = (min(left_eye_coords, key=lambda item: item[0]))[0]
    l_max_y = (max(left_eye_coords, key=lambda item: item[1]))[1]
    l_min_y = (min(left_eye_coords, key=lambda item: item[1]))[1]

    # croping the eyes from mask
    cropped_right = eyes[r_min_y: r_max_y, r_min_x: r_max_x]
    cropped_left = eyes[l_min_y: l_max_y, l_min_x: l_max_x]

    # returning the cropped eyes
    return cropped_right, cropped_left

# Eyes Postion Estimator


def positionEstimator(cropped_eye):
    # getting height and width of eye
    h, w = cropped_eye.shape

    # remove the noise from images
    gaussain_blur = cv2.GaussianBlur(cropped_eye, (9, 9), 0)
    median_blur = cv2.medianBlur(gaussain_blur, 3)

    # applying thrsholding to convert binary_image
    ret, threshed_eye = cv2.threshold(median_blur, 130, 255, cv2.THRESH_BINARY)

    # create fixd part for eye with
    piece = int(w/3)

    # slicing the eyes into three parts
    right_piece = threshed_eye[0:h, 0:piece]
    center_piece = threshed_eye[0:h, piece: piece+piece]
    left_piece = threshed_eye[0:h, piece + piece:w]

    # calling pixel counter function
    eye_position = pixelCounter(right_piece, center_piece, left_piece)

    return eye_position

# creating pixel counter function


def pixelCounter(first_piece, second_piece, third_piece):
    # counting black pixel in each part
    right_part = np.sum(first_piece == 0)
    center_part = np.sum(second_piece == 0)
    left_part = np.sum(third_piece == 0)
    # creating list of these values
    eye_parts = [right_part, center_part, left_part]

    # getting the index of max values in the list
    max_index = eye_parts.index(max(eye_parts))
    pos_eye = ''
    if max_index == 0:
        pos_eye = "RIGHT"
    elif max_index == 1:
        pos_eye = 'CENTER'
    elif max_index == 2:
        pos_eye = 'LEFT'
    else:
        pos_eye = "Closed"
    return pos_eye

# Function to check if eyes are looking up


def is_eyes_looking_up(landmarks, upper_eye_indices):
    upper_eye_points = [landmarks[idx] for idx in upper_eye_indices]
    average_y = sum(point[1]
                    for point in upper_eye_points) / len(upper_eye_points)
    return average_y < landmarks[LEFT_EYE[0]][1] and average_y < landmarks[RIGHT_EYE[0]][1]


our_time = 0

start_time = time.time()

nervous = False

mp_drawing = mp.solutions.drawing_utils  # Drawing helpers
mp_holistic = mp.solutions.holistic  # Mediapipe Solutions

holistic = mp_holistic.Holistic(
    min_detection_confidence=0.5, min_tracking_confidence=0.5)


class VideoProcessor:
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        global full_counter
        global TOTAL_BLINKS
        global CEF_COUNTER
        global CLOSED_EYES_FRAME

        frame = frame.to_ndarray(format="bgr24")

        full_counter += 1

        # Recolor Feed
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Make Detections
        results = holistic.process(image)

        # Recolor image back to BGR for rendering
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        img_h, img_w, img_c = image.shape
        face_3d = []
        face_2d = []

        # 1. Draw face landmarks
        mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
                                  mp_drawing.DrawingSpec(
                                      color=(80, 110, 10), thickness=1, circle_radius=1),
                                  mp_drawing.DrawingSpec(
                                      color=(80, 256, 121), thickness=1, circle_radius=1)
                                  )

        # 2. Right hand
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                  mp_drawing.DrawingSpec(
                                      color=(80, 22, 10), thickness=2, circle_radius=4),
                                  mp_drawing.DrawingSpec(
                                      color=(80, 44, 121), thickness=2, circle_radius=2)
                                  )

        # 3. Left Hand
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                  mp_drawing.DrawingSpec(
                                      color=(121, 22, 76), thickness=2, circle_radius=4),
                                  mp_drawing.DrawingSpec(
                                      color=(121, 44, 250), thickness=2, circle_radius=2)
                                  )

        # 4. Pose Detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(
                                      color=(245, 117, 66), thickness=2, circle_radius=4),
                                  mp_drawing.DrawingSpec(
                                      color=(245, 66, 230), thickness=2, circle_radius=2)
                                  )

        if (results.right_hand_landmarks and results.left_hand_landmarks and results.face_landmarks and results.pose_landmarks):
            right_hand_landmarks = results.right_hand_landmarks.landmark
            left_hand_landmarks = results.left_hand_landmarks.landmark
            pose_landmarks = results.pose_landmarks.landmark
            face_landmarks = results.face_landmarks.landmark

            mesh_coords = landmarksDetection(frame, results, False)

            bottom_lip = face_landmarks[18]
            nose_center = face_landmarks[5]

            right_wrist = pose_landmarks[16]

            lips_left = face_landmarks[287]
            lips_right = face_landmarks[57]

            right_ear_top = face_landmarks[21]
            right_ear_bottom = face_landmarks[215]

            left_ear_top = face_landmarks[389]
            left_ear_bottom = face_landmarks[361]

            forehead = face_landmarks[151]

            bottom_lip_y = int(bottom_lip.y * frame.shape[0])
            upper_nose_y = int(nose_center.y * frame.shape[0])

            lips_left_x = int(lips_left.x * frame.shape[1])
            lips_right_x = int(lips_right.x * frame.shape[1])
            lips_left_y = int(lips_left.y * frame.shape[0])
            lips_right_y = int(lips_right.y * frame.shape[0])

            right_ear_top_y = int(right_ear_top.y * frame.shape[0])
            right_ear_top_x = int(right_ear_top.x * frame.shape[1])
            right_ear_bottom_x = int(right_ear_bottom.x * frame.shape[1])
            right_ear_bottom_y = int(right_ear_bottom.y * frame.shape[0])

            left_ear_top_y = int(left_ear_top.y * frame.shape[0])
            left_ear_top_x = int(left_ear_top.x * frame.shape[1])
            left_ear_bottom_y = int(left_ear_bottom.y * frame.shape[0])
            left_ear_bottom_x = int(left_ear_bottom.x * frame.shape[1])

            right_hand_tip_x = int(
                right_hand_landmarks[mp_holistic.HandLandmark.INDEX_FINGER_TIP].x * frame.shape[1])
            left_hand_tip_x = int(
                left_hand_landmarks[mp_holistic.HandLandmark.INDEX_FINGER_TIP].x * frame.shape[1])
            right_hand_tip_y = int(
                right_hand_landmarks[mp_holistic.HandLandmark.INDEX_FINGER_TIP].y * frame.shape[0])
            left_hand_tip_y = int(
                left_hand_landmarks[mp_holistic.HandLandmark.INDEX_FINGER_TIP].y * frame.shape[0])
            right_hand_dip_y = int(
                right_hand_landmarks[mp_holistic.HandLandmark.INDEX_FINGER_DIP].y * frame.shape[0])
            left_hand_dip_y = int(
                left_hand_landmarks[mp_holistic.HandLandmark.INDEX_FINGER_DIP].y * frame.shape[0])
            right_hand_thumb = int(
                right_hand_landmarks[mp_holistic.HandLandmark.THUMB_TIP].y * frame.shape[0])
            left_hand_thumb = int(
                left_hand_landmarks[mp_holistic.HandLandmark.THUMB_TIP].y * frame.shape[0])
            right_shoulder = pose_landmarks[12]
            left_shoulder = pose_landmarks[11]
            mouth_left = pose_landmarks[9]
            mouth_right = pose_landmarks[10]

            right_hand_middle_finger_y = int(
                right_hand_landmarks[mp_holistic.HandLandmark.MIDDLE_FINGER_TIP].y * frame.shape[0])
            right_hand_middle_finger_x = int(
                right_hand_landmarks[mp_holistic.HandLandmark.MIDDLE_FINGER_TIP].x * frame.shape[1])
            right_hand_ring_finger_y = int(
                right_hand_landmarks[mp_holistic.HandLandmark.RING_FINGER_TIP].y * frame.shape[0])
            right_hand_ring_finger_x = int(
                right_hand_landmarks[mp_holistic.HandLandmark.RING_FINGER_TIP].x * frame.shape[1])
            right_hand_pinky_finger_y = int(
                right_hand_landmarks[mp_holistic.HandLandmark.PINKY_TIP].y * frame.shape[0])
            right_hand_pinky_finger_x = int(
                right_hand_landmarks[mp_holistic.HandLandmark.PINKY_TIP].x * frame.shape[1])

            left_hand_middle_finger_y = int(
                left_hand_landmarks[mp_holistic.HandLandmark.MIDDLE_FINGER_TIP].y * frame.shape[0])
            left_hand_middle_finger_x = int(
                left_hand_landmarks[mp_holistic.HandLandmark.MIDDLE_FINGER_TIP].x * frame.shape[1])
            left_hand_ring_finger_y = int(
                left_hand_landmarks[mp_holistic.HandLandmark.RING_FINGER_TIP].y * frame.shape[0])
            left_hand_ring_finger_x = int(
                left_hand_landmarks[mp_holistic.HandLandmark.RING_FINGER_TIP].x * frame.shape[1])
            left_hand_pinky_finger_y = int(
                left_hand_landmarks[mp_holistic.HandLandmark.PINKY_TIP].y * frame.shape[0])
            left_hand_pinky_finger_x = int(
                left_hand_landmarks[mp_holistic.HandLandmark.PINKY_TIP].x * frame.shape[1])

            forehead_y = int(forehead.y * frame.shape[0])

            right_hand_middle_finger_dip_y = int(
                right_hand_landmarks[11].y * frame.shape[0])
            right_hand_ring_finger_dip_y = int(
                right_hand_landmarks[15].y * frame.shape[0])
            right_hand_pinky_dip_y = int(
                right_hand_landmarks[19].y * frame.shape[0])

            left_hand_middle_finger_dip_y = int(
                right_hand_landmarks[11].y * frame.shape[0])
            left_hand_ring_finger_dip_y = int(
                right_hand_landmarks[15].y * frame.shape[0])
            left_hand_pinky_dip_y = int(
                left_hand_landmarks[19].y * frame.shape[0])

            right_index_middle_finger_distance = right_hand_middle_finger_x - right_hand_tip_x
            right_middle_ring_finger_distance = right_hand_ring_finger_x - \
                right_hand_middle_finger_x

            left_index_middle_finger_distance = left_hand_middle_finger_x - left_hand_tip_x
            right_middle_ring_finger_distance = right_hand_ring_finger_x - \
                right_hand_middle_finger_x

            # eye_pupil_right_x = int(face_landmarks[468].x * frame.shape[1])
            # eye_pupil_left_x = int(face_landmarks[473].x * frame.shape[1])

            ring_finger_distance = right_hand_ring_finger_y - left_hand_ring_finger_y

            mouth_left_y = int(mouth_left.y * frame.shape[0])
            mouth_right_y = int(mouth_right.y * frame.shape[0])
            right_shoulder_y = int(right_shoulder.y * frame.shape[0])
            left_shoulder_y = int(left_shoulder.y * frame.shape[0])

            val_hand_tips = left_hand_tip_x - right_hand_tip_x
            val_hand_thumbs = left_hand_thumb - right_hand_thumb

            lips_left = face_landmarks[287]
            lips_right = face_landmarks[57]

            lips_left_x = int(lips_left.x * frame.shape[1])
            lips_right_x = int(lips_right.x * frame.shape[1])

            face_landmarks = results.face_landmarks.landmark

            thumb_tip = right_hand_landmarks[mp_holistic.HandLandmark.THUMB_TIP]
            index_finger_tip = right_hand_landmarks[mp_holistic.HandLandmark.INDEX_FINGER_TIP]
            right_cheek = face_landmarks[116]
            right_ear = face_landmarks[147]
            chin = face_landmarks[152]
            left_cheek = face_landmarks[323]

            thumb_tip_y = int(thumb_tip.y * frame.shape[0])
            thumb_tip_x = int(thumb_tip.x * frame.shape[1])
            index_finger_tip_y = int(index_finger_tip.y * frame.shape[0])
            index_finger_tip_x = int(index_finger_tip.x * frame.shape[1])
            cheek_y = int(right_cheek.y * frame.shape[0])
            right_ear_x = int(right_ear.x * frame.shape[1])
            chin_y = int(chin.y * frame.shape[0])
            left_cheek_x = int(left_cheek.x * frame.shape[1])

            lips_left_y_distance = lips_left_y - lips_right_y
            lips_right_y_distance = lips_right_y - lips_left_y

            index_finger_distance_x = left_hand_tip_x - right_hand_tip_x
            if ((right_hand_tip_y < right_hand_middle_finger_y) and (right_hand_tip_y < right_hand_ring_finger_y) and (right_hand_tip_y < right_hand_pinky_finger_y) and (right_hand_tip_y < thumb_tip_y)):
                if ((right_hand_tip_y > upper_nose_y) and (right_hand_tip_y < bottom_lip_y) and (right_hand_tip_x > lips_left_x) and (right_hand_tip_x < lips_right_x)):
                    cv2.putText(image, "Disagree with Spoken Word", (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                    counter[0] += 1
            elif ((right_hand_tip_y > right_hand_dip_y) and (right_hand_middle_finger_y > right_hand_middle_finger_dip_y) and (right_hand_ring_finger_y > right_hand_ring_finger_dip_y) and (right_hand_pinky_finger_y > right_hand_pinky_dip_y) and (left_hand_tip_y > left_hand_dip_y) and (left_hand_middle_finger_y > left_hand_middle_finger_dip_y) and (left_hand_ring_finger_y > left_hand_ring_finger_dip_y) and (left_hand_pinky_finger_y > left_hand_pinky_dip_y) and (right_hand_tip_x > (right_ear_top_x-40)) and (right_hand_tip_x < right_ear_top_x) and (left_hand_tip_x < (left_ear_top_x+40)) and (left_hand_tip_x > left_ear_top_x) and (lips_left_y > lips_right_y or lips_right_y > lips_left_y) and ((lips_left_y_distance > 1 and lips_left_y_distance < 40) or (lips_right_y_distance > 1 and lips_right_y_distance < 40))):
                cv2.putText(image, "Annoyed", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                counter[2] += 1
            elif ((right_hand_tip_y < right_shoulder_y and left_hand_tip_y < left_shoulder_y) and (right_hand_tip_y > mouth_right_y) and (left_hand_tip_y > mouth_left_y) and (right_hand_dip_y > right_hand_tip_y and left_hand_dip_y > left_hand_tip_y) and (left_hand_tip_x < lips_left_x) and (right_hand_tip_x > lips_right_x) and (val_hand_tips < 20 and val_hand_thumbs < 20)):
                cv2.putText(image, "Wants her knowledge to be recognized now",
                            (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            elif (((right_hand_tip_y < right_hand_dip_y) and (right_hand_middle_finger_y < right_hand_middle_finger_dip_y) and (right_hand_ring_finger_y < right_hand_ring_finger_dip_y) and (right_hand_tip_y < right_hand_dip_y)) and ((left_hand_tip_y < left_hand_dip_y) and (left_hand_middle_finger_y < left_hand_middle_finger_dip_y) and (left_hand_ring_finger_y < left_hand_ring_finger_dip_y) and (left_hand_pinky_finger_y < left_hand_pinky_dip_y)) and (is_eyes_looking_up(mesh_coords, UPPER_EYE_LEFT + UPPER_EYE_RIGHT)) and (right_hand_tip_y < chin_y and right_hand_tip_y > forehead_y) and (left_hand_tip_y < chin_y and left_hand_tip_y > forehead_y) and (right_hand_tip_x > right_ear_top_x and left_hand_tip_x < left_ear_top_x) and (index_finger_distance_x > 20)):
                cv2.putText(image, "Reobserve, Feeling Uncomfortable", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                counter[1] += 1
            else:
                cv2.putText(image, "Neutral", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                counter[7] += 1

        elif results.face_landmarks and results.right_hand_landmarks and results.pose_landmarks:
            face_landmarks = results.face_landmarks.landmark
            right_hand_landmarks = results.right_hand_landmarks.landmark
            pose_landmarks = results.pose_landmarks.landmark

            for idx, lm in enumerate(face_landmarks):
                if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                    if idx == 1:
                        nose_2d = (lm.x * img_w, lm.y * img_h)
                        nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)

                    x, y = int(lm.x * img_w), int(lm.y * img_h)

                    # Get the 2D Coordinates
                    face_2d.append([x, y])

                    # Get the 3D Coordinates
                    face_3d.append([x, y, lm.z])

            # Convert it to the NumPy array
            face_2d = np.array(face_2d, dtype=np.float64)

            # Convert it to the NumPy array
            face_3d = np.array(face_3d, dtype=np.float64)

            # The camera matrix
            focal_length = 1 * img_w

            cam_matrix = np.array([[focal_length, 0, img_h / 2],
                                   [0, focal_length, img_w / 2],
                                   [0, 0, 1]])

            # The distortion parameters
            dist_matrix = np.zeros((4, 1), dtype=np.float64)

            # Solve PnP
            success, rot_vec, trans_vec = cv2.solvePnP(
                face_3d, face_2d, cam_matrix, dist_matrix)

            # Get rotational matrix
            rmat, jac = cv2.Rodrigues(rot_vec)

            # Get angles
            angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

            # Get the y rotation degree
            x = angles[0] * 360
            y = angles[1] * 360
            z = angles[2] * 360

            head_text = ""

            # See where the user's head tilting
            if y < -10:
                head_text = "Looking Left"
            elif y > 10:
                head_text = "Looking Right"
            elif x < -10:
                head_text = "Looking Down"
            elif x > 10:
                head_text = "Looking Up"
            else:
                head_text = "Forward"

            # Display the nose direction
            nose_3d_projection, jacobian = cv2.projectPoints(
                nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)

            p1 = (int(nose_2d[0]), int(nose_2d[1]))
            p2 = (int(nose_2d[0] + y * 10), int(nose_2d[1] - x * 10))

            # cv2.line(image, p1, p2, (255, 0, 0), 3)

            # Add the text on the image
            # cv2.putText(image, head_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
            # cv2.putText(image, "x: " + str(np.round(x,2)), (500, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            # cv2.putText(image, "y: " + str(np.round(y,2)), (500, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            # cv2.putText(image, "z: " + str(np.round(z,2)), (500, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            mesh_coords = landmarksDetection(frame, results, False)

            thumb_tip = right_hand_landmarks[mp_holistic.HandLandmark.THUMB_TIP]
            index_finger_tip = right_hand_landmarks[mp_holistic.HandLandmark.INDEX_FINGER_TIP]
            right_cheek = face_landmarks[116]
            right_ear = face_landmarks[147]
            chin = face_landmarks[152]
            left_cheek = face_landmarks[323]

            thumb_tip_y = int(thumb_tip.y * frame.shape[0])
            thumb_tip_x = int(thumb_tip.x * frame.shape[1])
            index_finger_tip_y = int(index_finger_tip.y * frame.shape[0])
            index_finger_tip_x = int(index_finger_tip.x * frame.shape[1])
            cheek_y = int(right_cheek.y * frame.shape[0])
            right_ear_x = int(right_ear.x * frame.shape[1])
            chin_y = int(chin.y * frame.shape[0])
            left_cheek_x = int(left_cheek.x * frame.shape[1])

            right_wrist = pose_landmarks[16]

            right_wrist_y = int(right_wrist.y * frame.shape[0])

            head_top = face_landmarks[10]
            head_below = face_landmarks[152]

            nose_top = face_landmarks[197]
            nose_bottom = face_landmarks[4]
            nose_left = face_landmarks[49]
            nose_right = face_landmarks[279]

            nose_top_y = int(nose_top.y * frame.shape[0])
            nose_bottom_y = int(nose_bottom.y * frame.shape[0])
            nose_left_x = int(nose_left.x * frame.shape[1])
            nose_right_x = int(nose_right.x * frame.shape[1])

            head_top_y = int(head_top.y * frame.shape[0])
            head_below_y = int(head_below.y * frame.shape[0])

            mouth_lip_upper = face_landmarks[13]
            mouth_lip_lower = face_landmarks[14]

            bottom_lip = face_landmarks[18]
            nose_center = face_landmarks[1]

            right_ear_top = face_landmarks[21]
            right_ear_bottom = face_landmarks[215]

            lips_left = face_landmarks[287]
            lips_right = face_landmarks[57]

            bottom_lip_y = int(bottom_lip.y * frame.shape[0])
            upper_nose_y = int(nose_center.y * frame.shape[0])

            lips_left_x = int(lips_left.x * frame.shape[1])
            lips_right_x = int(lips_right.x * frame.shape[1])

            left_cheek = face_landmarks[323]

            right_ear_top_y = int(right_ear_top.y * frame.shape[0])
            right_ear_top_x = int(right_ear_top.x * frame.shape[1])
            right_ear_bottom_x = int(right_ear_bottom.x * frame.shape[1])
            right_ear_bottom_y = int(right_ear_bottom.y * frame.shape[0])

            mouth_lip_upper_y = int(mouth_lip_upper.y * frame.shape[0])
            mouth_lip_lower_y = int(mouth_lip_lower.y * frame.shape[0])

            right_hand_tip_x = int(
                right_hand_landmarks[mp_holistic.HandLandmark.INDEX_FINGER_TIP].x * frame.shape[1])
            right_hand_tip_y = int(
                right_hand_landmarks[mp_holistic.HandLandmark.INDEX_FINGER_TIP].y * frame.shape[0])

            thumb_tip = right_hand_landmarks[mp_holistic.HandLandmark.THUMB_TIP]

            head_top_x = int(head_top.x * frame.shape[1])
            right_wrist_x = int(right_wrist.x * frame.shape[1])

            thumb_tip_y = int(thumb_tip.y * frame.shape[0])
            thumb_tip_x = int(thumb_tip.x * frame.shape[1])
            chin = face_landmarks[152]
            chin_y = int(chin.y * frame.shape[0])

            lips_distance = mouth_lip_lower_y - mouth_lip_upper_y

            right_ear_hand_distance = head_top_x - right_wrist_x

            right_hand_middle_finger_y = int(
                right_hand_landmarks[mp_holistic.HandLandmark.MIDDLE_FINGER_TIP].y * frame.shape[0])
            right_hand_middle_finger_x = int(
                right_hand_landmarks[mp_holistic.HandLandmark.MIDDLE_FINGER_TIP].x * frame.shape[1])
            right_hand_ring_finger_y = int(
                right_hand_landmarks[mp_holistic.HandLandmark.RING_FINGER_TIP].y * frame.shape[0])
            right_hand_ring_finger_x = int(
                right_hand_landmarks[mp_holistic.HandLandmark.RING_FINGER_TIP].x * frame.shape[1])
            right_hand_pinky_finger_y = int(
                right_hand_landmarks[mp_holistic.HandLandmark.PINKY_TIP].y * frame.shape[0])
            right_hand_pinky_finger_x = int(
                right_hand_landmarks[mp_holistic.HandLandmark.PINKY_TIP].x * frame.shape[1])

            left_cheek_x = int(left_cheek.x * frame.shape[1])

            if ((right_hand_tip_y < right_hand_middle_finger_y) and (right_hand_tip_y < right_hand_ring_finger_y) and (right_hand_tip_y < right_hand_pinky_finger_y) and (right_hand_tip_y < thumb_tip_y) and (right_hand_tip_y > upper_nose_y) and (right_hand_tip_y < bottom_lip_y) and (right_hand_tip_x < lips_left_x) and (right_hand_tip_x > lips_right_x)):
                cv2.putText(image, "Disagree with Spoken Word", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                counter[0] += 1
            elif ((right_hand_tip_y > right_ear_top_y) and (right_hand_tip_y < right_ear_bottom_y) and (right_hand_tip_x > (right_ear_top_x-40)) and (right_hand_tip_x < right_ear_top_x)):
                cv2.putText(image, "Disagree what was heard", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                counter[3] += 1
            elif (lips_distance > 10):
                cv2.putText(image, "Disbelief", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                counter[4] += 1
            elif ((right_hand_tip_y < nose_bottom_y) and (right_hand_tip_y > nose_top_y) and (right_hand_tip_x > nose_left_x) and (right_hand_tip_x < nose_right_x)):
                cv2.putText(image, "Untruthful", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                counter[5] += 1
            elif ((right_wrist_y > head_top_y) and (right_wrist_y < head_below_y) and (right_ear_hand_distance < 100) and (head_text == "Looking Down")):
                cv2.putText(image, "Embarrased, Got Caught", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                counter[6] += 1
            elif ((chin_y > thumb_tip_y) and (right_ear_x < thumb_tip_x) and (thumb_tip_y > cheek_y) and (thumb_tip_x < left_cheek_x) and (head_text == "Looking Up")) or ((chin_y > index_finger_tip_y) and (right_ear_x < index_finger_tip_x) and (index_finger_tip_y > cheek_y) and (index_finger_tip_x < left_cheek_x) and (head_text == "Looking Up")):
                cv2.putText(image, "Positive evaluation low risk situation", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            else:
                cv2.putText(image, "Neutral", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                counter[7] += 1

        elif results.pose_landmarks and results.face_landmarks:
            face_landmarks = results.face_landmarks.landmark
            pose_landmarks = results.pose_landmarks.landmark

            for idx, lm in enumerate(face_landmarks):
                if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                    if idx == 1:
                        nose_2d = (lm.x * img_w, lm.y * img_h)
                        nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)

                    x, y = int(lm.x * img_w), int(lm.y * img_h)

                    # Get the 2D Coordinates
                    face_2d.append([x, y])

                    # Get the 3D Coordinates
                    face_3d.append([x, y, lm.z])

            # Convert it to the NumPy array
            face_2d = np.array(face_2d, dtype=np.float64)

            # Convert it to the NumPy array
            face_3d = np.array(face_3d, dtype=np.float64)

            # The camera matrix
            focal_length = 1 * img_w

            cam_matrix = np.array([[focal_length, 0, img_h / 2],
                                   [0, focal_length, img_w / 2],
                                   [0, 0, 1]])

            # The distortion parameters
            dist_matrix = np.zeros((4, 1), dtype=np.float64)

            # Solve PnP
            success, rot_vec, trans_vec = cv2.solvePnP(
                face_3d, face_2d, cam_matrix, dist_matrix)

            # Get rotational matrix
            rmat, jac = cv2.Rodrigues(rot_vec)

            # Get angles
            angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

            # Get the y rotation degree
            x = angles[0] * 360
            y = angles[1] * 360
            z = angles[2] * 360

            head_text = ""

            # See where the user's head tilting
            if y < -10:
                head_text = "Looking Left"
            elif y > 10:
                head_text = "Looking Right"
            elif x < -10:
                head_text = "Looking Down"
            elif x > 10:
                head_text = "Looking Up"
            else:
                head_text = "Forward"

            # Display the nose direction
            nose_3d_projection, jacobian = cv2.projectPoints(
                nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)

            p1 = (int(nose_2d[0]), int(nose_2d[1]))
            p2 = (int(nose_2d[0] + y * 10), int(nose_2d[1] - x * 10))

            mesh_coords = landmarksDetection(frame, results, False)

            right_wrist = pose_landmarks[16]

            right_wrist_x = int(right_wrist.x * frame.shape[1])
            right_wrist_y = int(right_wrist.y * frame.shape[0])

            head_top = face_landmarks[10]
            head_below = face_landmarks[152]

            head_top_y = int(head_top.y * frame.shape[0])
            head_below_y = int(head_below.y * frame.shape[0])

            head_top_x = int(head_top.x * frame.shape[1])

            right_ear_hand_distance = head_top_x - right_wrist_x

            ratio = blinkRatio(frame, mesh_coords, RIGHT_EYE, LEFT_EYE)
            # cv.putText(frame, f'ratio {ratio}', (100, 100), FONTS, 1.0, utils.GREEN, 2)

            if ratio > 5.5:
                CEF_COUNTER += 1
                # cv.putText(frame, 'Blink', (200, 50), FONTS, 1.3, utils.PINK, 2)
                pass
            else:
                if CEF_COUNTER > CLOSED_EYES_FRAME:
                    TOTAL_BLINKS += 1
                    CEF_COUNTER = 0
            # # cv.putText(frame, f'Total Blinks: {TOTAL_BLINKS}', (100, 150), FONTS, 0.6, utils.GREEN, 2)
            # Blink Detector Counter Completed
            right_coords = [mesh_coords[p] for p in RIGHT_EYE]
            left_coords = [mesh_coords[p] for p in LEFT_EYE]
            crop_right, crop_left = eyesExtractor(
                frame, right_coords, left_coords)

            eye_position = positionEstimator(crop_right)
            eye_position_left = positionEstimator(crop_left)

            mouth_lip_upper = face_landmarks[13]
            mouth_lip_lower = face_landmarks[14]

            mouth_lip_upper_y = int(mouth_lip_upper.y * frame.shape[0])
            mouth_lip_lower_y = int(mouth_lip_lower.y * frame.shape[0])

            lips_distance = mouth_lip_lower_y - mouth_lip_upper_y

            if ((right_wrist_y > head_top_y) and (right_wrist_y < head_below_y) and (right_ear_hand_distance < 100) and (head_text == "Looking Down")):
                cv2.putText(image, "Embarrased, Got Caught", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                counter[6] += 1
            elif (lips_distance > 10):
                cv2.putText(image, "Disbelief", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                counter[4] += 1
            elif (head_text == "Looking Up" and ((eye_position == "LEFT" and eye_position_left == "LEFT") or (eye_position == "RIGHT" and eye_position_left == "RIGHT"))):
                cv2.putText(image, "Person Recalling Something", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            # elif((eye_position == "RIGHT" and eye_position_left == "RIGHT") or (eye_position == "LEFT" and eye_position_left == "LEFT")):
            #     cv2.putText(image, "Hostility, skeptical", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            elif (ratio > 5.5):
                cv2.putText(image, "Doesn't want to see, skeptical", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            elif (((eye_position == "Closed" and eye_position_left == "CENTER") or (eye_position_left == "Closed" and eye_position == "CENTER")) and lips_distance > 50):
                cv2.putText(image, "Approval", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            else:
                cv2.putText(image, "Neutral", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                counter[7] += 1
            # cv2.putText(image, "Disagree with Spoken Word", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        return av.VideoFrame.from_ndarray(image, format='bgr24')


webrtc_streamer(key="key",
                video_processor_factory=VideoProcessor,
                rtc_configuration={
                    "iceServers": get_ice_servers(),
                    "iceTransportPolicy": "relay",
                },
                media_stream_constraints={"video": True, "audio": False},
                )

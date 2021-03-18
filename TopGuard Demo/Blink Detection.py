import cv2  # 4.5.1
import dlib  # 19.21.1
from imutils import face_utils
from scipy.spatial import distance as dist

# video stream capture
cap = cv2.VideoCapture(0)  # 0 is capturing default system webcam

# facial recognition for bounding box
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# font
font = cv2.FONT_HERSHEY_SIMPLEX
# font scale
font_scale = 1
# Green color in BGR
blue = (255, 0, 0)
green = (0, 255, 0)
red = (0, 0, 255)
# text thickness (px)
thickness = 2

# from the perspective of the viewer
"""
    37  38              43  44
36          39      42          45
    41  40              47  46
"""

# for blink detection
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# initialize counters
frame_counter = 0
blink_frame_duration = 0
time_between_blink = 0
total_blinks = 0
total_people = 0
# caution True when liveness fails
caution = False


def eye_aspect_ratio(eye):
    # vertical landmarks distance (eye height)
    vert0 = dist.euclidean(eye[1], eye[5])
    vert1 = dist.euclidean(eye[2], eye[4])
    # horizontal landmarks distance (eye width
    horiz = dist.euclidean(eye[0], eye[3])

    # ratio of eye height to width
    ratio = (vert0 + vert1) / (2.0 * horiz)

    return ratio


# ---------- MAIN -------------------------

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    dlib_face = detector(gray)  # uses dlib face detector in grayscale to find faces
    if dlib_face:
        total_people = 1
    else:
        total_people = 0

    EYE_RATIO_THRESHOLD = 0.18  # ratio threshold to cross to count as a blink
    THRESHOLD_PRESENCE = 2  # number of frames for threshold to be active to count as a blink
    frame_counter += 1

    # find faces for bounding box
    faces = face_cascade.detectMultiScale(frame, scaleFactor=1.3, minNeighbors=5)

    # loop over dlib detected faces
    for (i, dlib_face) in enumerate(dlib_face):
        # prints face location in frame to console
        # [(top_left.x, top_left.y),(bottom_right.x, bottom_right.y)]
        print("i, dlib: " + str(i) + " " + str(dlib_face))

        # determine facial landmarks
        shape = predictor(gray, dlib_face)
        # convert landmark (x,y) coordinates to numpy array
        shape = face_utils.shape_to_np(shape)

        # draw bounding box for each face found
        for x, y, w, h in faces:
            # prints bounding box coordinates to console
            print((x, y), (x + w, y + h))
            frame = cv2.rectangle(frame, (x, y), (x + w, y + h), green, 2)
            center = (int(x + (w / 2)), int(y + (h / 2)))
            print("center: " + str(center))
            frame = cv2.circle(frame, center, 1, green, 1)
        print("No face visible")

        # find indexes of each eye
        (l_start, l_end) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (r_start, r_end) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

        # calculate average eye aspect ratio
        left_eye = shape[l_start:l_end]
        right_eye = shape[r_start:r_end]
        l_ear = eye_aspect_ratio(left_eye)
        r_ear = eye_aspect_ratio(right_eye)
        ear = (l_ear + r_ear) / 2.0

        # check if eye aspect ratio is below the threshold
        # if so, increment frame counter

        if ear < EYE_RATIO_THRESHOLD:
            blink_frame_duration += 1
            if blink_frame_duration > 90:  # 3 seconds closed
                frame = cv2.putText(frame, "WARNING:", (30, 200), font, 2, red, 3)
                frame = cv2.putText(frame, "CLOSED EYES", (30, 300), font, 2, red, 2)
                frame = cv2.putText(frame, "CONCERN", (30, 400), font, 2, red, 2)
        # otherwise ratio is not below threshold
        else:
            if ear > EYE_RATIO_THRESHOLD:
                time_between_blink += 1
                if time_between_blink > 450:  # 15 seconds without blinking
                    frame = cv2.putText(frame, "WARNING:", (30, 200), font, 2, red, 3)
                    frame = cv2.putText(frame, "OPEN EYES", (30, 300), font, 2, red, 2)
                    frame = cv2.putText(frame, "CONCERN", (30, 400), font, 2, red, 2)
            # if eyes were closed for long enough increment blinks
            if blink_frame_duration >= THRESHOLD_PRESENCE:
                total_blinks += 1
                # reset frame counters
                time_between_blink = 0
                blink_frame_duration = 0

        # posting text to frame
        frame = cv2.putText(frame, str(total_blinks) + " blinks", (x - 160, y + h - 5), font, 1, green, 2)
        frame = cv2.putText(frame, str(round(ear, 3)) + " EAR", (x - 190, y + h - 40), font, 1, green, 2)

    frame = cv2.putText(frame, "people: " + str(total_people), (30, 30), font, 1, green, 2)
    frame = cv2.putText(frame, "blink threshold: " + str(EYE_RATIO_THRESHOLD) + " EAR", (400, 30), font, 0.5, green, 1)

    cv2.imshow('TopGuard - detection model', frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        # manually restart blink count for testing and demo purposes
        total_blinks = 0
    if key == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()

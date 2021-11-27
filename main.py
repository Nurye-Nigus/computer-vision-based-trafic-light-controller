import cv2
from Nurye import *
# Create tracker object
tracker = EuclideanDistTracker()

abebe = cv2.VideoCapture("video.mp4")
# Object detection from Stable camera
abdi = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)
while True:
    ret, frame = abebe.read()
    height, width, _ = frame.shape
    # Extract Region of interest
    abeba_needed_size = frame[300: 900,600: 900]

    # 1. Object Detection
    solomon_masking = abdi.apply(abeba_needed_size)
    _, solomon_masking = cv2.threshold(solomon_masking, 254, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(solomon_masking, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    detections = []
    for cnt in contours:
        # Calculate area and remove small elements
        micki_area = cv2.contourArea(cnt)
        if micki_area > 100:
            #cv2.drawContours(roi, [cnt], -1, (0, 255, 0), 2)
            x, y, w, h = cv2.boundingRect(cnt)


            detections.append([x, y, w, h])

    # 2. Object Tracking
    boxes_ids = tracker.update(detections)
    for box_id in boxes_ids:
        x, y, w, h, id = box_id
        cv2.putText(abeba_needed_size , str(id), (x, y -10), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
        cv2.rectangle(abeba_needed_size , (x, y), (x + w, y + h), (0, 0, 0), 3)

    cv2.imshow("size", abeba_needed_size )
    cv2.imshow("Frame", frame)
    cv2.imshow("solomon_masking", solomon_masking)

    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()
abebe.release()

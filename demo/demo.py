import cv2
import time
import psutil
import GPUtil

import sys


def get_resource_usage():
    cpu_usage = psutil.cpu_percent(interval=None)
    gpu_usage = GPUtil.getGPUs()[0].load * 100 if GPUtil.getGPUs() else 0
    return cpu_usage, gpu_usage


# Initialize the webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Load the pre-trained Haar Cascade model for face detection
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

if not cap.isOpened():
    sys.exit("Failed to open webcam.")

prev_frame_time = 0
face_detection_interval = 1
emotion_analysis_interval = 1
frame_count = 0

# Open the log file
with open("performance_log.csv", "w") as log_file:
    log_file.write(
        "Timestamp,Frame Time (s),FPS,Latency (s),CPU Utilization (%),GPU Utilization (%)\n"
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        new_frame_time = time.time()
        frame_count += 1

        latency = "N/A"  # Default value if no face detection is performed
        if frame_count % face_detection_interval == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            start_time = time.time()
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            latency = time.time() - start_time  # Measure latency

            for x, y, w, h in faces:
                # Draw rectangle around each face
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

                if frame_count % emotion_analysis_interval == 0:
                    face = frame[y : y + h, x : x + w]
                    try:
                        # inference function
                        emotion = "Detection Failed"
                        pass
                    except:
                        emotion = "Detection Failed"
                    # Put emotion text above the rectangle
                    cv2.putText(
                        frame,
                        emotion,
                        (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (36, 255, 12),
                        2,
                    )

        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time

        # Get resource usage
        cpu_usage, gpu_usage = get_resource_usage()

        # Log data
        timestamp = time.time()
        frame_time = new_frame_time - prev_frame_time
        log_file.write(
            f"{timestamp},{frame_time},{fps},{latency},{cpu_usage},{gpu_usage}\n"
        )
        log_file.flush()

        cv2.imshow("Webcam", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()

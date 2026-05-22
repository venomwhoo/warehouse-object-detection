import cv2
import numpy as np
import requests
import tensorflow as tf
import time

# LOAD MODEL
model = tf.keras.models.load_model("warehouse_ai_model.h5")

# CLASSES
classes = ["Circle", "Square", "Unknown"]

# ESP32 URL
url = "http://172.20.10.4/capture"

# TIMER VARIABLES
last_detected = ""
detect_start = None

while True:

    try:

        # GET FRAME
        img_resp = requests.get(url, timeout=5)

        img_arr = np.frombuffer(img_resp.content, np.uint8)

        frame = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)

        if frame is None:
            continue

        # RESIZE
        frame = cv2.resize(frame, (900,700))

        h, w, _ = frame.shape

        # CENTER BOX
        size = 350

        x1 = w//2 - size//2
        y1 = h//2 - size//2

        x2 = x1 + size
        y2 = y1 + size

        # CROP ROI
        crop = frame[y1:y2, x1:x2]

        # MODEL INPUT
        img = cv2.resize(crop, (224,224))

        img = np.expand_dims(img, axis=0)

        # PREDICT
        prediction = model.predict(img, verbose=0)

        class_index = np.argmax(prediction)

        confidence = float(np.max(prediction))

        detected = classes[class_index]

        # LOW CONFIDENCE
        if confidence < 0.45:

            detected = ""

        # TIMER
        current_time = time.time()

        if detected == last_detected and detected != "":

            if detect_start is None:

                detect_start = current_time

            elapsed = current_time - detect_start

        else:

            last_detected = detected

            detect_start = current_time

            elapsed = 0

        # DEFAULT TEXT
        text = f"SHOW OBJECT ({confidence:.2f})"

        color = (255,255,255)

        # COUNTDOWN
        if detected != "" and elapsed < 3:

            remaining = 3 - int(elapsed)

            text = f"DETECTING {detected.upper()} IN {remaining}"

            color = (0,255,255)

        # FINAL RESULT
        elif elapsed >= 3:

            if detected == "ball":

                text = "NOW GO TO ID 1"

                color = (255,0,0)

            elif detected == "rubik":

                text = "NOW GO TO ID 2"

                color = (0,255,0)

            else:

                text = "NOW GO TO ID 3"

                color = (0,0,255)

        # DRAW BOX
        cv2.rectangle(
            frame,
            (x1,y1),
            (x2,y2),
            color,
            4
        )

        # SHOW TEXT
        cv2.putText(
            frame,
            text,
            (20,50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            3
        )

        # SHOW CONFIDENCE
        cv2.putText(
            frame,
            f"CONFIDENCE: {confidence:.2f}",
            (20,100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255,255,255),
            2
        )

        # SHOW VIDEO
        cv2.imshow("WAREHOUSE AI", frame)

    except Exception as e:

        print("ERROR:", e)

    # ESC TO EXIT
    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()

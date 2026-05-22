import cv2
import numpy as np
import requests
import tensorflow as tf

# LOAD MODEL
model = tf.keras.models.load_model("warehouse_ai_model.h5")

classes = ["ball", "rubik", "unknown"]

url = "http://172.20.10.4/capture"

while True:

    try:

        img_resp = requests.get(url, timeout=5)

        img_arr = np.frombuffer(img_resp.content, np.uint8)

        frame = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)

        if frame is None:
            continue

        frame = cv2.resize(frame, (900,700))

        h, w, _ = frame.shape

        # CENTER CROP
        size = 350

        x1 = w//2 - size//2
        y1 = h//2 - size//2

        x2 = x1 + size
        y2 = y1 + size

        crop = frame[y1:y2, x1:x2]

        # MODEL INPUT
        img = cv2.resize(crop, (224,224))

        img = np.expand_dims(img, axis=0)

        prediction = model.predict(img, verbose=0)

        class_index = np.argmax(prediction)

        confidence = np.max(prediction)

        detected = classes[class_index]

        # LOW CONFIDENCE
        if confidence < 0.85:
            detected = "unknown"

        # RESULT
        if detected == "ball":

            text = "BALL -> GO TO ID 1"
            color = (255,0,0)

        elif detected == "rubik":

            text = "RUBIK -> GO TO ID 2"
            color = (0,255,0)

        else:

            text = "UNKNOWN -> GO TO ID 3"
            color = (0,0,255)

        # DRAW CENTER BOX
        cv2.rectangle(
            frame,
            (x1,y1),
            (x2,y2),
            color,
            4
        )

        # TEXT
        cv2.putText(
            frame,
            f"{text} ({confidence:.2f})",
            (20,50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            3
        )

        cv2.imshow("WAREHOUSE AI", frame)

    except Exception as e:

        print(e)

    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()
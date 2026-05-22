import cv2
import numpy as np
import requests
import tensorflow as tf
import time  # Timer ke liye time module import kiya

# LOAD MODEL
model = tf.keras.models.load_model("warehouse_ai_model.h5")

classes = ["ball", "rubik", "unknown"]

url = "http://172.20.10.4/capture"

# --- TIMER VARIABLES ---
current_object = None
detect_start_time = 0
required_duration = 3.0  # 3 seconds ka timer

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

        # --- TIMER LOGIC ---
        # Agar naya object aaya hai ya pehli baar detect ho raha hai
        if detected != current_object:
            current_object = detected
            detect_start_time = time.time()  # Timer reset karo
            
        # Calculate karo kitna time ho gaya same object detect karte hue
        elapsed_time = time.time() - detect_start_time

        # --- RESULT & DISPLAY LOGIC ---
        if elapsed_time >= required_duration:
            # 3 second pure hone ke baad original message dikhao
            if detected == "ball":
                text = f"BALL -> GO TO ID 1 ({confidence:.2f})"
                color = (255, 0, 0)
            elif detected == "rubik":
                text = f"RUBIK -> GO TO ID 2 ({confidence:.2f})"
                color = (0, 255, 0)
            else:
                text = f"UNKNOWN -> GO TO ID 3 ({confidence:.2f})"
                color = (0, 0, 255)
        else:
            # 3 second se pehle waiting timer dikhao
            text = f"Confirming {detected.upper()}... {elapsed_time:.1f}s/3.0s"
            color = (0, 255, 255) # Yellow color processing dikhane ke liye

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
            text,
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

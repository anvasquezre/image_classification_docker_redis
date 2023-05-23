import json
import os
import time
from pathlib import Path  # Adding path library to avoid mistakes wheb building paths

import numpy as np
import redis
import settings
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import decode_predictions, preprocess_input
from tensorflow.keras.preprocessing import image

# Connect to Redis and assign to variable `db``
# Make use of settings.py module to get Redis settings like host, port, etc.


db = redis.Redis(
    db=settings.REDIS_DB_ID, host=settings.REDIS_IP, port=settings.REDIS_PORT
)

# Load your ML model and assign to variable `model`

model = ResNet50(include_top=True, weights="imagenet")


def predict(image_name):
    """
    Load image from the corresponding folder based on the image name
    received, then, run our ML model to get predictions.

    Parameters
    ----------
    image_name : str
        Image filename.

    Returns
    -------
    class_name, pred_probability : tuple(str, float)
        Model predicted class as a string and the corresponding confidence
        score as a number.
    """
    # Loading the image from UPLOAD_FOLDER
    img = image.load_img(
        f"{settings.UPLOAD_FOLDER}/{image_name}", target_size=(224, 224)
    )

    # Preprocessing

    x = image.img_to_array(img)

    x_batch = np.expand_dims(x, axis=0)

    ## Scaling pixel values to 0-1

    x_batch = preprocess_input(x_batch)

    # Making the prediction

    preds = model.predict(x_batch)

    # Decoding the prediction to get top1 probability
    _, class_name, pred_probability = decode_predictions(preds, top=1)[0][
        0
    ]  # Only 1 image, thus index 0

    return class_name, round(pred_probability, 4)


def classify_process():
    """
    Loop indefinitely asking Redis for new jobs.
    When a new job arrives, takes it from the Redis queue, uses the loaded ML
    model to get predictions and stores the results back in Redis using
    the original job ID so other services can see it was processed and access
    the results.

    Load image from the corresponding folder based on the image name
    received, then, run our ML model to get predictions.
    """
    while True:
        # Inside this loop you should add the code to:
        #   1. Take a new job from Redis
        #   2. Run your ML model on the given data
        #   3. Store model prediction in a dict with the following shape:
        #      {
        #         "prediction": str,
        #         "score": float,
        #      }
        #   4. Store the results on Redis using the original job ID as the key
        #      so the API can match the results it gets to the original job
        #      sent
        # Hint: You should be able to successfully implement the communication
        #       code with Redis making use of functions `brpop()` and `set()`.

        # Checking the queue with name settings.REDIS_QUEUE
        data = db.brpop(settings.REDIS_QUEUE, timeout=5)

        # Converting the JSON from job_data to a Dict
        if data:
            _, job_data = data
            msg = json.loads(job_data)

            # If both exists, then predict

            image_name, job_id = msg["image_name"], msg["id"]
            if (image_name, job_id) is not None:
                class_name, pred_probability = predict(image_name)
            else:
                print("Something went wrong, please try again")
            # Sending results to redis hashtable

            msg_content = {
                "prediction": class_name,
                "score": eval(str(pred_probability)),
            }

            # Turning msg content into a JSON
            prediction_content = json.dumps(msg_content)

            # Sending the message
            db.set(job_id, prediction_content)

        # Sleep for a bit
        time.sleep(settings.SERVER_SLEEP)


if __name__ == "__main__":
    # Now launch process
    print("Launching ML service...")
    classify_process()

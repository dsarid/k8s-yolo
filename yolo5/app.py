import time
from pathlib import Path
from detect import run
import yaml
from loguru import logger
import os
import boto3
import json
import polybot_helper_lib
import requests
import database_interface

REGION = os.environ['REGION']
S3_IMAGE_BUCKET = os.environ['S3_BUCKET']
QUEUE_NAME = os.environ['SQS_QUEUE_NAME']
DYNAMO_NAME = os.environ['DYNAMO_NAME']
ELB_URL = os.environ["TELEGRAM_APP_URL"]

sqs_client = boto3.client('sqs', region_name=REGION)
s3_client = boto3.client('s3')
dynamo_client = boto3.client('dynamodb', region_name=REGION)

with open("data/coco128.yaml", "r") as stream:
    names = yaml.safe_load(stream)['names']


def consume():
    logger.info("start runnin...")
    while True:
        response = sqs_client.receive_message(QueueUrl=QUEUE_NAME, MaxNumberOfMessages=1, WaitTimeSeconds=5)

        if 'Messages' in response:
            message = json.loads(response['Messages'][0]['Body'])
            receipt_handle = response['Messages'][0]['ReceiptHandle']

            # Use the ReceiptHandle as a prediction UUID
            prediction_id = response['Messages'][0]['MessageId']

            logger.info(f'prediction: {prediction_id}. start processing')

            # create a directory to store the images

            # images_dir = "images"
            # if not os.path.exists(images_dir):
            #     os.makedirs(images_dir)

            # Receives a URL parameter representing the image to download from S3
            img_name = message.get("img_name")
            chat_id = message.get("msg_id")
            s3_client.download_file(S3_IMAGE_BUCKET, img_name, img_name)
            original_img_path = img_name

            logger.info(f'prediction: {prediction_id}/{original_img_path}. Download img completed')

            # Predicts the objects in the image
            run(
                weights='yolov5s.pt',
                data='data/coco128.yaml',
                source=original_img_path,
                project='static/data',
                name=prediction_id,
                save_txt=True
            )

            logger.info(f'prediction: {prediction_id}/{original_img_path}. done')

            # This is the path for the predicted image with labels
            # The predicted image typically includes bounding boxes drawn around the detected objects,
            # along with class labels and possibly confidence scores.

            predicted_img_path = Path(f'static/data/{prediction_id}/{original_img_path}')

            # Uploads the predicted image (predicted_img_path) to S3 (be careful not to override the original image).
            polybot_helper_lib.upload_file(predicted_img_path, S3_IMAGE_BUCKET, s3_client, f"predicted_img/{img_name}")

            # Parse prediction labels and create a summary
            pred_summary_path = Path(f'static/data/{prediction_id}/labels/{original_img_path.split(".")[0]}.txt')
            if pred_summary_path.exists():
                with open(pred_summary_path) as f:
                    labels = f.read().splitlines()
                    labels = [line.split(' ') for line in labels]
                    labels = [{
                        'class': names[int(l[0])],
                        'cx': float(l[1]),
                        'cy': float(l[2]),
                        'width': float(l[3]),
                        'height': float(l[4]),
                    } for l in labels]

                logger.info(f'prediction: {prediction_id}/{original_img_path}. prediction summary:\n\n{labels}')

                prediction_summary = {
                    'prediction_id': prediction_id,
                    'chat_id': chat_id,
                    'original_img_path': original_img_path,
                    'predicted_img_path': predicted_img_path,
                    'labels': labels,
                    'time': time.time()
                }

                # store the prediction_summary in a DynamoDB table
                prediction_record = dynamo_client.put_item(
                    TableName=DYNAMO_NAME,
                    Item=polybot_helper_lib.dict_to_dynamo_format(prediction_summary)
                )

                prediction_record = database_interface.put_item(
                    "DYNAMODB", dynamo_client, polybot_helper_lib.dict_to_dynamo_format(prediction_summary),
                    DYNAMO_NAME
                )

                logger.info(f"http://{ELB_URL}/results?predictionId={prediction_id}")

                # perform a GET request to Polybot to `/results` endpoint
                result = requests.post(f"http://{ELB_URL}/results?predictionId={prediction_id}")
            else:
                result = requests.post(f"http://{ELB_URL}/results?predictionId=NONE:{chat_id}")
                logger.info("NOTHING TO PREDICT!")

            logger.info("Prediction done, keep running")
            # Delete the message from the queue as the job is considered as DONE
            if os.path.exists(original_img_path):
                os.remove(original_img_path)
            else:
                print("The file does not exist")
            sqs_client.delete_message(QueueUrl=QUEUE_NAME, ReceiptHandle=receipt_handle)


if __name__ == "__main__":
    consume()
    # final check

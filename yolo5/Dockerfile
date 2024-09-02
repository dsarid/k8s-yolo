FROM ultralytics/yolov5:latest-cpu
WORKDIR /usr/src/app
RUN pip install --upgrade pip
COPY requirements.txt .

RUN pip install --upgrade pip
# RUN pip cache purge
# RUN pip install --no-cache-dir loguru

RUN pip install -r requirements.txt
RUN curl -L https://github.com/ultralytics/yolov5/releases/download/v6.1/yolov5s.pt -o yolov5s.pt

COPY . .

ENV S3_BUCKET="dsarid-aws-project-s3"
ENV TELEGRAM_APP_URL="danielms-lb-1875524174.eu-central-1.elb.amazonaws.com:80"
ENV DYNAMO_NAME="danielms-aws-project-ddb"
ENV SQS_QUEUE_NAME="dms-aws-project-queue"

CMD ["python3", "app.py"]

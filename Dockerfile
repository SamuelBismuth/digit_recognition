FROM python:3.9

WORKDIR /digit_recognition

copy packages/requirements.txt .

RUN pip install -r requirements.txt

COPY src src
COPY data data
FROM python:3.8

WORKDIR /digit_recognition

COPY packages/requirements.txt .

RUN pip install -r requirements.txt

COPY src src
COPY data data
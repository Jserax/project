FROM python:3.10-slim

COPY requirements.txt .
COPY ./webapp/ /var/webapp/

RUN pip install --no-cache-dir -r requirements.txt

WORKDIR /var/webapp
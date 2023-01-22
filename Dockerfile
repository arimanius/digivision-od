FROM pytorch/pytorch:1.13.0-cuda11.6-cudnn8-devel

RUN rm /etc/apt/sources.list.d/cuda.list
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6 git -y

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy source code
COPY src .

ARG VERSION=local
ENV VERSION=${VERSION}
ENV REQUIREMENTS_FILE=requirements.txt
RUN pip install -e .

RUN chmod +x ./runserver.sh

ENTRYPOINT ./runserver.sh

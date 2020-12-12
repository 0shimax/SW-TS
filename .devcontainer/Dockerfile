FROM python:3.9.1-slim-buster

RUN apt update \
    && apt install -y curl wget gosu jq unzip libgomp1

RUN pip install -U pip \
    && pip install --pre -U scikit-learn \
    && pip install \
    numpy \
    pandas \
    lightgbm \
    ipython \
    jupyter \
    matplotlib \
    seaborn \
    plotly \
    flake8 \
    black \
    mlflow
# Start from the Python 3.10 slim image
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends \
        git \
        build-essential \
        libssl-dev \
        libffi-dev \
        libxml2-dev \
        libxslt1-dev \
        zlib1g-dev \
        libblas-dev \
        liblapack-dev \
        libatlas-base-dev \
        gfortran \
        cmake \
        openssh-client \
    && apt-get autoremove -y && apt-get clean -y && rm -rf /var/lib/apt/lists/*

# Set up Python environment
COPY requirements.txt /tmp/pip-tmp/
RUN pip3 --disable-pip-version-check --no-cache-dir install --upgrade pip \
    && pip3 --disable-pip-version-check --no-cache-dir install -r /tmp/pip-tmp/requirements.txt \
    && rm -rf /tmp/pip-tmp

# Set up streamlit_ace
RUN python3 -c "import streamlit_ace; import os; print(os.path.dirname(streamlit_ace.__file__))" > /tmp/streamlit_ace_path.txt
RUN mkdir -p $(cat /tmp/streamlit_ace_path.txt)/frontend/build
RUN chown -R root:root $(cat /tmp/streamlit_ace_path.txt)/frontend/build
RUN chmod -R 755 $(cat /tmp/streamlit_ace_path.txt)/frontend/build

# Create necessary directories
RUN mkdir -p /app/data/raw /app/data/prepared /app/data/test /app/metrics

# Set the working directory
WORKDIR /app

# Copy files
COPY . /app

# Set the default shell to bash
ENV SHELL /bin/bash

# Run your Python file and start the Streamlit server
CMD python /app/ui/mode.py && streamlit run ui_main.py
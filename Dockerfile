# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the \"License\");
# you may not use this file except in compliance with the License.\n",
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an \"AS IS\" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Dockerfile-gpu
FROM nvidia/cuda:9.0-cudnn7-runtime

# Installs necessary dependencies.
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    curl \
    make \
    libglib2.0-0 \
    libsm6 \
    libxrender-dev \
    libxext6 \
    apt-utils  \
    python3 \
    python3-dev && \
    rm -rf /var/lib/apt/lists/*

# Installs pip.
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3 get-pip.py && \
    pip3 install setuptools && \
    rm get-pip.py

WORKDIR /root

# Installs pytorch and torchvision.
RUN mkdir /root/dql_train
COPY requirements /root/dql_train/requirements
RUN pip3 install -r /root/dql_train/requirements

# Installs cloudml-hypertune for hyperparameter tuning.
# It’s not needed if you don’t want to do hyperparameter tuning.
RUN pip3 install cloudml-hypertune

# Installs google cloud sdk, this is mostly for using gsutil to export model.
RUN wget -nv \
    https://dl.google.com/dl/cloudsdk/release/google-cloud-sdk.tar.gz && \
    mkdir /root/tools && \
    tar xvzf google-cloud-sdk.tar.gz -C /root/tools && \
    rm google-cloud-sdk.tar.gz && \
    /root/tools/google-cloud-sdk/install.sh --usage-reporting=false \
    --path-update=false --bash-completion=false \
    --disable-installation-options && \
    rm -rf /root/.config/* && \
    ln -s /root/.config /config && \
    # Remove the backup directory that gcloud creates
    rm -rf /root/tools/google-cloud-sdk/.install/.backup

# Path configuration
ENV PATH $PATH:/root/tools/google-cloud-sdk/bin
# Make sure gsutil will use the default service account
RUN echo '[GoogleCompute]\nservice_account = default' > /etc/boto.cfg

# Copies the trainer code
COPY dqn_basic.py /root/dql_train/dqn_basic.py
COPY lib /root/dql_train/lib

WORKDIR /root/dql_train

# Sets up the entry point to invoke the trainer.
ENTRYPOINT ["python3", "dqn_basic.py"]
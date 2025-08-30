# 12.8.1-cudnn-devel-ubuntu22.04
FROM ngc.nju.edu.cn/nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

ARG VENV_NAME="cosyvoice"
ENV VENV=$VENV_NAME
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

ENV DEBIAN_FRONTEN=noninteractive
ENV PYTHONUNBUFFERED=1
SHELL ["/bin/bash", "--login", "-c"]

# 换源
RUN sed -i 's|http://.*archive.ubuntu.com|https://mirrors.aliyun.com|g' /etc/apt/sources.list && \
    sed -i 's|http://.*security.ubuntu.com|https://mirrors.aliyun.com|g' /etc/apt/sources.list && \
    apt-get update && \
    apt-get install -y curl git && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update -y  --fix-missing && \
    apt-get install -y git build-essential curl wget ffmpeg unzip git git-lfs sox libsox-dev && \
    apt-get clean && \
    git lfs install

RUN git config --global http.proxy http://host.docker.internal:7897  && \
    git config --global https.proxy http://host.docker.internal:7897

ARG HTTPS_PROXY=http://host.docker.internal:7897
ARG HTTP_PROXY=http://host.docker.internal:7897
# ==================================================================
# conda install and conda forge channel as default
# ------------------------------------------------------------------
# Install miniforge
# RUN wget --quiet -e  https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O ~/miniforge.sh
RUN curl -L -o ~/miniforge.sh "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"
RUN /bin/bash ~/miniforge.sh -b -p /opt/conda && \
    rm ~/miniforge.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo "source /opt/conda/etc/profile.d/conda.sh" >> /opt/nvidia/entrypoint.d/100.conda.sh && \
    echo "source /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate ${VENV}" >> /opt/nvidia/entrypoint.d/110.conda_default_env.sh && \
    echo "conda activate ${VENV}" >> $HOME/.bashrc

ENV PATH=/opt/conda/bin:$PATH

RUN conda config --add channels conda-forge && \
    conda config --set channel_priority strict
# ------------------------------------------------------------------
# ~conda
# ==================================================================

RUN conda create -y -n ${VENV} python=3.10
ENV CONDA_DEFAULT_ENV=${VENV}
ENV PATH=/opt/conda/bin:/opt/conda/envs/${VENV}/bin:$PATH
RUN mkdir -p /workspace/CosyVoice

RUN cd /workspace && git clone https://www.modelscope.cn/iic/CosyVoice-ttsfrd.git iic/CosyVoice-ttsfrd

RUN conda activate ${VENV} && cd /workspace/iic/CosyVoice-ttsfrd  && \
    unzip resource.zip -d . && \
    pip install ttsfrd_dependency-0.1-py3-none-any.whl && \
    pip install ttsfrd-0.4.2-cp310-cp310-linux_x86_64.whl

WORKDIR /workspace

ENV PYTHONPATH="${PYTHONPATH}:/workspace/CosyVoice:/workspace/CosyVoice/third_party/Matcha-TTS"

RUN git clone --recursive https://github.com/LemonCANDY42/CosyVoice.git && \
    cd CosyVoice && \
    git checkout Kenny && \
    cd ..

RUN conda activate ${VENV} && conda install -y -c conda-forge pynini==2.1.5
RUN conda activate ${VENV} && cd CosyVoice && \
    pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com

RUN conda activate ${VENV} && cd /workspace/CosyVoice && mkdir -p pretrained_models && \
    git clone https://www.modelscope.cn/iic/CosyVoice2-0.5B.git pretrained_models/CosyVoice2-0.5B  && \

    git clone https://www.modelscope.cn/iic/SenseVoiceSmall.git pretrained_models/SenseVoiceSmall


WORKDIR /workspace/CosyVoice

# ARG IMAGE=nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04
ARG IMAGE=nvidia/cudagl:11.3.0-devel-ubuntu20.04
FROM $IMAGE AS base

ARG JULIA_RELEASE=1.8
ARG JULIA_VERSION=1.8.1


# julia

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive \
    apt-get install --yes --no-install-recommends \
                    # basic stuff
                    curl ca-certificates && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update -q -y && \
        DEBIAN_FRONTEND=noninteractive \
        apt-get install -q -y --allow-change-held-packages\
        vim net-tools curl \
        libgl1-mesa-glx \
        xserver-xorg-video-dummy \
        libxrender1 \
        libpulse0 \
        libpulse-mainloop-glib0  \
        libnss3  \
        libxcomposite1 \
        libxcursor1 \
        libfontconfig1 \
        libxrandr2 \
        libasound2 \
        libglu1 \
        x11vnc \
        awesome \
        jq \
        nautilus\
        jupyter-core\
        zip\
        p7zip-full\
        apt-utils\
        octave\
        kmod\
        zlib1g\
        python-dev\
        bzip2\
        cmake\
        cuda-command-line-tools-11.3 \
        libcublas-11.3 \
        cuda-nvrtc-11.3\
        libcufft-11.3 \
        libcurand-11.3 \
        libcusolver-11.3 \
        libcusparse-11.3 \
        libfreetype6-dev \
        curl\
        libzmq3-dev \
        pkg-config\
        software-properties-common\
        libhdf5-serial-dev\
        git \
        at-spi2-core \
        libgtk-3-dev \
        xauth \
        xvfb \
        ninja-build \
        python3 \
        python3-pip \
        python3-setuptools \
        python3-wheel \
        ninja-build \
        wget \
        doxygen \
        autoconf \
        automake \
        cmake \
        g++ \
        gcc \
        pkg-config \
        make \
        nasm \
        xxd \
        yasm && \
        apt-get install -q -y --reinstall ca-certificates


#from https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html#package-manager-ubuntu-install
# wget https://developer.download.nvidia.com/compute/cuda/repos/${OS}/x86_64/cuda-${OS}.pin 

# sudo mv cuda-${OS}.pin /etc/apt/preferences.d/cuda-repository-pin-600
# sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/${OS}/x86_64/3bf863cc.pub
# sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/${OS}/x86_64/ /"
# sudo apt-get update
# Where ${OS} is ubuntu1804, ubuntu2004, or ubuntu2204.

# Install the cuDNN library:
# sudo apt-get install libcudnn8=${cudnn_version}-1+${cuda_version}
# sudo apt-get install libcudnn8-dev=${cudnn_version}-1+${cuda_version}
# Where:
# ${cudnn_version} is 8.5.0.*
# ${cuda_version} is cuda10.2 or cuda11.7


###############3from https://github.com/jrottenberg/ffmpeg/blob/main/docker-images/5.1/ubuntu2004/Dockerfile
RUN curl -s -L https://julialang-s3.julialang.org/bin/linux/x64/${JULIA_RELEASE}/julia-${JULIA_VERSION}-linux-x86_64.tar.gz | \
    tar -C /usr/local -x -z --strip-components=1 -f -


RUN export DISPLAY=:0

# system-wide packages

COPY Project.toml Manifest.toml /usr/local/share/julia/environments/v${JULIA_RELEASE}/

RUN JULIA_DEPOT_PATH=/usr/local/share/julia \
    julia -e 'using Pkg; Pkg.instantiate()'
# RUN JULIA_DEPOT_PATH=/usr/local/share/julia \
#     julia -e 'using Pkg; Pkg.instantiate(); Pkg.API.precompile()'

# generate the device runtime library for all known and supported devices
RUN JULIA_DEPOT_PATH=/usr/local/share/julia \
    julia -e 'using CUDA; CUDA.precompile_runtime()' && \
    chmod 644 /usr/local/share/julia/compiled/v${JULIA_RELEASE}/GPUCompiler/*/*.bc
    # TODO: fix this in GPUCompiler.jl

RUN julia -e 'using Pkg; Pkg.add("FillArrays")'


# user environment

# we use a single folder, /data, as the user depot regardless of the actual user
# (i.e., to be compatible with `docker run --user`, which might not have a $HOME)

RUN mkdir -m 0777 /data
ENV JULIA_DEPOT_PATH=/data:/usr/local/share/julia

ENV JULIA_HISTORY=/data/logs/repl_history.jl

ENV JULIA_CUDA_USE_BINARYBUILDER=false

ENV LD_LIBRARY_PATH=""


RUN git config --global user.name "Jakub Mitura"
RUN git config --global user.email "jakub.mitura14@gmail.com"

WORKDIR "/workspace"
COPY examples /workspace/examples

ENTRYPOINT ["/usr/local/bin/julia"]


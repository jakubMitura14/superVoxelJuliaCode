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
        cuda-command-line-tools-11.7 \
        libcublas-11.7 \
        cuda-nvrtc-11.7\
        libcufft-11.7 \
        libcurand-11.7 \
        libcusolver-11.7 \
        libcusparse-11.7 \
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


RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
RUN mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
RUN wget https://developer.download.nvidia.com/compute/cuda/11.7.0/local_installers/cuda-repo-ubuntu2004-11-7-local_11.7.0-465.19.01-1_amd64.deb
RUN dpkg -i cuda-repo-ubuntu2004-11-7-local_11.7.0-465.19.01-1_amd64.deb
RUN apt-key add /var/cuda-repo-ubuntu2004-11-7-local/7fa2af80.pub
RUN apt-get update
RUN apt-get -y install cuda-11.7
RUN apt-key add /var/cuda-repo-ubuntu2004-11-7-local/7fa2af80.pub
RUN echo 'APT::Install-Recommends "0" ; APT::Install-Suggests "0" ;' >> /etc/apt/apt.conf && \
    echo 'Dir::Cache::pkgcache "";\nDir::Cache::srcpkgcache "";' | tee /etc/apt/apt.conf.d/00_disable-cache-files



#from https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html#package-manager-ubuntu-install
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin 

RUN  mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
RUN  apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
RUN  add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
RUN  apt-get update

RUN  apt-get install libcudnn8=8.5.0.1-1+cuda11.7
RUN  apt-get install libcudnn8-dev=8.5.0.1-1+cuda11.7



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


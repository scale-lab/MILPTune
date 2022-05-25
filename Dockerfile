FROM continuumio/miniconda3

RUN apt update && apt install -y build-essential

COPY conda.yaml .
RUN conda install -c conda-forge -y mamba && \
    mamba env update -n base -f conda.yaml && \
    mamba clean -a

COPY . /MILPTune
WORKDIR /MILPTune
RUN python setup.py install

ENV LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/opt/conda/lib"
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
RUN rm /opt/conda/lib/libtinfo.so /opt/conda/lib/libtinfo.so.6
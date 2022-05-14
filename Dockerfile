FROM continuumio/miniconda3

RUN apt update && apt install -y build-essential

COPY conda.yaml .
RUN \
  conda install -c conda-forge -y mamba \
  && mamba env update -n base -f conda.yaml \
  && mamba clean -a
# RUN conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
RUN pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu113
RUN pip install torch-scatter -f https://data.pyg.org/whl/torch-1.11.0+cu113.html && \
    pip install torch-sparse -f https://data.pyg.org/whl/torch-1.11.0+cu113.html && \
    pip install torch-geometric
# RUN conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
# RUN pip install torch-scatter -f https://data.pyg.org/whl/torch-1.10.0+cu102.html && \
#     pip install torch-sparse -f https://data.pyg.org/whl/torch-1.10.0+cu102.html && \
#     pip install torch-geometric


COPY . /MILPTune
WORKDIR /MILPTune
RUN python setup.py install

RUN conda install -c nvidia libcusparse
RUN pip install python-dotenv pymongo pytorch_metric_learning matplotlib
# RUN conda install -c pytorch faiss-gpu
ENV LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/opt/conda/lib"
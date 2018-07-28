FROM ubuntu:16.04

LABEL maintainer="Connor Goddard <hello@connorlukegoddard.com>"

ENV JUPYTER_PASSWORD=ml
ENV JUPYTER_PORT=8888

COPY ./assets/ /assets/

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        wget \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install miniconda to /miniconda
RUN wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh --no-check-certificate
RUN bash Miniconda3-latest-Linux-x86_64.sh -p /miniconda -b
RUN rm Miniconda3-latest-Linux-x86_64.sh
ENV PATH=/miniconda/bin:${PATH}
RUN conda update -y conda

RUN conda create -n ml python=3.5 
RUN /bin/bash -c "source activate ml"
RUN conda install -y -c conda-forge python=3.5 anaconda jupyterlab pip ipython

# Set up our notebook config.
COPY ./assets/jupyter_notebook_config.py /root/.jupyter/

# Jupyter has issues with being run directly:
#   https://github.com/ipython/ipython/issues/7062

RUN cp /assets/run_jupyter.sh / && \
    cp /assets/run_jupyterlab.sh / && \
    rm -rf /assets

# IPython
EXPOSE 8888

CMD ["/run_jupyterlab.sh", "--allow-root"]
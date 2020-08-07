FROM nvidia/cuda:8.0-cudnn7-devel

RUN apt-get update

RUN apt install -y python3-pip

RUN apt install nano

RUN apt-get install -y git 
RUN apt-get install -y wget 

RUN apt-get clean 
RUN rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

ENV PATH=/opt/conda/bin:${PATH}

RUN wget --quiet https://repo.continuum.io/miniconda/Miniconda2-4.3.31-Linux-x86_64.sh \
&& /bin/bash /Miniconda2-4.3.31-Linux-x86_64.sh -b -p /opt/conda \
&& rm Miniconda2-4.3.31-Linux-x86_64.sh

RUN pip3 install --upgrade pip 

RUN pip3 install tensorflow-gpu
RUN pip3 install tensorflow-hub

RUN pip3 install scikit-learn
RUN pip3 install matplotlib
RUN pip3 install seaborn
RUN pip3 install keras


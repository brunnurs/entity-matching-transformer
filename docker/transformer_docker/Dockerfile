FROM pytorch/pytorch:latest

RUN pip install torch==1.2.0
RUN pip install pytorch-transformers==1.2.0
RUN pip install pandas
RUN pip install scikit-learn
RUN pip install imbalanced-learn
RUN pip install tensorboardX

ENV PYTHONPATH /workspace

WORKDIR /workspace
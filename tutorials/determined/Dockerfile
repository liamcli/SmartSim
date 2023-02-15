FROM determinedai/environments:cuda-10.2-pytorch-1.7-tf-1.15-gpu-9119094
RUN conda install git-lfs
RUN pip install jsonschema==3.0.1 tensorflow-gpu==2.5.2
RUN pip install smartsim
RUN smart build --device gpu && chmod -R 777 /opt/conda/lib/python3.7/site-packages/smart*
RUN chmod -R 777 /tmp


FROM python:3.6.5-stretch
RUN apt-get update && apt-get install -y build-essential cmake

# Install snet daemon
RUN mkdir -p /tmp/snetd && cd /tmp/snetd && \
      curl -OL https://github.com/singnet/snet-daemon/releases/download/v0.1.6/snet-daemon-v0.1.6-linux-amd64.tar.gz && \
      tar -xvf snet-daemon-v0.1.6-linux-amd64.tar.gz && \
      mv snet-daemon-v0.1.6-linux-amd64/snetd /usr/bin/snetd && \
      cd / && rm -r /tmp/snetd


ADD requirements.txt /requirements.txt
RUN pip3.6 install -r requirements.txt

ADD . /face-services
WORKDIR /face-services
RUN ./buildproto.sh

CMD ["python", "run_services.py"]
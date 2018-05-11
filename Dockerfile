FROM python:3.6.5-stretch
ADD . /face-services

RUN cd /face-services/
RUN apt-get update && apt-get install build-essential cmake
RUN pip3 install -r /requirements.txt

CMD ["python", "-m", "run_services.py"]
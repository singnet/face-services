FROM python:3.6.5-stretch
RUN apt-get update && apt-get install -y build-essential cmake

ADD . /face-services
WORKDIR /face-services
RUN pip install -r requirements.txt

CMD ["python", "run_services.py"]
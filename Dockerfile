FROM ubuntu

ADD requirements.txt .

RUN apt-get update \
  && apt-get install python3 python3-dev python3-pip \
  && pip3 install -r requirements.txt

CMD ["/bin/bash"]

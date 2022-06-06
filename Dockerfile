FROM mambaorg/micromamba:0.15.3
USER root
RUN apt-get update && DEBIAN_FRONTEND=“noninteractive” apt-get install -y --no-install-recommends \
  nginx \
  ca-certificates \
  apache2-utils \
  certbot \
  python3-certbot-nginx \
  sudo \
  cifs-utils \
  && \
  rm -rf /var/lib/apt/lists/*
RUN apt-get update && apt-get -y install cron
RUN mkdir /opt/pubag_app
RUN chmod -R 777 /opt/pubag_app
WORKDIR /opt/pubag_app
USER micromamba
COPY app.py app.py
COPY environment.yml environment.yml
COPY requirements.txt requirements.txt
COPY agri_pub2.csv agri_pub2.csv
COPY faiss_index.pickle faiss_index.pickle
RUN micromamba install -y -n base -f environment.yml && \
  micromamba clean --all --yes
COPY run.sh run.sh
COPY .env .env
USER root
RUN chmod a+x run.sh
CMD ["./run.sh"]

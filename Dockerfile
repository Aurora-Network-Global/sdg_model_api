FROM continuumio/anaconda3
LABEL maintainer="Eike Spielberg <eike.spielberg@uni-due.de>"
# We copy just the requirements.txt first to leverage Docker cache
EXPOSE 5000
WORKDIR /app
COPY . /app
RUN pip install --upgrade pip
RUN pip install -r conda-requirements.txt
RUN python -m nltk.downloader punkt
ENV FLASK_APP start.py

ENTRYPOINT ["gunicorn", "--bind", "0.0.0.0:5000", "--certfile", "/cert/sdg-classifier.crt", "--keyfile", "/cert/sdg-classifier.key", "start:app"]

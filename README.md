# SDG model deployment

A smal leight-weight flask application to facilitate the deployment of various SDG classifiers.

## Project structure

This project defines one route module with two endpoints (see below). The preprocessing and conversion of results is done in the model_service.py utilizing a number of tensorflow and transformer algorithms.

## Configuration

This project includes a docker compose file to start up the classifier. It needs two environment variables to be set:

* ```CERT_DIR```: The directory on the docker host, where the private key and the certificate for the flask application (or better said, for gunicorn) are stored
* ```MODEL_DIR```: The directory, where the model files are stored, which are then picked up by the tensorflow serving application


## Routes

This service offers one endpoint, which is accessible by GET or POST requests. The general URL scheme is

```HTTPS://<server-host>/classify/<model>```

where model depicts one of the four available models : aurora-sdg, aurora-sdg-multi (default), osdg, and elsevier-sdg-multi. As input, the value of the request param ```text``` (for GET requests) or a JSON formatted body ```{"text": text}``` is used 

## Docker

The project has two docker related file - a Dockerfile to prepare an image with the flask application served by a gunicorn instance and a docker-compose-single.yml file to start up this services together the osdg and the tensorflow-serving applications.


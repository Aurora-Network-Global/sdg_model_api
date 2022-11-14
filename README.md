# SDG model deployment

A smal leight-weight flask application to facilitate the deployment of various SDG classifiers.


# Configuration

This project includes a docker compose file to start up the classifier. It needs two environment variables to be set:

* ```CERT_DIR```: The directory on the docker host, where the private key and the certificate for the flask application (or better said, for gunicorn) are stored
* ```MODEL_DIR```: The directory, where the model files are stored, which are then picked up by the tensorflow serving application

import requests_cache
from flask import Flask
from flask_bootstrap import Bootstrap
from flask_cors import CORS


def create_app(config_filename=None):
    app = Flask(__name__, instance_relative_config=True)

    app.logger.info('enabling CORS support')
    # enable CORS support
    CORS(app, origins='*')

    requests_cache.install_cache('sdg_cache')

    Bootstrap(app)

    # register all blueprints
    app.logger.info('registering blueprints')
    register_blueprints(app)
    return app


def register_blueprints(app):
    # Since the application instance is now created, register each Blueprint
    # with the Flask application instance (app)
    from app.model import model_blueprint

    app.register_blueprint(model_blueprint)


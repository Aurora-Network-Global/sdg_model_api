"""
The eids Blueprint handles the creation and retrieval of EID lists
"""
from flask import Blueprint
model_blueprint = Blueprint('model', __name__, template_folder='templates')

from . import model_routes
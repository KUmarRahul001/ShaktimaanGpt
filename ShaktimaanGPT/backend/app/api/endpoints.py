from flask import Blueprint

api = Blueprint('api', __name__)

@api.route('/example')
def example():
    return 'Hello from the API!'
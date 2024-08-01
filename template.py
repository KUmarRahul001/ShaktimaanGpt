import os

def create_file(path, content):
    """Create a file with the given content."""
    with open(path, 'w') as f:
        f.write(content)

def create_structure(base_path, structure):
    """Recursively create folders and files based on the provided structure."""
    for name, value in structure.items():
        path = os.path.join(base_path, name)
        if isinstance(value, dict):
            os.makedirs(path, exist_ok=True)
            create_structure(path, value)
        elif isinstance(value, list):
            os.makedirs(path, exist_ok=True)
            for file_name in value:
                create_file(os.path.join(path, file_name['name']), file_name.get('content', ''))

def create_project_structure(base_path):
    """Create the full project structure."""
    structure = {
        'frontend': {
            'public': [
                {'name': 'index.html', 'content': '<!DOCTYPE html>\n<html>\n<head>\n    <title>ShaktimaanGPT</title>\n</head>\n<body>\n    <div id="root"></div>\n    <script src="index.js"></script>\n</body>\n</html>'},
                {'name': 'favicon.ico', 'content': ''}
            ],
            'src': {
                'components': [
                    {'name': 'Header.js', 'content': "import React from 'react';\n\nfunction Header() {\n    return <header>Header</header>;\n}\n\nexport default Header;"},
                    {'name': 'Footer.js', 'content': "import React from 'react';\n\nfunction Footer() {\n    return <footer>Footer</footer>;\n}\n\nexport default Footer;"},
                    {'name': 'ChatBox.js', 'content': "import React from 'react';\n\nfunction ChatBox() {\n    return <div>ChatBox</div>;\n}\n\nexport default ChatBox;"}
                ],
                'pages': [
                    {'name': 'HomePage.js', 'content': "import React from 'react';\n\nfunction HomePage() {\n    return <div>Home Page</div>;\n}\n\nexport default HomePage;"},
                    {'name': 'ProfilePage.js', 'content': "import React from 'react';\n\nfunction ProfilePage() {\n    return <div>Profile Page</div>;\n}\n\nexport default ProfilePage;"}
                ],
                'services': [
                    {'name': 'apiService.js', 'content': "import axios from 'axios';\n\nconst api = axios.create({\n    baseURL: 'http://localhost:5000/api'\n});\n\nexport default api;"}
                ],
                'styles': [
                    {'name': 'App.css', 'content': '/* Add your CSS here */'},
                    {'name': 'index.css', 'content': '/* Add your CSS here */'}
                ]
            },
            'package.json': '{\n  "name": "shaktimaan-gpt-frontend",\n  "version": "1.0.0",\n  "main": "index.js",\n  "dependencies": {},\n  "scripts": {}\n}',
            'README.md': '# Shaktimaan GPT Frontend\n\nThis is the frontend for the Shaktimaan GPT project.'
        },
        'backend': {
            'app': {
                'api': [
                    {'name': 'endpoints.py', 'content': "from flask import Blueprint\n\napi = Blueprint('api', __name__)\n\n@api.route('/example')\ndef example():\n    return 'Hello from the API!'"}
                ],
                'models': [
                    {'name': 'user.py', 'content': "from flask_sqlalchemy import SQLAlchemy\n\ndb = SQLAlchemy()\n\nclass User(db.Model):\n    id = db.Column(db.Integer, primary_key=True)\n    username = db.Column(db.String(80), unique=True, nullable=False)\n    email = db.Column(db.String(120), unique=True, nullable=False)\n    password = db.Column(db.String(120), nullable=False)"}
                ],
                'services': [
                    {'name': 'chat_service.py', 'content': "def process_chat(message):\n    return 'Processed message: ' + message"},
                    {'name': 'user_service.py', 'content': "def get_user(user_id):\n    return 'User with ID: ' + str(user_id)"}
                ]
            },
            'config': [
                {'name': 'settings.py', 'content': "import os\n\nclass Config:\n    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL')\n    SQLALCHEMY_TRACK_MODIFICATIONS = False"},
                {'name': 'config.yaml', 'content': 'database:\n  uri: "sqlite:///db.sqlite3"'}
            ],
            'docker': [
                {'name': 'Dockerfile', 'content': 'FROM python:3.9\n\nWORKDIR /app\n\nCOPY requirements.txt .\nRUN pip install -r requirements.txt\n\nCOPY . .\n\nCMD ["python", "main.py"]'},
                {'name': 'docker-compose.yml', 'content': 'version: "3"\nservices:\n  backend:\n    build: ./backend\n    ports:\n      - "5000:5000"\n    volumes:\n      - ./backend:/app'}
            ],
            'main.py': "from flask import Flask\nfrom app.api.endpoints import api\n\napp = Flask(__name__)\napp.config.from_object('config.settings.Config')\napp.register_blueprint(api, url_prefix='/api')\n\nif __name__ == '__main__':\n    app.run(debug=True)",
            'requirements.txt': 'Flask\nFlask-SQLAlchemy\n'
        },
        'ml': {
            'notebooks': [
                {'name': 'data_preprocessing.ipynb', 'content': 'Preprocessing notebook content'},
                {'name': 'model_training.ipynb', 'content': 'Model training notebook content'}
            ],
            'scripts': [
                {'name': 'preprocess_data.py', 'content': 'Data preprocessing script content'},
                {'name': 'train_model.py', 'content': 'Model training script content'}
            ],
            'models': [
                {'name': 'gpt_model.pkl', 'content': 'Trained GPT model file'}
            ],
            'data': [
                {'name': 'training_data.csv', 'content': 'Training data content'},
                {'name': 'validation_data.csv', 'content': 'Validation data content'}
            ]
        },
        'database': {
            'migrations': [
                {'name': 'migration_script.sql', 'content': 'SQL migration script content'}
            ],
            'schemas': [
                {'name': 'schema.sql', 'content': 'Database schema content'}
            ]
        },
        'docs': {
            'architecture': [
                {'name': 'system_architecture.pdf', 'content': 'System architecture document'}
            ],
            'user_manual': [
                {'name': 'user_manual.md', 'content': 'User manual content'}
            ],
            'api_docs': [
                {'name': 'api_documentation.md', 'content': 'API documentation content'}
            ]
        },
        'README.md': '# Shaktimaan GPT\n\nThis is the Shaktimaan GPT project.\n\n## Project Structure\n\n- frontend\n- backend\n- ml\n- database\n- docs\n\n## Getting Started\n\nInstructions to get started with the project.',
        'requirements.txt': 'Flask\nFlask-SQLAlchemy\nTensorFlow\nPyTorch\n'
    }

    create_structure(base_path, structure)

if __name__ == "__main__":
    project_base_path = "ShaktimaanGPT"
    create_project_structure(project_base_path)

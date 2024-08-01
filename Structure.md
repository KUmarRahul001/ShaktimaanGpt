# ShaktimaanGPT

## Project Overview

ShaktimaanGPT is an advanced AI system leveraging GPT technology to create an interactive chatbot. The project aims to deliver a robust user interface, powerful backend, and effective language model integration.

## Project Structure

```plaintext
E:.
│   README.MD
│   template.py
│
└───ShaktimaanGPT
    ├───backend
    │   ├───app
    │   │   ├───api
    │   │   │       endpoints.py
    │   │   ├───models
    │   │   │       user.py
    │   │   └───services
    │   │           chat_service.py
    │   │           user_service.py
    │   ├───config
    │   │       config.yaml
    │   │       settings.py
    │   └───docker
    │           docker-compose.yml
    │           Dockerfile
    ├───database
    │   ├───migrations
    │   │       migration_script.sql
    │   └───schemas
    │           schema.sql
    ├───docs
    │   ├───api_docs
    │   │       api_documentation.md
    │   ├───architecture
    │   │       system_architecture.pdf
    │   └───user_manual
    │           user_manual.md
    ├───frontend
    │   ├───public
    │   │       favicon.ico
    │   │       index.html
    │   └───src
    │       ├───components
    │       │       ChatBox.js
    │       │       Footer.js
    │       │       Header.js
    │       ├───pages
    │       │       HomePage.js
    │       │       ProfilePage.js
    │       ├───services
    │       │       apiService.js
    │       └───styles
    │               App.css
    │               index.css
    └───ml
        ├───data
        │   │   training_data.csv
        │   │   validation_data.csv
        │   └───Raw_Dataset
        ├───models
        │       gpt_model.pkl
        ├───notebooks
        │       data_preprocessing.ipynb
        │       model_training.ipynb
        └───scripts
                preprocess_data.py
                train_model.py

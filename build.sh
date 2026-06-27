#!/usr/bin/env bash

pip install -r requirements.txt

python -c "import nltk; nltk.download('stopwords', download_dir='/opt/render/nltk_data')"
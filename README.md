# AQI
Air Quality Index Dashboard - for an assignment

![Python](https://img.shields.io/badge/Python-3.12-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Deep_Learning-orange.svg)
![GitHub Actions](https://img.shields.io/badge/GitHub_Actions-Automated-success.svg)



#### How to just run dashboard and download models ####

###### B1: Clone and create virtual environment
```bash
git clone git@github:Aza126/AQI.git
# 'yes', spam 'Enter'
cd AQI
python -m venv .venv
```
###### B2: Activate virtual environment
```bash
# Linux/MacOS
source .venv/bin/activate
# Windows
.venv/Scripts/activate
```
###### B3: Create .env -> ask Aza for the words needed
```bash
cp .env.example .env
# or
copy .env.example .env
# replace some words in .env
```
###### B4: Download needed librabies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```
###### Run dashboard
```bash
python -m streamlit run dashboard/app.py
# How to stop
Ctrl + C 
```
##### Download models
```bash
mkdir -p artifacts models/rf models/lstm
python -m src.scripts.download_models
```


#### How to explore codes yourself, add these steps ####

###### B1: Create your DB
```bash
# Create your own MongoDB Atlas DB
# Change db_name in `config.yaml` & MONGO_URI in `.env`
```
###### B2: Run
```bash
python -m src.scripts.setup_mongodb
python -m src.ingestion
python -m src.scripts.prepare_training_data
python -m src.training.rf
python -m src.training.lstm
python -m src.preprocess
python -m src.inference
```
###### Run dashboard
```bash
python -m streamlit run dashboard/app.py
# How to stop
Ctrl + C 
```

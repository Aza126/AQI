# AQI
Air Quality Index Dashboard - for an assignment

![Python](https://img.shields.io/badge/Python-3.13+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Deep_Learning-orange.svg)
![GitHub Actions](https://img.shields.io/badge/GitHub_Actions-Automated-success.svg)

#### Steps for training models

###### B1: Clone
```bash
git clone git@github:Aza126/AQI.git
# spam 'Enter'
```
 ###### B2: Set up virtual environment
```bash
cd AQI
python -m venv .venv
source .venv/bin/activate
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
pip install -r requirements.txt
```
###### B5: Download needed files
```bash
python -m src.scripts.download_models
```
###### B6: Train your models
```bash
# Write and run your `rf.py` and `lstm.py`
```
###### B7: Test your models
```bash
# Test `rf_v1.pkl` and `lstm_v1.h5`
python -m src.inference
python -m streamlit run dashboard/app.py
```
###### B8: Save your `rf.pkl` and `lstm.h5` to Google Drive
```bash
# Replace the old empty ones
```
# 📈 Walmart Demand Forecasting — Sales Prediction Intelligence

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white) ![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white) ![XGBoost](https://img.shields.io/badge/XGBoost-006400?style=for-the-badge) ![Scikit--learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white) ![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

> Predict next week's store sales — before you overstock or run out. Beat guesswork with data.

---

## 🚀 Live Demo

🔗 [Try it Live Here](https://walmart-demand-forecastingg.streamlit.app/)

## 💡 Problem Statement
Retailers plan inventory and staffing weeks in advance, but most still rely on "same as last week" guesswork — leading to overstock (wasted capital) or stockouts (lost sales). Walmart Demand Forecasting predicts store-level weekly sales using 3 years of historical data, giving planning teams a validated number instead of a hunch.

## ✨ Key Features

- 📊 **EDA Dashboard** — Sales trend, holiday impact, seasonality, correlation heatmap
- 🏬 **Store Segmentation** — High/Medium/Low performing stores by total sales
- 🎯 **Model Evaluation** — Baseline comparison + 4-model benchmark + residual diagnostics
- 🔮 **Forecast Engine** — Pick any store → forecast 1–12 weeks ahead → download as CSV

## 🔍 Key Findings

- Recent sales momentum (lag + rolling averages) drives **>95% of prediction power** — temperature, fuel price, and unemployment barely matter
- Holiday weeks boost sales by **~7.8%** — a quantified number for inventory/staffing planning
- Top store contributes **4.5% of total revenue** — a handful of stores carry outsized weight
- December is the **peak sales month** — supply chain capacity should be planned around it

## 🛠️ Tech Stack

- **XGBoost** — Primary forecasting model (1.21% MAPE)
- **LightGBM, Random Forest, Linear Regression** — Benchmarked for model selection
- **Scikit-learn** — Training, evaluation, train/test splitting
- **Pandas + NumPy** — Feature engineering (lags, rolling stats)
- **Plotly** — Interactive visualizations
- **Streamlit** — Dashboard + deployment

## 📊 Model Performance

- **Linear Regression** — MAPE 7.43% | MAE $41,562 | RMSE $60,021
- **Random Forest** — MAPE 2.17% | MAE $15,862 | RMSE $26,153
- **LightGBM** — MAPE 1.32% | MAE $9,888 | RMSE $19,897
- **XGBoost ✅** — MAPE **1.21%** | MAE **$8,481** | RMSE **$18,456**

XGBoost beats a naive last-week-sales baseline (7.00% MAPE) by **83%**, with residuals centered near $0 — confirming no systematic bias.
## 📸 Screenshots

## Overview
<img width="1097" height="595" alt="image" src="https://github.com/user-attachments/assets/1250f2bd-912c-49d8-9956-e70403615444" />


## EDA
<img width="1108" height="568" alt="image" src="https://github.com/user-attachments/assets/4759e648-9949-419f-b36a-43edfacb53b6" />
<img width="1115" height="590" alt="image" src="https://github.com/user-attachments/assets/c05adc5f-6852-4cbf-879b-815c6b0ad808" />


## Store Segmentation
<img width="1105" height="551" alt="image" src="https://github.com/user-attachments/assets/d81be9b7-0fb3-4852-b006-b5cce2762e6b" />

## Model Performance
<img width="1139" height="607" alt="image" src="https://github.com/user-attachments/assets/2129d567-222c-49d2-b8c2-a20963689e3a" />
<img width="1073" height="461" alt="image" src="https://github.com/user-attachments/assets/d0a7456b-0205-4ac4-adf4-d24bb291bf66" />

## Forecast
<img width="1049" height="615" alt="image" src="https://github.com/user-attachments/assets/316d20ae-81b1-4824-9791-973fb93fbca0" />



⚙️ Run Locally
bashgit clone https://github.com/pritamk001/walmart-demand-forecasting.git

cd walmart-demand-forecasting

pip install -r requirements.txt

streamlit run app.py

🔗 Links
🌐 Live App: walmart-demand-forecastingg.streamlit.app

💻 GitHub: github.com/pritamk001/walmart-demand-forecasting

Built with ❤️ using XGBoost + Streamlit + Plotly

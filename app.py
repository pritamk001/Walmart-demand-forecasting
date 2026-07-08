import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
try:
    from lightgbm import LGBMRegressor
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

# ----------------------------- PAGE CONFIG -----------------------------
st.set_page_config(
    page_title="Walmart Demand Forecasting",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------- DARK THEME CSS -----------------------------
st.markdown("""
<style>
    .stApp { background-color: #0E1117; color: #FAFAFA; }
    .metric-card {
        background-color: #1A1D24;
        border: 1px solid #2A2D34;
        border-radius: 10px;
        padding: 18px 20px;
        margin-bottom: 10px;
    }
    .metric-label { font-size: 13px; color: #9CA3AF; font-weight: 500; }
    .metric-value { font-size: 26px; color: #FAFAFA; font-weight: 700; }
    .metric-sub { font-size: 12px; color: #6EE7B7; }
    h1, h2, h3 { color: #FAFAFA; }
    .insight-box {
        background-color: #1A1D24;
        border-left: 4px solid #2196F3;
        border-radius: 6px;
        padding: 14px 18px;
        margin-bottom: 12px;
        color: #E5E7EB;
    }
</style>
""", unsafe_allow_html=True)

st.title("📈 Walmart Weekly Sales — Demand Forecasting Dashboard")
st.caption("XGBoost-powered forecasting with lag features, rolling statistics, and business insights")

# ----------------------------- DATA LOADING -----------------------------
@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y', errors='coerce')
    if df['Date'].isna().all():
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.sort_values('Date').reset_index(drop=True)
    df['Month'] = df['Date'].dt.month
    df['Year'] = df['Date'].dt.year
    df['Week'] = df['Date'].dt.isocalendar().week.astype(int)
    return df

data_source = None
local_path = "Walmart_Sales.csv"
import os
if os.path.exists(local_path):
    data_source = local_path
    st.sidebar.success("✅ Loaded local Walmart_Sales.csv")
else:
    uploaded = st.sidebar.file_uploader("Upload Walmart_Sales.csv", type=["csv"])
    if uploaded is not None:
        data_source = uploaded

if data_source is None:
    st.info("👈 Upload `Walmart_Sales.csv` from the sidebar to get started.")
    st.stop()

df = load_data(data_source)

# ----------------------------- FEATURE ENGINEERING -----------------------------
@st.cache_data
def engineer_features(df):
    df = df.sort_values(['Store', 'Date']).reset_index(drop=True)
    df['Lag_1'] = df.groupby('Store')['Weekly_Sales'].shift(1)
    df['Lag_2'] = df.groupby('Store')['Weekly_Sales'].shift(2)
    df['Lag_4'] = df.groupby('Store')['Weekly_Sales'].shift(4)
    df['Rolling_Mean_4'] = df.groupby('Store')['Weekly_Sales'].transform(lambda x: x.shift(1).rolling(4).mean())
    df['Rolling_Std_4'] = df.groupby('Store')['Weekly_Sales'].transform(lambda x: x.shift(1).rolling(4).std())
    df['Sales_Growth'] = df.groupby('Store')['Weekly_Sales'].pct_change()
    df = df.dropna().reset_index(drop=True)
    return df

df_feat = engineer_features(df)

# ----------------------------- MODEL TRAINING -----------------------------
@st.cache_resource
def train_model(df_feat):
    feature_cols = ['Store', 'Month', 'Year', 'Week', 'Holiday_Flag',
                     'Temperature', 'Fuel_Price', 'CPI', 'Unemployment',
                     'Lag_1', 'Lag_2', 'Lag_4', 'Rolling_Mean_4',
                     'Rolling_Std_4', 'Sales_Growth']
    X = df_feat[feature_cols]
    y = df_feat['Weekly_Sales']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6, random_state=42)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    y_pred = model.predict(X_test)

    # Baseline: naive forecast using Lag_1 (last week's sales)
    baseline_pred = X_test['Lag_1'].values

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

    baseline_mae = mean_absolute_error(y_test, baseline_pred)
    baseline_rmse = np.sqrt(mean_squared_error(y_test, baseline_pred))
    baseline_mape = np.mean(np.abs((y_test - baseline_pred) / y_test)) * 100

    importance_df = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)

    # ---------------- Model comparison ----------------
    comparison_models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42),
        'XGBoost': model,
    }
    if LIGHTGBM_AVAILABLE:
        comparison_models['LightGBM'] = LGBMRegressor(n_estimators=500, learning_rate=0.05, max_depth=6, random_state=42, verbose=-1)

    comparison_rows = []
    for name, m in comparison_models.items():
        if name != 'XGBoost':
            m.fit(X_train, y_train)
            pred = m.predict(X_test)
        else:
            pred = y_pred
        comparison_rows.append({
            'Model': name,
            'MAE': mean_absolute_error(y_test, pred),
            'RMSE': np.sqrt(mean_squared_error(y_test, pred)),
            'MAPE': np.mean(np.abs((y_test - pred) / y_test)) * 100
        })
    comparison_df = pd.DataFrame(comparison_rows).sort_values('MAPE').reset_index(drop=True)

    return {
        'model': model, 'feature_cols': feature_cols,
        'X_test': X_test, 'y_test': y_test, 'y_pred': y_pred,
        'baseline_pred': baseline_pred,
        'metrics': {'mae': mae, 'rmse': rmse, 'mape': mape},
        'baseline_metrics': {'mae': baseline_mae, 'rmse': baseline_rmse, 'mape': baseline_mape},
        'importance_df': importance_df,
        'comparison_df': comparison_df
    }

results = train_model(df_feat)

# ----------------------------- TABS -----------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏠 Overview", "📊 EDA", "🏬 Store Segmentation",
    "🎯 Model Performance", "🔮 Forecast"
])

# ============================================================
# TAB 1: OVERVIEW
# ============================================================
with tab1:
    st.subheader("Business Snapshot")

    total_sales = df['Weekly_Sales'].sum()
    avg_sales = df['Weekly_Sales'].mean()
    n_stores = df['Store'].nunique()
    n_weeks = df['Date'].nunique()

    holiday_avg = df[df['Holiday_Flag'] == 1]['Weekly_Sales'].mean()
    non_holiday_avg = df[df['Holiday_Flag'] == 0]['Weekly_Sales'].mean()
    holiday_boost = ((holiday_avg - non_holiday_avg) / non_holiday_avg) * 100

    c1, c2, c3, c4 = st.columns(4)
    for col, label, value, sub in [
        (c1, "Total Sales", f"${total_sales/1e6:,.1f}M", f"{n_weeks} weeks tracked"),
        (c2, "Avg Weekly Sales", f"${avg_sales:,.0f}", f"across {n_stores} stores"),
        (c3, "Holiday Sales Boost", f"+{holiday_boost:.1f}%", "vs non-holiday weeks"),
        (c4, "Model MAPE", f"{results['metrics']['mape']:.2f}%", "XGBoost forecast error"),
    ]:
        col.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-sub">{sub}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("💡 Business Insights")
    top_store = df.groupby('Store')['Weekly_Sales'].sum().idxmax()
    top_store_share = df.groupby('Store')['Weekly_Sales'].sum().max() / total_sales * 100
    best_month = df.groupby('Month')['Weekly_Sales'].mean().idxmax()

    st.markdown(f"""
    <div class="insight-box">📌 <b>Holiday weeks boost sales by {holiday_boost:.1f}%</b> on average — inventory and staffing should scale up ahead of these weeks.</div>
    <div class="insight-box">📌 <b>Store {top_store}</b> is the single largest contributor at <b>{top_store_share:.1f}%</b> of total sales — prioritize stock allocation and uptime here.</div>
    <div class="insight-box">📌 Month <b>{best_month}</b> shows the highest average weekly sales historically — plan promotions and supply chain capacity accordingly.</div>
    <div class="insight-box">📌 The XGBoost model beats a naive last-week-sales baseline by <b>{(results['baseline_metrics']['mape'] - results['metrics']['mape']):.2f} percentage points</b> in MAPE — confirming the model adds real forecasting value beyond a simple heuristic.</div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.caption("⚠️ **Model limitations:** forecasts rely on recent lag/rolling features, so new stores with no sales history, or extreme one-off shocks (e.g. natural disasters, sudden macro events), are not well captured without retraining.")

# ============================================================
# TAB 2: EDA
# ============================================================
with tab2:
    st.subheader("Exploratory Data Analysis")

    total_trend = df.groupby('Date')['Weekly_Sales'].sum().reset_index()
    fig1 = px.line(total_trend, x='Date', y='Weekly_Sales', title="Overall Weekly Sales Trend")
    fig1.update_traces(line_color='#2196F3')
    st.plotly_chart(fig1, use_container_width=True)

    colA, colB = st.columns(2)
    with colA:
        hol = df.groupby('Holiday_Flag')['Weekly_Sales'].mean().reset_index()
        hol['Holiday_Flag'] = hol['Holiday_Flag'].map({0: 'Non-Holiday', 1: 'Holiday'})
        fig2 = px.bar(hol, x='Holiday_Flag', y='Weekly_Sales', color='Holiday_Flag',
                      title="Average Sales: Holiday vs Non-Holiday",
                      color_discrete_map={'Non-Holiday': '#4CAF50', 'Holiday': '#F44336'})
        st.plotly_chart(fig2, use_container_width=True)
    with colB:
        top10 = df.groupby('Store')['Weekly_Sales'].sum().sort_values(ascending=False).head(10).reset_index()
        top10['Store'] = top10['Store'].astype(str)
        fig3 = px.bar(top10, x='Store', y='Weekly_Sales', title="Top 10 Stores by Total Sales",
                      color_discrete_sequence=['#9C27B0'])
        st.plotly_chart(fig3, use_container_width=True)

    colC, colD = st.columns(2)
    with colC:
        month_names = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
        monthly = df.groupby('Month')['Weekly_Sales'].mean().reset_index()
        monthly['Month_Name'] = monthly['Month'].apply(lambda m: month_names[m-1])
        fig4 = px.bar(monthly, x='Month_Name', y='Weekly_Sales', title="Average Sales by Month",
                      color_discrete_sequence=['#FF9800'])
        st.plotly_chart(fig4, use_container_width=True)
    with colD:
        fig5 = px.scatter(df, x='Temperature', y='Weekly_Sales', opacity=0.3,
                           title="Temperature vs Weekly Sales", color_discrete_sequence=['#E91E63'])
        st.plotly_chart(fig5, use_container_width=True)

    st.subheader("Correlation Heatmap")
    corr_cols = ['Weekly_Sales', 'Holiday_Flag', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment']
    corr = df[corr_cols].corr()
    fig6 = px.imshow(corr, text_auto=".2f", color_continuous_scale='RdYlGn', title="Feature Correlation Matrix")
    st.plotly_chart(fig6, use_container_width=True)

# ============================================================
# TAB 3: STORE SEGMENTATION
# ============================================================
with tab3:
    st.subheader("Store Performance Segmentation")

    store_stats = df.groupby('Store').agg(
        Total_Sales=('Weekly_Sales', 'sum'),
        Avg_Weekly_Sales=('Weekly_Sales', 'mean'),
        Max_Sales=('Weekly_Sales', 'max'),
        Min_Sales=('Weekly_Sales', 'min')
    ).reset_index()

    store_stats['Segment'] = pd.qcut(store_stats['Total_Sales'], q=3,
                                      labels=['Low Performing', 'Medium Performing', 'High Performing'])

    fig7 = px.bar(store_stats.sort_values('Total_Sales', ascending=False),
                  x='Store', y='Total_Sales', color='Segment',
                  color_discrete_map={'High Performing': '#4CAF50', 'Medium Performing': '#FF9800', 'Low Performing': '#F44336'},
                  title="Store Segmentation by Total Sales")
    fig7.update_xaxes(type='category')
    st.plotly_chart(fig7, use_container_width=True)

    st.subheader("Top 5 High Performing Stores")
    st.dataframe(store_stats.sort_values('Total_Sales', ascending=False).head(5), use_container_width=True)

# ============================================================
# TAB 4: MODEL PERFORMANCE
# ============================================================
with tab4:
    st.subheader("Model Evaluation")

    m1, m2, m3 = st.columns(3)
    m1.metric("MAE", f"${results['metrics']['mae']:,.2f}", f"vs ${results['baseline_metrics']['mae']:,.2f} baseline", delta_color="inverse")
    m2.metric("RMSE", f"${results['metrics']['rmse']:,.2f}", f"vs ${results['baseline_metrics']['rmse']:,.2f} baseline", delta_color="inverse")
    m3.metric("MAPE", f"{results['metrics']['mape']:.2f}%", f"vs {results['baseline_metrics']['mape']:.2f}% baseline", delta_color="inverse")

    st.caption("Baseline = naive forecast using last week's actual sales (Lag_1). Delta shows how much XGBoost improves over this heuristic.")

    st.markdown("---")
    st.subheader("Actual vs Predicted (First 100 Test Samples)")
    y_test_arr = results['y_test'].values[:100]
    y_pred_arr = results['y_pred'][:100]
    fig8 = go.Figure()
    fig8.add_trace(go.Scatter(y=y_test_arr, mode='lines', name='Actual', line=dict(color='#2196F3')))
    fig8.add_trace(go.Scatter(y=y_pred_arr, mode='lines', name='Predicted', line=dict(color='#F44336', dash='dash')))
    fig8.update_layout(xaxis_title="Week Index", yaxis_title="Weekly Sales ($)")
    st.plotly_chart(fig8, use_container_width=True)

    st.markdown("---")
    st.subheader("Model Comparison")
    st.caption("Multiple algorithms tested on the same train/test split to validate that XGBoost is genuinely the best choice, not just a default pick.")
    colM1, colM2 = st.columns([1, 1])
    with colM1:
        display_comp = results['comparison_df'].copy()
        display_comp['MAE'] = display_comp['MAE'].apply(lambda x: f"${x:,.2f}")
        display_comp['RMSE'] = display_comp['RMSE'].apply(lambda x: f"${x:,.2f}")
        display_comp['MAPE'] = display_comp['MAPE'].apply(lambda x: f"{x:.2f}%")
        st.dataframe(display_comp, use_container_width=True, hide_index=True)
    with colM2:
        fig_comp = px.bar(results['comparison_df'].sort_values('MAPE', ascending=False),
                           x='MAPE', y='Model', orientation='h',
                           title="MAPE by Model (lower is better)",
                           color='Model',
                           color_discrete_sequence=['#F44336', '#FF9800', '#9C27B0', '#4CAF50'])
        fig_comp.update_layout(showlegend=False)
        st.plotly_chart(fig_comp, use_container_width=True)

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Residual Distribution")
        residuals = results['y_test'].values - results['y_pred']
        fig9 = px.histogram(residuals, nbins=40, title="Prediction Errors (Actual − Predicted)",
                             color_discrete_sequence=['#9C27B0'])
        fig9.update_layout(xaxis_title="Residual ($)", showlegend=False)
        st.plotly_chart(fig9, use_container_width=True)
        st.caption("Roughly centered around zero indicates no strong systematic bias in the model.")
    with col2:
        st.subheader("Feature Importance")
        fig10 = px.bar(results['importance_df'], x='Importance', y='Feature', orientation='h',
                       color_discrete_sequence=['#9C27B0'])
        fig10.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig10, use_container_width=True)

# ============================================================
# TAB 5: FORECAST
# ============================================================
with tab5:
    st.subheader("Forecast Future Weeks")

    stores = sorted(df_feat['Store'].unique())
    colX, colY = st.columns(2)
    with colX:
        selected_store = st.selectbox("Select Store", stores)
    with colY:
        n_weeks_ahead = st.slider("Weeks to forecast", 1, 12, 4)

    store_hist = df_feat[df_feat['Store'] == selected_store].sort_values('Date').reset_index(drop=True)
    model = results['model']
    feature_cols = results['feature_cols']

    working = store_hist.copy()
    future_rows = []
    last_row = working.iloc[-1].copy()
    recent_sales = list(working['Weekly_Sales'].tail(4).values)

    for i in range(n_weeks_ahead):
        next_date = last_row['Date'] + pd.Timedelta(weeks=1)
        new_row = last_row.copy()
        new_row['Date'] = next_date
        new_row['Month'] = next_date.month
        new_row['Year'] = next_date.year
        new_row['Week'] = next_date.isocalendar()[1]
        new_row['Lag_1'] = recent_sales[-1]
        new_row['Lag_2'] = recent_sales[-2] if len(recent_sales) >= 2 else recent_sales[-1]
        new_row['Lag_4'] = recent_sales[-4] if len(recent_sales) >= 4 else recent_sales[0]
        new_row['Rolling_Mean_4'] = np.mean(recent_sales[-4:])
        new_row['Rolling_Std_4'] = np.std(recent_sales[-4:]) if len(recent_sales) > 1 else 0
        new_row['Sales_Growth'] = (recent_sales[-1] - recent_sales[-2]) / recent_sales[-2] if len(recent_sales) >= 2 else 0

        X_new = pd.DataFrame([new_row[feature_cols]])
        pred_sales = model.predict(X_new)[0]

        new_row['Weekly_Sales'] = pred_sales
        future_rows.append({'Date': next_date, 'Weekly_Sales': pred_sales, 'Type': 'Forecast'})
        recent_sales.append(pred_sales)
        last_row = new_row

    hist_plot = store_hist[['Date', 'Weekly_Sales']].tail(20).copy()
    hist_plot['Type'] = 'Actual'
    future_df = pd.DataFrame(future_rows)
    combined = pd.concat([hist_plot, future_df], ignore_index=True)

    fig11 = px.line(combined, x='Date', y='Weekly_Sales', color='Type', markers=True,
                     title=f"Store {selected_store} — Sales Forecast (Next {n_weeks_ahead} Weeks)",
                     color_discrete_map={'Actual': '#2196F3', 'Forecast': '#F44336'})
    st.plotly_chart(fig11, use_container_width=True)

    st.subheader("Forecast Table")
    display_df = future_df[['Date', 'Weekly_Sales']].copy()
    display_df['Weekly_Sales'] = display_df['Weekly_Sales'].apply(lambda x: f"${x:,.2f}")
    display_df.columns = ['Week Starting', 'Predicted Sales']
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    csv = future_df[['Date', 'Weekly_Sales']].to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Forecast as CSV", csv, f"store_{selected_store}_forecast.csv", "text/csv")

    st.caption("⚠️ Forecasts assume recent trend patterns continue; accuracy decreases the further out you forecast, since predictions are built on prior predictions (recursive forecasting).")
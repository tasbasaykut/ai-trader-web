import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use('Agg')  # Bulut sunucusu (headless) uyumu
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler

# --- SAYFA YAPILANDIRMASI ---
st.set_page_config(page_title="A.S.T. Ultra Finansal Terminal", layout="wide")

# --- GELİŞMİŞ CSS (Karanlık Mod & Okunabilirlik) ---
st.markdown("""
    <style>
    .stApp { background-color: #0E1117; }
    div[data-testid="stMetric"] {
        background-color: #1E1E26;
        border: 1px solid #3E3E4E;
        padding: 15px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.4);
    }
    div[data-testid="stMetric"] > div > label { color: #BFC5D3 !important; font-weight: 500; }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] { color: #FFFFFF !important; font-weight: 700; }
    .stTabs [data-baseweb="tab"] { color: #BFC5D3; }
    .stTabs [aria-selected="true"] { color: #FFFFFF !important; border-bottom-color: #FF4B4B !important; }
    </style>
    """, unsafe_allow_html=True)

st.title("🛡️ A.S.T. Ultra Hibrit Yatırım Terminali")
st.caption("Teknik Analiz + XGBoost AI + Risk Yönetimi + Şirket Büyüme Analizi")

# --- YAN PANEL (SIDEBAR) ---
st.sidebar.header("🕹️ Kontrol Paneli")
hisse = st.sidebar.text_input("Hisse Kodu (Örn: THYAO.IS)", value="THYAO.IS")
zaman_secenekleri = {"1 Yıl": "1y", "2 Yıl": "2y", "5 Yıl": "5y"}
secilen_period = zaman_secenekleri[st.sidebar.selectbox("Analiz Dönemi", list(zaman_secenekleri.keys()), index=1)]
nakit = st.sidebar.number_input("Başlangıç Sermayesi (TL)", value=1000)

st.sidebar.divider()
st.sidebar.subheader("Algoritma Ayarları")
sma_k = st.sidebar.slider("Kısa SMA", 5, 30, 20)
sma_u = st.sidebar.slider("Uzun SMA", 30, 100, 50)


# --- FONKSİYONLAR ---

@st.cache_data
def verileri_hazirla(ticker, period, _sma_k, _sma_u):
    df = yf.download(ticker, period=period, interval="1d")
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    if df.empty: return None

    # Teknik Göstergeler
    df['SMA_K'] = df['Close'].rolling(_sma_k).mean()
    df['SMA_U'] = df['Close'].rolling(_sma_u).mean()
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['RSI'] = 100 - (100 / (1 + (gain / loss)))
    df['Volatilite'] = df['Close'].pct_change().rolling(10).std()
    df['Ret_Lag1'] = df['Close'].pct_change()
    df['Target_Return'] = df['Close'].pct_change().shift(-1)
    return df.dropna()


@st.cache_data
def sirket_temellerini_getir(ticker):
    s = yf.Ticker(ticker)
    return s.financials, s.info


def model_calistir(df):
    features = ['RSI', 'Volatilite', 'Ret_Lag1']
    X = df[features]
    y = df['Target_Return']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42)
    model.fit(X_scaled, y)
    df['AI_Pred_Return'] = model.predict(X_scaled)
    return model, scaler, features


# --- ANA UYGULAMA MANTIĞI ---
try:
    data = verileri_hazirla(hisse, secilen_period, sma_k, sma_u)
    financials, info = sirket_temellerini_getir(hisse)

    if data is not None:
        model, scaler, feature_cols = model_calistir(data)

        # 1. Strateji ve Kar/Zarar
        data['Sinyal'] = np.where((data['SMA_K'] > data['SMA_U']) & (data['AI_Pred_Return'] > 0), 1, 0)
        data['Market_Cum'] = (1 + data['Close'].pct_change()).cumprod() * nakit
        data['Strategy_Cum'] = (1 + (data['Close'].pct_change() * data['Sinyal'].shift(1))).cumprod() * nakit

        # Risk Metrikleri (MDD)
        data['Strategy
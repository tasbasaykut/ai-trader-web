import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler

# --- 1. SAYFA YAPILANDIRMASI ---
st.set_page_config(page_title="A.S.T. Ultra Terminal v14", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #0E1117; }
    div[data-testid="stMetric"] { background-color: #1E1E26; border: 1px solid #3E3E4E; padding: 15px; border-radius: 12px; }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] { color: #FFFFFF !important; font-weight: 700; }
    </style>
    """, unsafe_allow_html=True)

st.title("🛡️ A.S.T. Ultra Hibrit v14: Gelişmiş Teknik Katman")

# --- 2. GELİŞMİŞ VERİ MOTORU ---

@st.cache_data
def tum_verileri_hazirla(ticker, period, _sma_k, _sma_u):
    df = yf.download(ticker, period=period, interval="1d")
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    if df.empty: return None, None, None
    
    s = yf.Ticker(ticker)
    finansallar, info = s.financials, s.info
    
    # --- YENİ İNDİKATÖRLER ---
    # 1. Bollinger Bantları (20 günlük)
    df['BB_Mid'] = df['Close'].rolling(window=20).mean()
    df['BB_Std'] = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Mid'] + (df['BB_Std'] * 2)
    df['BB_Lower'] = df['BB_Mid'] - (df['BB_Std'] * 2)
    
    # 2. RSI
    delta = df['Close'].diff()
    df['RSI'] = 100 - (100 / (1 + (delta.where(delta > 0, 0).rolling(14).mean() / 
                                  -delta.where(delta < 0, 0).rolling(14).mean())))
    
    # 3. Hacim Değişimi (Volume)
    df['Vol_Change'] = df['Volume'].pct_change()
    
    # Mevcutlar
    df['SMA_K'] = df['Close'].rolling(_sma_k).mean()
    df['SMA_U'] = df['Close'].rolling(_sma_u).mean()
    df['Volatility'] = df['Close'].pct_change().rolling(10).std()
    df['Ret'] = df['Close'].pct_change()
    df['Target'] = df['Close'].pct_change().shift(-1)
    
    return df.dropna(), finansallar, info

def model_egit_ve_tahmin(df):
    # Yeni eklenen özellikler modelin karar mekanizmasına eklendi
    features = ['RSI', 'Volatility', 'Ret', 'BB_Upper', 'BB_Lower', 'Vol_Change']
    X = df[features]
    y = df['Target']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42)
    model.fit(X_scaled, y)
    
    df['AI_Pred'] = model.predict(X_scaled)
    return model, scaler, features

# --- 3. SIDEBAR ---
hisse = st.sidebar.text_input("Hisse Sembolü", value="THYAO.IS")
period = st.sidebar.selectbox("Analiz Dönemi", ["1y", "2y", "5y"], index=1)
nakit = st.sidebar.number_input("Başlangıç Bakiyesi", value=1000)

# --- 4. ANA AKIŞ ---
try:
    data, financials, info = tum_verileri_hazirla(hisse, period, 20, 50)

    if data is not None:
        model, scaler, f_list = model_egit_ve_tahmin(data)
        
        # Hibrit Karar Mekanizması
        data['Sinyal'] = np.where((data['SMA_K'] > data['SMA_U']) & (data['AI_Pred'] > 0), 1, 0)
        data['Strategy_Cum'] = (1 + (data['Close'].pct_change() * data['Sinyal'].shift(1))).cumprod() * nakit
        data['Market_Cum'] = (1 + data['Close'].pct_change()).cumprod() * nakit

        # Üst Metrikler
        c1, c2, c3 = st.columns(3)
        c1.metric("Son Fiyat", f"{data['Close'].iloc[-1]:.2f} TL")
        c2.metric("Robot Final", f"{data['Strategy_Cum'].iloc[-1]:.2f} TL")
        peak = data['Strategy_Cum'].cummax()
        mdd = ((data['Strategy_Cum'] - peak) / peak).min()
        c3.metric("Max Risk (MDD)", f"%{mdd*100:.2f}")

        # --- GÖRSELLEŞTİRME SEKMELERİ ---
        t1, t2, t3, t4 = st.tabs(["📈 Robot Performans", "🔍 Teknik Analiz Grafikleri", "🤖 AI Model Girdileri", "🏢 Şirket Büyümesi"])
        
        with t1:
            st.subheader("Robot vs Piyasa Kıyaslaması")
            st.line_chart(data[['Market_Cum', 'Strategy_Cum']])

        with t2:
            st.subheader("Bollinger Bantları ve Fiyat")
            # Fiyat ve BB Bantlarını birlikte gösterelim
            st.line_chart(data[['Close', 'BB_Upper', 'BB_Lower', 'BB_Mid']])
            
            st.subheader("İşlem Hacmi (Volume)")
            st.bar_chart(data['Volume'])

        with t3:
            st.subheader("RSI (Göreli Güç Endeksi)")
            st.area_chart(data['RSI'])
            st.caption("RSI 70 üzeri aşırı alım, 30 altı aşırı satım bölgesidir.")
            
            st.subheader("AI Tahmin Doğruluğu")
            st.bar_chart(pd.DataFrame({'Gerçek': data['Target'].tail(15), 'AI': data['AI_Pred'].tail(15)}))

        with t4:
            st.subheader("Finansal Röntgen")
            if financials is not None and not financials.empty:
                st.bar_chart(financials.loc[['Total Revenue', 'Net Income']].T)
            else: st.info("Veri kısıtlı.")

except Exception as e:
    st.error(f"Hata: {e}")

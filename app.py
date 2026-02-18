import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler

# --- SAYFA YAPILANDIRMASI ---
st.set_page_config(page_title="A.S.T. Pro AI Terminal v9", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #0E1117; }
    div[data-testid="stMetric"] { background-color: #1E1E26; border-radius: 12px; padding: 15px; }
    </style>
    """, unsafe_allow_html=True)

st.title("🚀 A.S.T. Hibrit v9: Temel Veri Destekli AI")

# --- SIDEBAR ---
hisse = st.sidebar.text_input("Hisse Kodu", value="THYAO.IS")
zaman_secenekleri = {"2 Yıl": "2y", "5 Yıl": "5y"}
secilen_period = zaman_secenekleri[st.sidebar.selectbox("Analiz Dönemi", list(zaman_secenekleri.keys()), index=0)]

# --- FONKSİYONLAR ---

@st.cache_data
def gelismis_veri_hazirla(ticker, period):
    # 1. Teknik Verileri Çek
    df = yf.download(ticker, period=period, interval="1d")
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    
    # 2. Temel Verileri Çek (Bilanço)
    sirket = yf.Ticker(ticker)
    try:
        fin = sirket.financials.T.sort_index()
        fin.index = pd.to_datetime(fin.index).year # Yıllara çevir
        # Yıllık Büyüme Oranları
        fin['Rev_Growth'] = fin['Total Revenue'].pct_change()
        fin['Net_Margin'] = fin['Net Income'] / fin['Total Revenue']
    except:
        fin = pd.DataFrame()

    # 3. VERİ BİRLEŞTİRME (Mühendislik kısmı)
    df['Year'] = df.index.year
    if not fin.empty:
        # Her güne o yılın büyüme verisini eşle
        df['Growth_Feature'] = df['Year'].map(fin['Rev_Growth']).fillna(method='ffill').fillna(0)
        df['Margin_Feature'] = df['Year'].map(fin['Net_Margin']).fillna(method='ffill').fillna(0)
    else:
        df['Growth_Feature'] = 0
        df['Margin_Feature'] = 0

    # Teknik Göstergeler
    df['SMA_20'] = df['Close'].rolling(20).mean()
    df['RSI'] = 100 - (100 / (1 + (df['Close'].diff().where(df['Close'].diff() > 0, 0).rolling(14).mean() / 
                                  -df['Close'].diff().where(df['Close'].diff() < 0, 0).rolling(14).mean())))
    df['Volatilite'] = df['Close'].pct_change().rolling(10).std()
    df['Target_Return'] = df['Close'].pct_change().shift(-1)
    
    return df.dropna()

def model_egit_v9(df):
    # Model artık hem teknik hem temel verilere bakıyor!
    features = ['RSI', 'Volatilite', 'Growth_Feature', 'Margin_Feature']
    X = df[features]
    y = df['Target_Return']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # XGBoost parametrelerini daha hassas yaptık
    model = XGBRegressor(n_estimators=200, learning_rate=0.03, max_depth=5, random_state=42)
    model.fit(X_scaled, y)
    
    df['AI_Pred'] = model.predict(X_scaled)
    return model, scaler, features

# --- UYGULAMA AKIŞI ---
try:
    data = gelismis_veri_hazirla(hisse, secilen_period)
    model, scaler, feature_list = model_egit_v9(data)
    
    # Metrikler
    st.subheader("📊 Performans & AI Tahmin")
    c1, c2, c3 = st.columns(3)
    c1.metric("Güncel Fiyat", f"{data['Close'].iloc[-1]:.2f} TL")
    
    son_input = scaler.transform(data[feature_list].tail(1))
    yarinki_getiri = model.predict(son_input)[0]
    c2.metric("AI Beklenen Değişim", f"%{yarinki_getiri*100:.2f}")
    
    # Şirket Büyüme Bilgisi
    c3.metric("Şirket Gelir Büyümesi", f"%{data['Growth_Feature'].iloc[-1]*100:.1f}")

    # Grafik
    tab1, tab2 = st.tabs(["Fiyat Tahmini", "Temel Veri Etkisi"])
    with tab1:
        st.line_chart(data[['Close']])
    with tab2:
        st.write("Modelin kullandığı Temel Büyüme Verisi (Zaman Serisi)")
        st.line_chart(data['Growth_Feature'])

    if yarinki_getiri > 0.005:
        st.success("🔥 **AI ONAYI:** Şirket büyümesi ve teknik veriler yükselişi destekliyor.")
    else:
        st.warning("⚖️ **BEKLE:** Tahmin edilen getiri risk sınırının altında.")

except Exception as e:
    st.error(f"Hata: {e}")

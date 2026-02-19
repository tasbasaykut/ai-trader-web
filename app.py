import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from xgboost import XGBRegressor, XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

# --- 1. SAYFA YAPILANDIRMASI ---
st.set_page_config(page_title="A.S.T. Ultra Terminal v16.1", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #0E1117; }
    div[data-testid="stMetric"] { background-color: #1E1E26; border: 1px solid #3E3E4E; padding: 15px; border-radius: 12px; }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] { color: #FFFFFF !important; font-weight: 700; }
    .stTabs [data-baseweb="tab"] { color: #BFC5D3; }
    .stTabs [aria-selected="true"] { color: #FFFFFF !important; border-bottom-color: #FF4B4B !important; }
    </style>
    """, unsafe_allow_html=True)

st.title("🔮 A.S.T. Ultra v16.1: Profesyonel Tahmin & Analiz")
st.caption("Gelişmiş Görselleştirme + Tarih Doğrulama + Multi-Target AI")

# --- 2. VERİ VE MODEL MOTORU ---
@st.cache_data
def v16_veri_hazirla(ticker, period):
    df = yf.download(ticker, period=period, interval="1d")
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    if df.empty: return None, None, None
    
    s = yf.Ticker(ticker)
    
    # İndikatörler
    df['SMA_20'] = df['Close'].rolling(20).mean()
    df['BB_Std'] = df['Close'].rolling(20).std()
    df['BB_Upper'] = df['SMA_20'] + (df['BB_Std'] * 2)
    df['BB_Lower'] = df['SMA_20'] - (df['BB_Std'] * 2)
    
    delta = df['Close'].diff()
    df['RSI'] = 100 - (100 / (1 + (delta.where(delta > 0, 0).rolling(14).mean() / 
                                  -delta.where(delta < 0, 0).rolling(14).mean())))
    
    df['Volatility'] = df['Close'].pct_change().rolling(10).std()
    
    # HEDEFLER
    df['Target_1d'] = df['Close'].pct_change().shift(-1)
    df['Target_7d'] = (df['Close'].shift(-7) / df['Close']) - 1
    df['Target_Binary'] = (df['Target_1d'] > 0).astype(int)
    
    return df.dropna(), s.financials, s.info

def model_merkezi(df):
    features = ['RSI', 'Volatility', 'BB_Upper', 'BB_Lower', 'Close']
    X = df[features]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Modeller
    m1 = XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=5)
    m1.fit(X_scaled, df['Target_1d'])
    
    m7 = XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=5)
    m7.fit(X_scaled, df['Target_7d'])
    
    mp = XGBClassifier(n_estimators=100, learning_rate=0.05, eval_metric='logloss')
    mp.fit(X_scaled, df['Target_Binary'])
    
    df['AI_Pred_1d'] = m1.predict(X_scaled)
    return m1, m7, mp, scaler, features

# --- 3. ANA AKIŞ ---
hisse = st.sidebar.text_input("Hisse Sembolü", value="THYAO.IS")
nakit = st.sidebar.number_input("Bakiye", value=1000)

try:
    data, financials, info = v16_veri_hazirla(hisse, "2y")

    if data is not None:
        m1, m7, mp, scaler, f_list = model_merkezi(data)
        
        # Tahminleri Al
        last_row = scaler.transform(data[f_list].tail(1))
        p_1d = m1.predict(last_row)[0]
        p_7d = m7.predict(last_row)[0]
        prob = mp.predict_proba(last_row)[0, 1]

        # --- TARİH HESAPLAMALARI ---
        son_veri_tarihi = data.index[-1].strftime('%d.%m.%Y')
        tahmin_hedefi = (data.index[-1] + pd.Timedelta(days=1)).strftime('%d.%m.%Y')

        # ÜST METRİKLER
        st.subheader(f"🏁 Performans Özet Tablosu ({son_veri_tarihi} Verileriyle)")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Son Kapanış Fiyatı", f"{data['Close'].iloc[-1]:.2f} TL")
        c2.metric("Tahmin Hedefi", tahmin_hedefi)
        c3.metric("Yükseliş Güveni", f"%{prob*100:.1f}")
        c4.metric("Veri Güncelliği", son_veri_tarihi)

        # SEKMELER
        t1, t2, t3, t4, t5 = st.tabs(["🔮 Gelecek Kahini", "🤖 AI Model Performansı", "📈 Teknik Analiz", "🏢 Şirket Röntgeni", "🛡️ Risk Yönetimi"])
        
        with t1:
            st.subheader(f"🚀 {hisse} 7 Günlük Fiyat Projeksiyonu")
            
            # Gelecek Verisi Oluşturma
            future_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=7)
            current_price = data['Close'].iloc[-1]
            
            # Hata payını volatiliteye göre hesaplayalım
            vol = data['Volatility'].iloc[-1]
            future_prices = [current_price * (1 + (p_7d/7) * i) for i in range(1, 8)]
            upper_bound = [p * (1 + vol * np.sqrt(i)) for i

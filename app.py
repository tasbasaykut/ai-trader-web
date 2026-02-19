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
st.set_page_config(page_title="A.S.T. Ultra Terminal v13", layout="wide")

# Metrik kutularının okunabilirliği için geliştirilmiş CSS
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
st.caption("Teknik Analiz + Temel Veri + Yapay Zeka + Risk Yönetimi")

# --- 2. GELİŞMİŞ FONKSİYONLAR ---

@st.cache_data
def tum_verileri_hazirla(ticker, period, _sma_k, _sma_u):
    # Teknik + Temel Veriler
    df = yf.download(ticker, period=period, interval="1d")
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    if df.empty: return None, None, None
    
    s = yf.Ticker(ticker)
    finansallar, info = s.financials, s.info
    
    # İndikatörler
    df['SMA_K'] = df['Close'].rolling(_sma_k).mean()
    df['SMA_U'] = df['Close'].rolling(_sma_u).mean()
    delta = df['Close'].diff()
    df['RSI'] = 100 - (100 / (1 + (delta.where(delta > 0, 0).rolling(14).mean() / 
                                  -delta.where(delta < 0, 0).rolling(14).mean())))
    df['Vol'] = df['Close'].pct_change().rolling(10).std()
    df['Ret'] = df['Close'].pct_change()
    df['Target'] = df['Close'].pct_change().shift(-1)
    
    return df.dropna(), finansallar, info

def model_egit_ve_tahmin(df):
    # Haber verisi (Sentiment) çıkarıldı
    features = ['RSI', 'Vol', 'Ret']
    X = df[features]
    y = df['Target']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42)
    model.fit(X_scaled, y)
    df['AI_Pred'] = model.predict(X_scaled)
    return model, scaler, features

# --- 3. SIDEBAR VE KONTROL ---
hisse = st.sidebar.text_input("Hisse Sembolü (BIST için .IS ekleyin)", value="THYAO.IS")
period = st.sidebar.selectbox("Analiz Dönemi", ["1y", "2y", "5y"], index=1)
nakit = st.sidebar.number_input("Başlangıç Bakiyesi (TL)", value=1000)

sma_k = st.sidebar.slider("Kısa SMA", 5, 30, 20)
sma_u = st.sidebar.slider("Uzun SMA", 30, 100, 50)

# --- 4. ANA AKIŞ ---
try:
    data, financials, info = tum_verileri_hazirla(hisse, period, sma_k, sma_u)

    if data is not None:
        model, scaler, f_list = model_egit_ve_tahmin(data)
        
        # Strateji ve Kar/Zarar
        data['Sinyal'] = np.where((data['SMA_K'] > data['SMA_U']) & (data['AI_Pred'] > 0), 1, 0)
        data['Strategy_Cum'] = (1 + (data['Close'].pct_change() * data['Sinyal'].shift(1))).cumprod() * nakit
        data['Market_Cum'] = (1 + data['Close'].pct_change()).cumprod() * nakit

        # ÜST METRİKLER
        st.subheader("🏁 Performans & Risk Özeti")
        c1, c2, c3 = st.columns(3)
        c1.metric("Son Fiyat", f"{data['Close'].iloc[-1]:.2f} TL")
        c2.metric("Robot Final", f"{data['Strategy_Cum'].iloc[-1]:.2f} TL", 
                  f"{((data['Strategy_Cum'].iloc[-1]/data['Market_Cum'].iloc[-1])-1)*100:.1f}% vs Piyasa")
        
        peak = data['Strategy_Cum'].cummax()
        mdd = ((data['Strategy_Cum'] - peak) / peak).min()
        c3.metric("Max Zarar (Risk)", f"%{mdd*100:.2f}")

        # SEKMELER
        t1, t2, t3, t4 = st.tabs(["📈 Getiri", "🔍 Teknik & Risk", "🤖 AI Tahmin Analizi", "🏢 Şirket Büyümesi"])
        
        with t1:
            st.subheader("Kümülatif Getiri Kıyaslaması")
            st.line_chart(data[['Market_Cum', 'Strategy_Cum']])

        with t2:
            st.subheader("Anlık Zarar (Drawdown)")
            st.area_chart(((data['Strategy_Cum'] - peak) / peak))
            st.subheader("Teknik Trend")
            st.line_chart(data[['Close', 'SMA_K', 'SMA_U']])

        with t3:
            st.subheader("AI Tahminleri ve Model Performansı")
            st.bar_chart(pd.DataFrame({'Gerçek': data['Target'].tail(15), 'AI': data['AI_Pred'].tail(15)}))
            
            # Yarın Tahmini
            yarinki_input = scaler.transform(data[f_list].tail(1))
            y_pred = model.predict(yarinki_input)[0]
            if y_pred > 0.005: st.success(f"🚀 AI Yükseliş Bekliyor: %{y_pred*100:.2f}")
            else: st.info(f"⚖️ Robot Temkinli: %{y_pred*100:.2f}")

        with t4:
            st.subheader(f"🏢 {info.get('longName', hisse)} Temel Analiz")
            if financials is not None and not financials.empty:
                yillik = financials.loc[['Total Revenue', 'Net Income']].T.sort_index()
                st.bar_chart(yillik)
                rev_growth = yillik['Total Revenue'].pct_change().iloc[-1] * 100
                st.metric("Son Yıl Gelir Büyümesi", f"%{rev_growth:.2f}")
            else: st.info("Finansal veriler çekilemedi.")

except Exception as e:
    st.error(f"Sistem Hatası: {e}")

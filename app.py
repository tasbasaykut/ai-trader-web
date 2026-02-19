import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from xgboost import XGBRegressor, XGBClassifier
from sklearn.preprocessing import StandardScaler

# --- 1. SAYFA YAPILANDIRMASI ---
st.set_page_config(page_title="A.S.T. Ultra Terminal v15", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #0E1117; }
    div[data-testid="stMetric"] { background-color: #1E1E26; border: 1px solid #3E3E4E; padding: 15px; border-radius: 12px; }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] { color: #FFFFFF !important; font-weight: 700; }
    .stProgress > div > div > div > div { background-color: #FF4B4B; }
    </style>
    """, unsafe_allow_html=True)

st.title("🔮 A.S.T. Ultra v15: Gelecek Kahini & İhtimal Katmanı")

# --- 2. GELİŞMİŞ VERİ MOTORU ---
@st.cache_data
def v15_veri_hazirla(ticker, period, _sma_k, _sma_u):
    df = yf.download(ticker, period=period, interval="1d")
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    if df.empty: return None, None, None
    
    s = yf.Ticker(ticker)
    
    # Teknik Göstergeler (Model Girdileri)
    df['BB_Mid'] = df['Close'].rolling(window=20).mean()
    df['BB_Std'] = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Mid'] + (df['BB_Std'] * 2)
    df['BB_Lower'] = df['BB_Mid'] - (df['BB_Std'] * 2)
    
    delta = df['Close'].diff()
    df['RSI'] = 100 - (100 / (1 + (delta.where(delta > 0, 0).rolling(14).mean() / 
                                  -delta.where(delta < 0, 0).rolling(14).mean())))
    
    df['SMA_K'] = df['Close'].rolling(_sma_k).mean()
    df['SMA_U'] = df['Close'].rolling(_sma_u).mean()
    df['Volatility'] = df['Close'].pct_change().rolling(10).std()
    df['Volume_Change'] = df['Volume'].pct_change()

    # HEDEFLER (Modellerin tahmin edeceği şeyler)
    df['Target_1d'] = df['Close'].pct_change().shift(-1) # Yarın ne olur?
    df['Target_7d'] = (df['Close'].shift(-7) / df['Close']) - 1 # 1 Hafta sonra ne olur?
    df['Target_Binary'] = (df['Target_1d'] > 0).astype(int) # Yarın artar mı? (1/0)
    
    return df.dropna(), s.financials, s.info

def gelecek_motoru(df):
    features = ['RSI', 'Volatility', 'BB_Upper', 'BB_Lower', 'Volume_Change', 'Close']
    X = df[features]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 1. Kısa Vade Modeli (Regressor)
    model_1d = XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=5)
    model_1d.fit(X_scaled, df['Target_1d'])
    
    # 2. Uzun Vade Modeli (Regressor)
    model_7d = XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=5)
    model_7d.fit(X_scaled, df['Target_7d'])
    
    # 3. İhtimal Modeli (Classifier)
    model_prob = XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=5, use_label_encoder=False, eval_metric='logloss')
    model_prob.fit(X_scaled, df['Target_Binary'])
    
    # Tahminleri DataFrame'e ekle
    df['Pred_1d'] = model_1d.predict(X_scaled)
    df['Pred_7d'] = model_7d.predict(X_scaled)
    df['Prob_Up'] = model_prob.predict_proba(X_scaled)[:, 1]
    
    return model_1d, model_7d, model_prob, scaler, features

# --- 3. ANA AKIŞ ---
hisse = st.sidebar.text_input("Hisse Sembolü", value="THYAO.IS")
period = st.sidebar.selectbox("Analiz Dönemi", ["2y", "5y"], index=0)
nakit = st.sidebar.number_input("Bakiye", value=1000)

try:
    data, financials, info = v15_veri_hazirla(hisse, period, 20, 50)

    if data is not None:
        m1, m7, m_p, scaler, f_list = gelecek_motoru(data)
        
        # Son veriyi al ve yarını tahmin et
        last_row = scaler.transform(data[f_list].tail(1))
        p_1d = m1.predict(last_row)[0]
        p_7d = m7.predict(last_row)[0]
        prob = m_p.predict_proba(last_row)[0, 1]

        # --- ARAYÜZ ---
        st.header(f"🔮 {hisse} İçin Gelecek Tahminleri")
        
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Yarın Beklenen Değişim", f"%{p_1d*100:.2f}")
            st.caption("XGBoost 24 Saatlik Projeksiyon")
        with c2:
            st.metric("7 Günlük Beklenen Değişim", f"%{p_7d*100:.2f}")
            st.caption("1 Haftalık Kümülatif Trend Tahmini")
        with c3:
            st.write("**Yükseliş İhtimali**")
            st.progress(float(prob))
            st.write(f"Sistemin Güveni: %{prob*100:.1f}")

        # SEKMELER
        t1, t2, t3 = st.tabs(["📈 Tahmin Grafikleri", "🔍 Teknik İndikatörler", "🏢 Şirket Verileri"])
        
        with t1:
            st.subheader("Fiyat ve Tahmin Edilen Trend")
            # Gelecek projeksiyonunu basitçe çizelim
            future_dates = pd.date_range(start=data.index[-1], periods=8, freq='D')
            future_prices = [data['Close'].iloc[-1]]
            for i in range(1, 8):
                # Basit bir doğrusal projeksiyon: (Günlük tahminin kümülatif etkisi)
                future_prices.append(future_prices[-1] * (1 + p_1d))
            
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(data.index[-30:], data['Close'].tail(30), label="Gerçek Fiyat", color='white')
            ax.plot(future_dates, future_prices, label="AI Projeksiyonu", linestyle='--', color='cyan')
            ax.set_facecolor('#0E1117')
            fig.patch.set_facecolor('#0E1117')
            plt.legend()
            st.pyplot(fig)

        with t2:
            col_l, col_r = st.columns(2)
            with col_l:
                st.write("**RSI Grafiği**")
                st.area_chart(data['RSI'].tail(100))
            with col_r:
                st.write("**Bollinger Bantları**")
                st.line_chart(data[['Close', 'BB_Upper', 'BB_Lower']].tail(100))
            
            st.write("**İşlem Hacmi**")
            st.bar_chart(data['Volume'].tail(100))

        with t3:
            if financials is not None and not financials.empty:
                st.bar_chart(financials.loc[['Total Revenue', 'Net Income']].T)

except Exception as e:
    st.error(f"Tahmin Motoru Hatası: {e}")

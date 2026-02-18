import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# --- 1. SAYFA VE TEMA AYARLARI ---
st.set_page_config(page_title="A.S.T. Terminal v11.2", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #0E1117; }
    div[data-testid="stMetric"] {
        background-color: #1E1E26;
        border: 1px solid #3E3E4E;
        padding: 15px;
        border-radius: 12px;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] { color: #FFFFFF !important; font-weight: 700; }
    </style>
    """, unsafe_allow_html=True)

st.title("🛡️ A.S.T. Ultra Hibrit v11.2")
st.caption("Savunmacı NLP Katmanı + XGBoost + Finansal Röntgen")

# --- 2. GELİŞMİŞ HABER MOTORU (KeyError Çözümü) ---

def haber_duygusu_analizi_v2(ticker):
    """Haberleri çekerken anahtar hatalarını (KeyError) kontrol eder."""
    try:
        s = yf.Ticker(ticker)
        haberler = s.news
        if not haberler or len(haberler) == 0:
            return 0.0, ["Haber verisi şu an sunucudan gelmiyor."]
        
        analyzer = SentimentIntensityAnalyzer()
        skorlar = []
        detaylar = []
        
        for h in haberler[:5]:
            # SAVUNMACI YAKLAŞIM: title yoksa text'e, o da yoksa boşluğa bak
            baslik = h.get('title', h.get('text', h.get('headline', 'Başlıksız Haber')))
            
            vs = analyzer.polarity_scores(baslik)
            skor = vs['compound']
            skorlar.append(skor)
            detaylar.append(f"🔹 {baslik} (Puan: {skor})")
            
        ortalama = sum(skorlar) / len(skorlar) if skorlar else 0.0
        return ortalama, detaylar
    except Exception as e:
        return 0.0, [f"Sistem Uyarısı: {str(e)}"]

@st.cache_data
def verileri_komple_hazirla(ticker, period, _sma_k, _sma_u):
    df = yf.download(ticker, period=period, interval="1d")
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    if df.empty: return None, None, None
    
    s = yf.Ticker(ticker)
    fin, inf = s.financials, s.info
    
    df['SMA_K'] = df['Close'].rolling(_sma_k).mean()
    df['SMA_U'] = df['Close'].rolling(_sma_u).mean()
    # RSI Hesaplama
    delta = df['Close'].diff()
    df['RSI'] = 100 - (100 / (1 + (delta.where(delta > 0, 0).rolling(14).mean() / 
                                  -delta.where(delta < 0, 0).rolling(14).mean())))
    df['Vol'] = df['Close'].pct_change().rolling(10).std()
    df['Ret'] = df['Close'].pct_change()
    df['Target'] = df['Close'].pct_change().shift(-1)
    
    return df.dropna(), fin, inf

# --- 3. SIDEBAR VE AKIŞ ---
hisse = st.sidebar.text_input("Hisse Sembolü", value="THYAO.IS")
period = st.sidebar.selectbox("Dönem", ["1y", "2y", "5y"], index=1)
nakit = st.sidebar.number_input("Bakiye (TL)", value=1000)

try:
    data, financials, info = verileri_komple_hazirla(hisse, period, 20, 50)
    duygu_skor, haber_list = haber_duygusu_analizi_v2(hisse)

    if data is not None:
        # AI Eğitim (Sentiment Dahil)
        data['Sentiment'] = duygu_skor
        features = ['RSI', 'Vol', 'Sentiment', 'Ret']
        X = data[features]
        y = data['Target']
        
        scaler = StandardScaler()
        X_sc = scaler.fit_transform(X)
        model = XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=5)
        model.fit(X_sc, y)
        data['AI_Pred'] = model.predict(X_sc)

        # Kar/Zarar
        data['Sinyal'] = np.where((data['SMA_K'] > data['SMA_U']) & (data['AI_Pred'] > 0), 1, 0)
        data['Strategy_Cum'] = (1 + (data['Close'].pct_change() * data['Sinyal'].shift(1))).cumprod() * nakit
        data['Market_Cum'] = (1 + data['Close'].pct_change()).cumprod() * nakit

        # ÜST METRİKLER
        st.subheader("🏁 Finansal Özet")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Son Fiyat", f"{data['Close'].iloc[-1]:.2f} TL")
        m2.metric("Robot Getirisi", f"{data['Strategy_Cum'].iloc[-1]:.2f} TL")
        
        # Max Drawdown (MDD) Formülü: $$MDD = \frac{Value - Peak}{Peak}$$
        peak = data['Strategy_Cum'].cummax()
        mdd = ((data['Strategy_Cum'] - peak) / peak).min()
        m3.metric("Max Risk (MDD)", f"%{mdd*100:.2f}")
        
        durum = "Pozitif 🚀" if duygu_skor > 0.05 else "Negatif ⚠️" if duygu_skor < -0.05 else "Nötr 😐"
        m4.metric("NLP Duygu", f"{duygu_skor:.2f}", durum)

        # SEKMELER
        t1, t2, t3, t4 = st.tabs(["📊 Getiri", "🔍 Teknik", "🤖 AI/NLP", "🏢 Büyüme"])
        with t1: st.line_chart(data[['Market_Cum', 'Strategy_Cum']])
        with t2: st.line_chart(data[['Close', 'SMA_K', 'SMA_U']])
        with t3:
            st.write("**AI Duygu Analizi ve Haber Teşhis:**")
            for h in haber_list: st.write(h)
        with t4:
            if financials is not None and not financials.empty:
                st.bar_chart(financials.loc[['Total Revenue', 'Net Income']].T)
            else: st.warning("Finansal veriye ulaşılamadı.")

except Exception as e:
    st.error(f"Genel Hata: {e}")

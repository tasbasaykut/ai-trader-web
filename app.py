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
st.set_page_config(page_title="A.S.T. Ultra Terminal v11", layout="wide")

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

st.title("🛡️ A.S.T. Ultra Hibrit Yatırım Terminali v11")
st.caption("Teknik + Temel + Yapay Zeka + Duygu Analizi + Risk Yönetimi")

# --- 2. YARDIMCI FONKSİYONLAR ---

def haber_duygusu_analizi(ticker):
    try:
        s = yf.Ticker(ticker)
        haberler = s.news
        if not haberler: return 0.0
        analyzer = SentimentIntensityAnalyzer()
        skorlar = [analyzer.polarity_scores(h['title'])['compound'] for h in haberler[:5]]
        return sum(skorlar) / len(skorlar)
    except: return 0.0

@st.cache_data
def tum_verileri_hazirla(ticker, period, _sma_k, _sma_u):
    # Teknik Veriler
    df = yf.download(ticker, period=period, interval="1d")
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    if df.empty: return None, None, None
    
    # Şirket Temelleri
    s = yf.Ticker(ticker)
    finansallar = s.financials
    info = s.info
    
    # Özellik Mühendisliği (Technical Features)
    df['SMA_K'] = df['Close'].rolling(_sma_k).mean()
    df['SMA_U'] = df['Close'].rolling(_sma_u).mean()
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['RSI'] = 100 - (100 / (1 + (gain/loss)))
    df['Volatilite'] = df['Close'].pct_change().rolling(10).std()
    df['Ret_Lag1'] = df['Close'].pct_change()
    
    # Duygu Analizi Entegrasyonu
    duygu = haber_duygusu_analizi(ticker)
    df['Sentiment'] = duygu
    
    # Hedef Değişken
    df['Target_Return'] = df['Close'].pct_change().shift(-1)
    
    return df.dropna(), finansallar, info

def model_egit_ve_tahmin(df):
    features = ['RSI', 'Volatilite', 'Sentiment', 'Ret_Lag1']
    X = df[features]
    y = df['Target_Return']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42)
    model.fit(X_scaled, y)
    
    df['AI_Pred'] = model.predict(X_scaled)
    return model, scaler, features

# --- 3. KONTROL PANELİ (SIDEBAR) ---
hisse = st.sidebar.text_input("Hisse Kodu (Örn: THYAO.IS)", value="THYAO.IS")
zaman_secenekleri = {"1 Yıl": "1y", "2 Yıl": "2y", "5 Yıl": "5y"}
secilen_period = zaman_secenekleri[st.sidebar.selectbox("Analiz Dönemi", list(zaman_secenekleri.keys()), index=1)]
nakit = st.sidebar.number_input("Başlangıç Sermayesi (TL)", value=1000)

st.sidebar.divider()
sma_k = st.sidebar.slider("Kısa SMA", 5, 30, 20)
sma_u = st.sidebar.slider("Uzun SMA", 30, 100, 50)

# --- 4. ANA AKIŞ ---
try:
    data, financials, info = tum_verileri_hazirla(hisse, secilen_period, sma_k, sma_u)
    
    if data is not None:
        model, scaler, feature_cols = model_egit_ve_tahmin(data)
        
        # Kar/Zarar ve Strateji Hesapları
        data['Sinyal'] = np.where((data['SMA_K'] > data['SMA_U']) & (data['AI_Pred'] > 0), 1, 0)
        data['Market_Cum'] = (1 + data['Close'].pct_change()).cumprod() * nakit
        data['Strategy_Cum'] = (1 + (data['Close'].pct_change() * data['Sinyal'].shift(1))).cumprod() * nakit
        
        # MDD Hesaplama
        strategy_peak = data['Strategy_Cum'].cummax()
        data['Strategy_DD'] = (data['Strategy_Cum'] - strategy_peak) / strategy_peak
        mdd_val = data['Strategy_DD'].min()

        # --- ÜST METRİKLER ---
        st.subheader("🏁 Strateji ve Duygu Özeti")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Son Fiyat", f"{data['Close'].iloc[-1]:.2f} TL")
        m2.metric("Robot Getirisi", f"{data['Strategy_Cum'].iloc[-1]:.2f} TL", 
                  f"{((data['Strategy_Cum'].iloc[-1]/data['Market_Cum'].iloc[-1])-1)*100:.1f}% vs Piyasa")
        m3.metric("Max Zarar (MDD)", f"%{mdd_val*100:.2f}")
        
        g_duygu = data['Sentiment'].iloc[-1]
        durum = "Pozitif 🚀" if g_duygu > 0.05 else "Negatif ⚠️" if g_duygu < -0.05 else "Nötr 😐"
        m4.metric("Haber Duygusu", f"{g_duygu:.2f}", durum)

        # --- SEKMELER ---
        t1, t2, t3, t4 = st.tabs(["📈 Performans Kıyaslama", "🔍 Teknik & Risk", "🤖 AI & Sentiment", "🏢 Şirket Büyümesi"])
        
        with t1:
            st.subheader("Robot vs Piyasa (Al-Tut) Kümülatif Getiri")
            st.line_chart(data[['Market_Cum', 'Strategy_Cum']])
            st.caption("Mavi: A.S.T. Hibrit Robot | Gri: Piyasa Getirisi")

        with t2:
            st.subheader("Drawdown (Kayıp Durumu)")
            st.area_chart(data['Strategy_DD'])
            st.subheader("Teknik Trend (SMA & Fiyat)")
            st.line_chart(data[['Close', 'SMA_K', 'SMA_U']])

        with t3:
            st.subheader("XGBoost + Sentiment Tahmin Başarısı")
            compare = pd.DataFrame({'Gerçek': data['Target_Return'].tail(15), 'AI': data['AI_Pred'].tail(15)})
            st.bar_chart(compare)
            
            # Yarın Tahmini
            yarinki_input = scaler.transform(data[feature_cols].tail(1))
            y_pred = model.predict(yarinki_input)[0]
            if y_pred > 0.005: st.success(f"🔥 AI ve Haber Analizi Yarın İçin Yükseliş Bekliyor: %{y_pred*100:.2f}")
            else: st.warning(f"⚖️ Robot Yarın İçin Temkinli: %{y_pred*100:.2f}")

        with t4:
            st.subheader(f"🏢 {info.get('longName', hisse)} Büyüme Analizi")
            st.write(f"**Sektör:** {info.get('sector', 'N/A')} | **P/E:** {info.get('trailingPE', 'N/A')}")
            if financials is not None and not financials.empty:
                yillik = financials.loc[['Total Revenue', 'Net Income']].T.sort_index()
                yillik.index = yillik.index.year
                st.bar_chart(yillik)
                rev_growth = yillik['Total Revenue'].pct_change().iloc[-1] * 100
                st.metric("Son Yıl Gelir Büyümesi", f"%{rev_growth:.2f}")
            else: st.info("Finansal veriler çekilemedi.")

except Exception as e:
    st.error(f"Sistem Hatası: {e}")

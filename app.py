import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') # Bulut sunucusu uyumu için zorunlu
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

# --- 1. SAYFA YAPILANDIRMASI (En başta olmalı) ---
st.set_page_config(page_title="A.S.T. Hibrit Risk Terminali", layout="wide")

# --- 2. GELİŞMİŞ CSS (Karanlık Mod & Okunabilirlik) ---
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

st.title("🛡️ A.S.T. Hibrit Terminali: Risk & Performans")
st.sidebar.header("🕹️ Kontrol Paneli")

# --- 3. YARDIMCI FONKSİYONLAR ---
@st.cache_data
def verileri_hazirla(ticker, period, _sma_k, _sma_u):
    df = yf.download(ticker, period=period, interval="1d")
    if isinstance(df.columns, pd.MultiIndex): 
        df.columns = df.columns.get_level_values(0)
    if df.empty: return None

    df['SMA_K'] = df['Close'].rolling(_sma_k).mean()
    df['SMA_U'] = df['Close'].rolling(_sma_u).mean()
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['RSI'] = 100 - (100 / (1 + (gain/loss)))
    df['Volatilite'] = df['Close'].pct_change().rolling(10).std()
    df['Ret_Lag1'] = df['Close'].pct_change()
    df['Target_Return'] = df['Close'].pct_change().shift(-1)
    return df.dropna()

@st.cache_data
def sirket_temellerini_getir(ticker):
    sirket = yf.Ticker(ticker)
    return sirket.financials, sirket.info

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

# --- 4. SIDEBAR GİRDİLERİ ---
hisse = st.sidebar.text_input("Hisse Kodu (Örn: THYAO.IS)", value="THYAO.IS")
zaman_secenekleri = {"1 Yıl": "1y", "2 Yıl": "2y", "5 Yıl": "5y"}
secilen_period = zaman_secenekleri[st.sidebar.selectbox("Analiz Dönemi", list(zaman_secenekleri.keys()), index=1)]
nakit = st.sidebar.number_input("Başlangıç Sermayesi (TL)", value=1000)

st.sidebar.divider()
sma_k = st.sidebar.slider("Kısa SMA", 5, 30, 20)
sma_u = st.sidebar.slider("Uzun SMA", 30, 100, 50)

# --- 5. ANA UYGULAMA MANTIĞI ---
try:
    data = verileri_hazirla(hisse, secilen_period, sma_k, sma_u)
    gelir_tablosu, sirket_bilgisi = sirket_temellerini_getir(hisse)

    if data is not None:
        model, scaler, feature_cols = model_calistir(data)

        # Strateji Hesapları
        data['Sinyal'] = np.where((data['SMA_K'] > data['SMA_U']) & (data['AI_Pred_Return'] > 0), 1, 0)
        data['Market_Cum'] = (1 + data['Close'].pct_change()).cumprod() * nakit
        data['Strategy_Cum'] = (1 + (data['Close'].pct_change() * data['Sinyal'].shift(1))).cumprod() * nakit

        # MDD (Max Drawdown)
        data['Strategy_Peak'] = data['Strategy_Cum'].cummax()
        data['Strategy_DD'] = (data['Strategy_Cum'] - data['Strategy_Peak']) / data['Strategy_Peak']
        mdd_strategy = data['Strategy_DD'].min()
        
        data['Market_Peak'] = data['Market_Cum'].cummax()
        data['Market_DD'] = (data['Market_Cum'] - data['Market_Peak']) / data['Market_Peak']
        mdd_market = data['Market_DD'].min()

        # --- ARAYÜZÜ OLUŞTUR ---
        st.subheader("🏁 Performans & Risk Özeti")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Son Fiyat", f"{data['Close'].iloc[-1]:.2f} TL")
        c2.metric("Robot Final", f"{data['Strategy_Cum'].iloc[-1]:.2f} TL")
        c3.metric("Robot Max Zarar", f"%{mdd_strategy * 100:.2f}")
        
        son_input = scaler.transform(data[feature_cols].tail(1))
        yarinki_beklenti = model.predict(son_input)[0]
        c4.metric("AI Yarın Tahmini", f"%{yarinki_beklenti * 100:.2f}")

        # SEKME YAPISI
        tab1, tab2, tab3, tab4 = st.tabs(["📈 Getiri Kıyaslama", "🔍 Teknik & Risk", "🤖 AI Analizi", "🏢 Şirket Büyümesi"])

        with tab1:
            st.subheader("Robot vs Piyasa Performansı")
            st.line_chart(data[['Market_Cum', 'Strategy_Cum']])

        with tab2:
            st.subheader("Drawdown (Kayıp) ve Trend Analizi")
            st.area_chart(data[['Strategy_DD', 'Market_DD']])
            st.line_chart(data[['Close', 'SMA_K', 'SMA_U']])

        with tab3:
            st.subheader("Tahmin Başarısı (Son 15 Gün)")
            compare = pd.DataFrame({
                'Gerçekleşen': data['Target_Return'].tail(15),
                'AI Tahmini': data['AI_Pred_Return'].tail(15)
            })
            st.bar_chart(compare)
            if yarinki_beklenti > 0.005:
                st.success(f"🚀 **AI Kararı:** Güçlü yükseliş beklentisi (%{yarinki_beklenti * 100:.2f})")
            else:
                st.info("⚖️ **AI Kararı:** Belirgin bir yön yok.")

        with tab4:
            st.subheader(f"🏢 {sirket_bilgisi.get('longName', hisse)} Büyüme Analizi")
            col_inf1, col_inf2 = st.columns([1, 2])
            col_inf1.write(f"**Sektör:** {sirket_bilgisi.get('sector', 'Bilinmiyor')}")
            col_inf1.write(f"**P/E Oranı:** {sirket_bilgisi.get('trailingPE', 'N/A')}")
            col_inf2.write(f"**Özet:** {sirket_bilgisi.get('longBusinessSummary', 'Özet yok.')[:300]}...")

            if gelir_tablosu is not None and not gelir_tablosu.empty:
                yillik_veriler = gelir_tablosu.loc[['Total Revenue', 'Net Income']].T
                yillik_veriler.index = yillik_veriler.index.year
                yillik_veriler = yillik_veriler.sort_index()
                
                st.write("### Yıllık Gelir ve Net Kar Trendi")
                st.bar_chart(yillik_veriler)
                
                rev_growth = yillik_veriler['Total Revenue'].pct_change().iloc[-1] * 100
                st.metric("Son Yıl Gelir Büyümesi", f"%{rev_growth:.2f}")
            else:
                st.warning("Finansal tablo verisi çekilemedi.")

    else:
        st.error("Veri çekilemedi. Lütfen sembolü kontrol edin.")

except Exception as e:
    st.error(f"Sistem Hatası: {e}")

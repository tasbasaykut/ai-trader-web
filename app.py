import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use('Agg')  # Bulut sunucusu uyumu
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="A.S.T. Finansal Terminal", layout="wide")

# Şık bir görünüm için özel CSS
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stMetric { border: 1px solid #e0e0e0; padding: 15px; border-radius: 12px; background: white; }
    </style>
    """, unsafe_allow_html=True)

st.title("📊 A.S.T. Hibrit Yatırım Terminali")
st.sidebar.header("🕹️ Kontrol Paneli")

# --- SIDEBAR GİRDİLERİ ---
hisse = st.sidebar.text_input("Hisse Kodu", value="THYAO.IS")
zaman_secenekleri = {"1 Yıl": "1y", "2 Yıl": "2y", "5 Yıl": "5y"}
secilen_period = zaman_secenekleri[st.sidebar.selectbox("Analiz Dönemi", list(zaman_secenekleri.keys()), index=1)]
nakit = st.sidebar.number_input("Başlangıç Sermayesi (TL)", value=1000)

st.sidebar.divider()
sma_k = st.sidebar.slider("Kısa SMA", 5, 30, 20)
sma_u = st.sidebar.slider("Uzun SMA", 30, 100, 50)


# --- VERİ VE MODEL MOTORU ---
@st.cache_data
def verileri_hazirla(ticker, period, _sma_k, _sma_u):
    df = yf.download(ticker, period=period, interval="1d")
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    if df.empty: return None

    # Teknikler
    df['SMA_K'] = df['Close'].rolling(_sma_k).mean()
    df['SMA_U'] = df['Close'].rolling(_sma_u).mean()
    df['RSI'] = 100 - (100 / (1 + (df['Close'].diff().where(df['Close'].diff() > 0, 0).rolling(14).mean() /
                                   -df['Close'].diff().where(df['Close'].diff() < 0, 0).rolling(14).mean())))
    df['Volatilite'] = df['Close'].pct_change().rolling(10).std()
    df['Ret_Lag1'] = df['Close'].pct_change()
    df['Target_Return'] = df['Close'].pct_change().shift(-1)
    return df.dropna()


def model_calistir(df):
    features = ['RSI', 'Volatilite', 'Ret_Lag1']
    X = df[features]
    y = df['Target_Return']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42)
    model.fit(X_scaled, y)

    # Tüm veri için tahminler (Kar/Zarar analizi için)
    df['AI_Pred_Return'] = model.predict(X_scaled)
    return model, scaler, features


# --- ANA UYGULAMA MANTIĞI ---
data = verileri_hazirla(hisse, secilen_period, sma_k, sma_u)

if data is not None:
    model, scaler, feature_cols = model_calistir(data)

    # 1. HİBRİT STRATEJİ HESAPLAMA
    # Kural: SMA_K > SMA_U (Trend Yukarı) VE AI Pozitif Getiri Bekliyor
    data['Sinyal'] = np.where((data['SMA_K'] > data['SMA_U']) & (data['AI_Pred_Return'] > 0), 1, 0)

    # Getiri Hesapları
    data['Market_Cum'] = (1 + data['Close'].pct_change()).cumprod() * nakit
    data['Strategy_Cum'] = (1 + (data['Close'].pct_change() * data['Sinyal'].shift(1))).cumprod() * nakit

    # 2. ÜST METRİKLER (ÖZET)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Son Fiyat", f"{data['Close'].iloc[-1]:.2f} TL")
    m2.metric("Piyasa (Al-Tut)", f"{data['Market_Cum'].iloc[-1]:.2f} TL")
    m3.metric("Hibrit Robot", f"{data['Strategy_Cum'].iloc[-1]:.2f} TL",
              f"{((data['Strategy_Cum'].iloc[-1] / data['Market_Cum'].iloc[-1]) - 1) * 100:.1f}% Fark")

    # Yarınki AI Tahmini
    son_input = scaler.transform(data[feature_cols].tail(1))
    yarinki_beklenti = model.predict(son_input)[0]
    m4.metric("AI Yarın Beklentisi", f"%{yarinki_beklenti * 100:.2f}")

    # 3. GÖRSEL PANELLER
    tab1, tab2, tab3 = st.tabs(["📈 Kar/Zarar Kıyaslaması", "🔍 Teknik Analiz", "🤖 AI Detayları"])

    with tab1:
        st.subheader("Robot vs Piyasa Performansı")
        st.line_chart(data[['Market_Cum', 'Strategy_Cum']])
        st.caption(f"{nakit} TL yatırımın zaman içindeki değişimi (Mavi: Robot, Gri: Piyasa)")

    with tab2:
        st.subheader("Fiyat ve Hareketli Ortalamalar")
        st.line_chart(data[['Close', 'SMA_K', 'SMA_U']])
        st.subheader("RSI (Güç) Göstergesi")
        st.area_chart(data['RSI'])

    with tab3:
        st.subheader("Modelin Son Tahmin Performansı")
        compare = pd.DataFrame({
            'Gerçekleşen': data['Target_Return'].tail(15),
            'AI Tahmini': data['AI_Pred_Return'].tail(15)
        })
        st.bar_chart(compare)

        # Karar Kutusu
        if yarinki_beklenti > 0.005:
            st.success(f"🚀 **AI Kararı:** Yarın için güçlü yükseliş beklentisi (%{yarinki_beklenti * 100:.2f})")
        elif yarinki_beklenti < -0.005:
            st.error(f"⚠️ **AI Kararı:** Yarın için düşüş riski (%{yarinki_beklenti * 100:.2f})")
        else:
            st.warning("⚖️ **AI Kararı:** Belirgin bir yön yok, yatay seyir bekleniyor.")

else:
    st.error("Veri alınamadı. Lütfen sembolü (Örn: THYAO.IS) kontrol edin.")
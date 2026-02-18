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
st.set_page_config(page_title="A.S.T. Hibrit Risk Terminali", layout="wide")

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
    </style>
    """, unsafe_allow_html=True)

st.title("🛡️ A.S.T. Hibrit Terminali: Risk & Performans")
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
    df['AI_Pred_Return'] = model.predict(X_scaled)
    return model, scaler, features


# --- ANA UYGULAMA MANTIĞI ---
data = verileri_hazirla(hisse, secilen_period, sma_k, sma_u)

if data is not None:
    model, scaler, feature_cols = model_calistir(data)

    # 1. HİBRİT STRATEJİ HESAPLAMA
    data['Sinyal'] = np.where((data['SMA_K'] > data['SMA_U']) & (data['AI_Pred_Return'] > 0), 1, 0)

    # Getiri Hesapları
    data['Market_Cum'] = (1 + data['Close'].pct_change()).cumprod() * nakit
    data['Strategy_Cum'] = (1 + (data['Close'].pct_change() * data['Sinyal'].shift(1))).cumprod() * nakit

    # 2. RİSK METRİKLERİ HESAPLAMA (Maximum Drawdown)
    # Strateji için MDD
    data['Strategy_Peak'] = data['Strategy_Cum'].cummax()
    data['Strategy_DD'] = (data['Strategy_Cum'] - data['Strategy_Peak']) / data['Strategy_Peak']
    mdd_strategy = data['Strategy_DD'].min()

    # Piyasa için MDD
    data['Market_Peak'] = data['Market_Cum'].cummax()
    data['Market_DD'] = (data['Market_Cum'] - data['Market_Peak']) / data['Market_Peak']
    mdd_market = data['Market_DD'].min()

    # 3. ÜST METRİKLER (2 Satır Halinde)
    st.subheader("🏁 Performans Özet Tablosu")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Son Fiyat", f"{data['Close'].iloc[-1]:.2f} TL")
    c2.metric("Piyasa Final", f"{data['Market_Cum'].iloc[-1]:.2f} TL")
    c3.metric("Robot Final", f"{data['Strategy_Cum'].iloc[-1]:.2f} TL")

    son_input = scaler.transform(data[feature_cols].tail(1))
    yarinki_beklenti = model.predict(son_input)[0]
    c4.metric("AI Yarın Tahmini", f"%{yarinki_beklenti * 100:.2f}")

    st.subheader("🛡️ Risk Analizi (Zarar Durumu)")
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Robot Max Zarar", f"%{mdd_strategy * 100:.2f}")
    r2.metric("Piyasa Max Zarar", f"%{mdd_market * 100:.2f}")

    # Kazanma Oranı (Win Rate)
    win_rate = (data['Sinyal'].shift(1) * data['Close'].pct_change() > 0).sum() / (data['Sinyal'].shift(1) > 0).sum()
    r3.metric("İşlem Başarı Oranı", f"%{win_rate * 100:.1f}")

    # Toplam İşlem Sayısı
    islem_sayisi = (data['Sinyal'].diff() == 1).sum()
    r4.metric("Toplam Alım", f"{islem_sayisi} Kez")

    # 4. GÖRSEL PANELLER
    tab1, tab2, tab3 = st.tabs(["📈 Kümülatif Getiri", "🔍 Teknik & Drawdown", "🤖 AI Detayları"])

    with tab1:
        st.subheader("Strateji Karşılaştırması")
        st.line_chart(data[['Market_Cum', 'Strategy_Cum']])

    with tab2:
        st.subheader("Drawdown (Anlık Kayıp Grafiği)")
        # Drawdown grafiği hissenin nerede tepeden düştüğünü gösterir
        st.area_chart(data[['Strategy_DD', 'Market_DD']])
        st.caption("Grafiğin aşağı sarkması, o dönemde zirveden yaşanan kaybı temsil eder.")

        st.subheader("Teknik Trend (SMA)")
        st.line_chart(data[['Close', 'SMA_K', 'SMA_U']])

    with tab3:
        st.subheader("Tahmin vs Gerçek")
        compare = pd.DataFrame({
            'Gerçekleşen': data['Target_Return'].tail(15),
            'AI Tahmini': data['AI_Pred_Return'].tail(15)
        })
        st.bar_chart(compare)

        if yarinki_beklenti > 0.005:
            st.success(f"🚀 **AI Kararı:** Yarın için güçlü yükseliş beklentisi (%{yarinki_beklenti * 100:.2f})")
        elif yarinki_beklenti < -0.005:
            st.error(f"⚠️ **AI Kararı:** Yarın için düşüş riski (%{yarinki_beklenti * 100:.2f})")
        else:
            st.warning("⚖️ **AI Kararı:** Belirgin bir yön yok.")

else:
    st.error("Veri alınamadı.")
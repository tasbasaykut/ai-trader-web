import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

# --- SAYFA YAPILANDIRMASI ---
st.set_page_config(page_title="Hibrit Finansal Terminal v2", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_html=True)

st.title("🛡️ A.S.T. Hibrit Terminali (Zaman Ayarlı)")
st.caption("Teknik Analiz + AI + Parametre Optimizasyonu + Dinamik Zaman Aralığı")

# --- YAN PANEL (SIDEBAR) ---
st.sidebar.header("⚙️ Sistem Ayarları")
hisse = st.sidebar.text_input("Hisse Sembolü", value="THYAO.IS")

# --- YENİ ÖZELLİK: ZAMAN ARALIĞI SEÇİMİ ---
zaman_secenekleri = {
    "1 Ay": "1mo",
    "3 Ay": "3mo",
    "6 Ay": "6mo",
    "1 Yıl": "1y",
    "2 Yıl": "2y",
    "5 Yıl": "5y",
    "Maksimum": "max"
}
secilen_etiket = st.sidebar.selectbox("Analiz Dönemi (Veri Derinliği)", list(zaman_secenekleri.keys()), index=4)
secilen_period = zaman_secenekleri[secilen_etiket]

baslangic_nakit = st.sidebar.number_input("Başlangıç Bakiyesi (TL)", value=1000)

st.sidebar.divider()
st.sidebar.subheader("Strateji Parametreleri")
sma_kisa = st.sidebar.slider("Kısa SMA", 5, 50, 20)
sma_uzun = st.sidebar.slider("Uzun SMA", 20, 200, 50)
rsi_limit = st.sidebar.slider("RSI Üst Sınır", 50, 90, 70)
stop_loss = st.sidebar.slider("Stop-Loss (%)", 1, 20, 5) / 100

st.sidebar.divider()
opt_mode = st.sidebar.checkbox("Optimizasyon Modunu Aç")
ai_mode = st.sidebar.checkbox("AI Tahminini Etkinleştir", value=True)


# --- FONKSİYONLAR ---
@st.cache_data
def verileri_hazirla(ticker, period):  # period parametresi eklendi
    df = yf.download(ticker, period=period, interval="1d")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Teknik Göstergeler
    df['SMA_K'] = df['Close'].rolling(window=sma_kisa).mean()
    df['SMA_U'] = df['Close'].rolling(window=sma_uzun).mean()

    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['RSI'] = 100 - (100 / (1 + (gain / loss)))

    # AI Özellikleri (Lagged Features)
    df['Lag_1'] = df['Close'].shift(1)
    df['Lag_2'] = df['Close'].shift(2)

    return df.dropna()


def ai_model_egit(df):
    if len(df) < 50:  # Çok kısa sürelerde model eğitilemez
        return None, 0, 0

    X = df[['Lag_1', 'Lag_2', 'SMA_K', 'RSI']]
    y = df['Close']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    dogruluk = np.sum(np.sign(np.diff(y_test)) == np.sign(np.diff(y_pred))) / len(np.diff(y_test))

    return model, mae, dogruluk


# --- ANA PROGRAM AKIŞI ---
try:
    data = verileri_hazirla(hisse, secilen_period)

    if data.empty:
        st.error("Bu zaman aralığında veri bulunamadı.")
    else:
        # 1. OPTİMİZASYON BÖLÜMÜ
        if opt_mode:
            st.subheader("🧪 Parametre Optimizasyonu (Grid Search)")
            if st.button("En İyi SMA Kombinasyonunu Bul"):
                best_ret = -np.inf
                best_s, best_l = 0, 0
                for s in range(10, 31, 10):
                    for l in range(40, 101, 20):
                        temp_pos = np.where(data['Close'].rolling(s).mean() > data['Close'].rolling(l).mean(), 1, 0)
                        ret = (1 + (data['Close'].pct_change() * pd.Series(temp_pos).shift(1).values)).cumprod().iloc[
                            -1]
                        if ret > best_ret:
                            best_ret, best_s, best_l = ret, s, l
                st.success(f"En İyi Sonuç: SMA {best_s} - SMA {best_l} | Getiri: %{((best_ret - 1) * 100):.2f}")

        # 2. AI VE STRATEJI BÖLÜMÜ
        model, mae, dogruluk = None, 0, 0
        yarin_tahmin = 0

        if ai_mode:
            model, mae, dogruluk = ai_model_egit(data)
            if model:
                son_input = np.array(
                    [[data['Close'].iloc[-1], data['Close'].iloc[-2], data['SMA_K'].iloc[-1], data['RSI'].iloc[-1]]])
                yarin_tahmin = model.predict(son_input)[0]

        # Strateji Uygulama
        data['Teknik_Sinyal'] = np.where((data['SMA_K'] > data['SMA_U']) & (data['RSI'] < rsi_limit), 1, 0)

        if ai_mode and model:
            data['AI_Onay'] = np.where(model.predict(data[['Lag_1', 'Lag_2', 'SMA_K', 'RSI']]) > data['Close'], 1, 0)
            data['Final_Sinyal'] = data['Teknik_Sinyal'] * data['AI_Onay']
        else:
            data['Final_Sinyal'] = data['Teknik_Sinyal']

        data['Strateji_Getiri'] = (1 + (
                    data['Close'].pct_change() * data['Final_Sinyal'].shift(1))).cumprod() * baslangic_nakit
        data['Piyasa_Getiri'] = (1 + data['Close'].pct_change()).cumprod() * baslangic_nakit

        # --- ARAYÜZ METRİKLERİ ---
        st.subheader(f"📊 {secilen_etiket} İçin Performans Özeti")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Güncel Fiyat", f"{data['Close'].iloc[-1]:.2f} TL")
        c2.metric("Strateji Final", f"{data['Strateji_Getiri'].iloc[-1]:.2f} TL")
        c3.metric("Piyasa Final", f"{data['Piyasa_Getiri'].iloc[-1]:.2f} TL")
        if ai_mode and model:
            c4.metric("AI Tahmin Doğruluğu", f"%{dogruluk * 100:.1f}")

        # --- KARAR PANELİ ---
        st.divider()
        if ai_mode and model:
            fark = yarin_tahmin - data['Close'].iloc[-1]
            if fark > 0 and data['SMA_K'].iloc[-1] > data['SMA_U'].iloc[-1]:
                st.success(f"🚀 **GÜÇLÜ AL:** Teknik pozitif ve AI yarın için {yarin_tahmin:.2f} bekliyor.")
            elif fark < 0:
                st.error(f"⚠️ **DİKKAT:** AI yarın için düşüş ({yarin_tahmin:.2f}) bekliyor.")
            else:
                st.info("⚖️ **NÖTR:** Net bir sinyal oluşmadı.")

        # --- GRAFİKLER ---
        tab1, tab2 = st.tabs(["📈 Performans Grafiği", "📊 Teknik Göstergeler"])
        with tab1:
            st.line_chart(data[['Strateji_Getiri', 'Piyasa_Getiri']])
        with tab2:
            st.line_chart(data[['Close', 'SMA_K', 'SMA_U']])
            st.area_chart(data['RSI'])

except Exception as e:
    st.error(f"Sistem Hatası: {e}")

st.divider()
st.caption(f"Veri Periyodu: {secilen_etiket} | Computer Engineering Project")
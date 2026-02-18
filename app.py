import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

# --- SAYFA YAPILANDIRMASI ---
st.set_page_config(page_title="XGBoost AI Finansal Terminal", layout="wide")

st.title("🛡️ A.S.T. Ultra AI Terminal v4")
st.caption("XGBoost Regressor + Teknik İndikatörler + Yüzdesel Getiri Analizi")

# --- SIDEBAR ---
st.sidebar.header("⚙️ Ayarlar")
hisse = st.sidebar.text_input("Hisse Sembolü", value="THYAO.IS")
zaman_secenekleri = {"2 Yıl": "2y", "5 Yıl": "5y", "Maksimum": "max"}
secilen_period = zaman_secenekleri[st.sidebar.selectbox("Veri Derinliği", list(zaman_secenekleri.keys()), index=1)]

st.sidebar.divider()
sma_kisa = st.sidebar.slider("Kısa SMA", 5, 50, 20)
sma_uzun = st.sidebar.slider("Uzun SMA", 20, 200, 50)


# --- FONKSİYONLAR ---

@st.cache_data
def verileri_hazirla(ticker, period):
    df = yf.download(ticker, period=period, interval="1d")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # 1. TEKNİK ÖZELLİKLER (FEATURES)
    df['SMA_K'] = df['Close'].rolling(window=sma_kisa).mean()
    df['SMA_U'] = df['Close'].rolling(window=sma_uzun).mean()

    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['RSI'] = 100 - (100 / (1 + (gain / loss)))

    # MACD (Trendin gücünü ölçer)
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2

    # Volatilite ve Getiri
    df['Volatilite'] = df['Close'].pct_change().rolling(window=10).std()
    df['Ret_Lag1'] = df['Close'].pct_change()

    # Hedef (Target): Yarınki yüzde değişim
    df['Target_Return'] = df['Close'].pct_change().shift(-1)

    return df.dropna()


def xgboost_model_egit(df):
    features = ['RSI', 'MACD', 'Volatilite', 'Ret_Lag1', 'SMA_K']
    X = df[features]
    y = df['Target_Return']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # XGBoost Regressor Yapılandırması
    # n_estimators: Ağaç sayısı
    # learning_rate: Öğrenme hızı (Küçük olması daha iyi öğrenme ama daha yavaş işlem demek)
    model = XGBRegressor(
        n_estimators=1000,
        learning_rate=0.01,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    mae = mean_absolute_error(y_test, y_pred)
    direction_acc = np.mean(np.sign(y_test) == np.sign(y_pred))

    return model, scaler, mae, direction_acc, features


# --- ANA AKIŞ ---
try:
    data = verileri_hazirla(hisse, secilen_period)

    if not data.empty:
        # XGBoost Eğitim
        model, scaler, mae, acc, f_list = xgboost_model_egit(data)

        # Tahmin
        son_veri_scaled = scaler.transform(data[f_list].tail(1))
        beklenen_getiri = model.predict(son_veri_scaled)[0]

        # Arayüz
        st.subheader("🚀 XGBoost AI Tahmin Raporu")
        c1, c2, c3 = st.columns(3)
        c1.metric("Tahmin Edilen Yarınki Değişim", f"%{beklenen_getiri * 100:.2f}")
        c2.metric("Yön Doğruluğu (Test)", f"%{acc * 100:.1f}")
        c3.metric("Ortalama Hata", f"{mae:.4f}")

        # Strateji Onayı
        st.divider()
        if beklenen_getiri > 0.003 and data['SMA_K'].iloc[-1] > data['SMA_U'].iloc[-1]:
            st.success("🟢 **GÜÇLÜ AL SİNYALİ:** Teknik Trend + XGBoost Onayı.")
        elif beklenen_getiri < -0.003:
            st.error("🔴 **GÜÇLÜ SAT SİNYALİ:** Yapay Zeka Düşüş Bekliyor.")
        else:
            st.warning("🟡 **BEKLE:** Net bir yön tayin edilemedi.")

        # Tahmin vs Gerçek Bar Chart
        st.write("Modelin Son 15 Günlük Performansı (Tahmin vs Gerçek)")
        test_indices = int(len(data) * 0.8)
        y_test_real = data['Target_Return'].iloc[test_indices:]
        y_pred_all = model.predict(scaler.transform(data[f_list].iloc[test_indices:]))

        compare_df = pd.DataFrame({
            'Gerçekleşen': y_test_real,
            'XGBoost Tahmini': y_pred_all
        }).tail(15)
        st.bar_chart(compare_df)

except Exception as e:
    st.error(f"Sistem Hatası: {e}")
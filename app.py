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
st.set_page_config(page_title="A.S.T. Ultra Terminal v16.2", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #0E1117; }
    div[data-testid="stMetric"] { background-color: #1E1E26; border: 1px solid #3E3E4E; padding: 15px; border-radius: 12px; }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] { color: #FFFFFF !important; font-weight: 700; }
    .stTabs [data-baseweb="tab"] { color: #BFC5D3; }
    .stTabs [aria-selected="true"] { color: #FFFFFF !important; border-bottom-color: #FF4B4B !important; }
    </style>
    """, unsafe_allow_html=True)

st.title("🔮 A.S.T. Ultra v16.2: Profesyonel Tahmin & Analiz")
st.caption("Gelişmiş Görselleştirme + Veri Senkronizasyonu + Multi-Target AI")

# --- 2. VERİ VE MODEL MOTORU ---
@st.cache_data
def v16_veri_hazirla(ticker, period):
    df = yf.download(ticker, period=period, interval="1d")
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    if df.empty: return None, None, None, None
    
    s = yf.Ticker(ticker)
    
    # İndikatörler (Tüm satırlar için hesaplanır)
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
    
    # Tahmin için en son satırı sakla (henüz Target'ı oluşmamış en güncel veri)
    current_data = df.tail(1).copy()
    
    # Eğitim için sadece Target'ı dolu olanları kullan
    train_data = df.dropna().copy()
    
    return train_data, current_data, s.financials, s.info

def model_merkezi(train_df):
    features = ['RSI', 'Volatility', 'BB_Upper', 'BB_Lower', 'Close']
    X = train_df[features]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Modeller
    m1 = XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=5)
    m1.fit(X_scaled, train_df['Target_1d'])
    
    m7 = XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=5)
    m7.fit(X_scaled, train_df['Target_7d'])
    
    mp = XGBClassifier(n_estimators=100, learning_rate=0.05, eval_metric='logloss')
    mp.fit(X_scaled, train_df['Target_Binary'])
    
    train_df['AI_Pred_1d'] = m1.predict(X_scaled)
    return m1, m7, mp, scaler, features

# --- 3. ANA AKIŞ ---
hisse = st.sidebar.text_input("Hisse Sembolü", value="THYAO.IS")
nakit = st.sidebar.number_input("Bakiye", value=1000)

try:
    train_data, current_data, financials, info = v16_veri_hazirla(hisse, "2y")

    if train_data is not None:
        m1, m7, mp, scaler, f_list = model_merkezi(train_data)
        
        # En güncel satır ile gerçek yarını tahmin et
        last_row_scaled = scaler.transform(current_data[f_list])
        p_1d = m1.predict(last_row_scaled)[0]
        p_7d = m7.predict(last_row_scaled)[0]
        prob = mp.predict_proba(last_row_scaled)[0, 1]

        # --- TARİH HESAPLAMALARI ---
        son_veri_tarihi = current_data.index[-1].strftime('%d.%m.%Y')
        tahmin_hedefi = (current_data.index[-1] + pd.Timedelta(days=1)).strftime('%d.%m.%Y')

        # ÜST METRİKLER
        st.subheader(f"🏁 Stratejik Özet ({son_veri_tarihi} Kapanış Verileriyle)")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Son Kapanış Fiyatı", f"{current_data['Close'].iloc[-1]:.2f} TL")
        c2.metric("Tahmin Hedefi", tahmin_hedefi)
        c3.metric("Yükseliş Güveni", f"%{prob*100:.1f}")
        c4.metric("Veri Güncelliği", son_veri_tarihi)

        # SEKMELER
        t1, t2, t3, t4, t5 = st.tabs(["🔮 Gelecek Kahini", "🤖 AI Performans", "📈 Teknik Analiz", "🏢 Şirket Röntgeni", "🛡️ Risk Yönetimi"])
        
        with t1:
            st.subheader(f"🚀 {hisse} 7 Günlük Fiyat Projeksiyonu")
            future_dates = pd.date_range(start=current_data.index[-1] + pd.Timedelta(days=1), periods=7)
            current_price = current_data['Close'].iloc[-1]
            vol = current_data['Volatility'].iloc[-1]
            
            # Fiyat ve Güven Aralığı Hesaplama
            future_prices = [current_price * (1 + (p_7d/7) * i) for i in range(1, 8)]
            upper_bound = [p * (1 + vol * np.sqrt(i)) for i, p in enumerate(future_prices, 1)]
            lower_bound = [p * (1 - vol * np.sqrt(i)) for i, p in enumerate(future_prices, 1)]
            
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(train_data.index[-20:], train_data['Close'].tail(20), label="Geçmiş Fiyat", color='#00d4ff', linewidth=2)
            ax.plot(future_dates, future_prices, label="AI Tahmini", linestyle='--', color='#ff4b4b', marker='o')
            ax.fill_between(future_dates, lower_bound, upper_bound, color='#ff4b4b', alpha=0.1, label="Güven Aralığı")
            
            ax.set_facecolor('#0E1117'); fig.patch.set_facecolor('#0E1117')
            ax.tick_params(colors='white'); plt.grid(color='#3E3E4E', linestyle='--', alpha=0.5)
            plt.legend(facecolor='#1E1E26', labelcolor='white')
            st.pyplot(fig)
            
            st.table(pd.DataFrame({
                "Tarih": future_dates.strftime('%Y-%m-%d'),
                "Tahmin": [f"{p:.2f} TL" for p in future_prices],
                "Alt Sınır": [f"{l:.2f} TL" for l in lower_bound],
                "Üst Sınır": [f"{u:.2f} TL" for u in upper_bound]
            }))

        with t2:
            st.subheader("🤖 Model Doğruluk Analizi")
            kiyas_df = pd.DataFrame({
                'Gerçekleşen': train_data['Target_1d'].tail(15),
                'AI Tahmini': train_data['AI_Pred_1d'].tail(15)
            })
            st.bar_chart(kiyas_df)
            mae = mean_absolute_error(train_data['Target_1d'], train_data['AI_Pred_1d'])
            st.info(f"💡 Ortalama Tahmin Hatası (MAE): %{mae*100:.4f}")

        with t3:
            col_l, col_r = st.columns(2)
            col_l.area_chart(train_data['RSI'].tail(100))
            col_r.line_chart(train_data[['Close', 'BB_Upper', 'BB_Lower']].tail(100))
            st.bar_chart(train_data['Volume'].tail(100))

        with t4:
            if financials is not None and not financials.empty:
                st.bar_chart(financials.loc[['Total Revenue', 'Net Income']].T)

        with t5:
            train_data['Sinyal'] = np.where((train_data['AI_Pred_1d'] > 0), 1, 0)
            train_data['Strategy_Cum'] = (1 + (train_data['Target_1d'] * train_data['Sinyal'])).cumprod() * nakit
            train_data['Market_Cum'] = (1 + train_data['Target_1d']).cumprod() * nakit
            st.line_chart(train_data[['Market_Cum', 'Strategy_Cum']])

except Exception as e:
    st.error(f"Sistem Hatası: {e}")

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import requests
from xgboost import XGBRegressor, XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

# --- 1. SAYFA YAPILANDIRMASI ---
st.set_page_config(page_title="A.S.T. Ultra Terminal v20.2", layout="wide")

# Telegram API Ayarları
TELEGRAM_TOKEN = "8438099476:AAHWz26Y0bnInuskr_Qjgno4TjjiHOpJ7ao"
CHAT_ID = "5026797450"

def telegram_sinyal_gonder(mesaj, debug=False):
    """
    Telegram'a mesaj gönderir. 
    debug=True ise hata mesajlarını ekranda gösterir.
    """
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": mesaj, "parse_mode": "Markdown"}
    try:
        response = requests.post(url, json=payload, timeout=10)
        res_data = response.json()
        if res_data.get("ok"):
            if debug: st.sidebar.success("✅ Telegram Bağlantısı Başarılı!")
            return True
        else:
            if debug: st.sidebar.error(f"❌ Telegram Hatası: {res_data.get('description')}")
            return False
    except Exception as e:
        if debug: st.sidebar.error(f"❌ Bağlantı Hatası: {e}")
        return False

st.markdown("""
    <style>
    .stApp { background-color: #0E1117; }
    div[data-testid="stMetric"] { background-color: #1E1E26; border: 1px solid #3E3E4E; padding: 15px; border-radius: 12px; }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] { color: #FFFFFF !important; font-weight: 700; }
    .desc-box { background-color: #1E1E26; padding: 15px; border-radius: 8px; border-left: 5px solid #ff4b4b; margin: 10px 0; font-size: 14px; color: #BFC5D3; }
    </style>
    """, unsafe_allow_html=True)

st.title("🛡️ A.S.T. Ultra v20.2: Master Terminal")

# --- 2. VERİ VE MODEL MOTORU ---
@st.cache_data
def v20_veri_hazirla(ticker, period):
    df = yf.download(ticker, period=period, interval="1d")
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    if df.empty: return None, None, None, None
    s = yf.Ticker(ticker)
    
    # Teknik İndikatörler
    df['SMA_20'] = df['Close'].rolling(20).mean()
    df['BB_Std'] = df['Close'].rolling(20).std()
    df['BB_Upper'] = df['SMA_20'] + (df['BB_Std'] * 2)
    df['BB_Lower'] = df['SMA_20'] - (df['BB_Std'] * 2)
    delta = df['Close'].diff()
    df['RSI'] = 100 - (100 / (1 + (delta.where(delta > 0, 0).rolling(14).mean() / -delta.where(delta < 0, 0).rolling(14).mean())))
    df['Volatility'] = df['Close'].pct_change().rolling(10).std()
    
    df['Target_1d'] = df['Close'].pct_change().shift(-1)
    df['Target_7d'] = (df['Close'].shift(-7) / df['Close']) - 1
    df['Target_Binary'] = (df['Target_1d'] > 0).astype(int)
    
    current_data = df.tail(1).copy()
    train_data = df.dropna().copy()
    return train_data, current_data, s.financials, s.info

def model_merkezi(train_df):
    features = ['RSI', 'Volatility', 'BB_Upper', 'BB_Lower', 'Close']
    X = train_df[features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    m1 = XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=5).fit(X_scaled, train_df['Target_1d'])
    m7 = XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=5).fit(X_scaled, train_df['Target_7d'])
    mp = XGBClassifier(n_estimators=100, eval_metric='logloss').fit(X_scaled, train_df['Target_Binary'])
    
    train_df['AI_Pred_1d'] = m1.predict(X_scaled)
    return m1, m7, mp, scaler, features

# --- 3. SIDEBAR (TAM ÖZELLEŞTİRME) ---
st.sidebar.header("⚙️ Portföy Kontrol")
custom_input = st.sidebar.text_area("Kıyaslanacak Hisse Listesi (Virgül ile)", value="THYAO.IS, ASELS.IS, EREGL.IS, FROTO.IS")
izleme_listesi = [x.strip() for x in custom_input.split(",") if x.strip()]
hisse = st.sidebar.selectbox("Detaylı Analiz Odağı", izleme_listesi)
nakit = st.sidebar.number_input("Bakiye (TL)", value=1000)

st.sidebar.markdown("---")
# Bot Bağlantı Testi
if st.sidebar.button("🔌 Bot Bağlantısını Test Et"):
    telegram_sinyal_gonder("🔔 *Sistem Testi:* A.S.T. Bot bağlantısı şu an aktif! 🚀", debug=True)

# --- 4. ANA AKIŞ ---
try:
    train_data, current_data, financials, info = v20_veri_hazirla(hisse, "2y")

    if train_data is not None:
        m1, m7, mp, scaler, f_list = model_merkezi(train_data)
        last_row_scaled = scaler.transform(current_data[f_list])
        p_1d, p_7d, prob = m1.predict(last_row_scaled)[0], m7.predict(last_row_scaled)[0], mp.predict_proba(last_row_scaled)[0, 1]
        son_tarih = current_data.index[-1].strftime('%d.%m.%Y')

        # ÜST METRİKLER
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Son Fiyat", f"{current_data['Close'].iloc[-1]:.2f} TL")
        c2.metric("Yarın Beklenti", f"%{p_1d*100:.2f}")
        c3.metric("Yükseliş Güveni", f"%{prob*100:.1f}")
        c4.metric("Veri Tarihi", son_tarih)

        # Telegram Sinyal Butonu
        if st.sidebar.button("📩 Hisse Sinyalini Gönder"):
            if prob >= 0.70:
                mesaj = f"🚀 *SİNYAL:* {hisse}\n🔥 *Güven:* %{prob*100:.1f}\n💰 *Beklenti:* %{p_1d*100:.2f}\n📅 *Tarih:* {son_tarih}"
                telegram_sinyal_gonder(mesaj, debug=True)
                st.sidebar.success("Sinyal iletildi!")
            else:
                st.sidebar.warning(f"⚠️ Güven düşük (%{prob*100:.1f}). Sinyal engellendi.")

        t1, t2, t3, t4, t5, t6 = st.tabs(["🔮 Gelecek Kahini", "🤖 AI Performans", "📈 Teknik Analiz", "🏢 Şirket Röntgeni", "🛡️ Risk Yönetimi", "📊 Hızlı Kıyaslama"])

        with t1:
            st.subheader("📍 7 Günlük Fiyat Projeksiyonu")
            future_dates = pd.date_range(start=current_data.index[-1] + pd.Timedelta(days=1), periods=7)
            future_prices = [current_data['Close'].iloc[-1] * (1 + (p_7d/7) * i) for i in range(1, 8)]
            vol = train_data['Volatility'].iloc[-1]
            upper_bound = [p * (1 + vol * np.sqrt(i)) for i, p in enumerate(future_prices, 1)]
            lower_bound = [p * (1 - vol * np.sqrt(i)) for i, p in enumerate(future_prices, 1)]
            
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(train_data.index[-30:], train_data['Close'].tail(30), label="Geçmiş Fiyat", color='#00d4ff', marker='o')
            ax.plot(future_dates, future_prices, label="AI Tahmini", linestyle='--', color='#ff4b4b', marker='s')
            ax.fill_between(future_dates, lower_bound, upper_bound, color='#ff4b4b', alpha=0.15, label="Güven Aralığı")
            for i, txt in enumerate(future_prices): ax.annotate(f"{txt:.1f}", (future_dates[i], future_prices[i]), xytext=(0,10), textcoords='offset points', ha='center', color='white', fontsize=9)
            ax.set_facecolor('#0E1117'); fig.patch.set_facecolor('#0E1117'); ax.tick_params(colors='white'); plt.legend()
            st.pyplot(fig)
            st.table(pd.DataFrame({"Tarih": future_dates.strftime('%d.%m.%Y'), "Hedef Fiyat": [f"{p:.2f} TL" for p in future_prices], "Günlük Getiri (%)": [f"%{((p/current_data['Close'].iloc[-1])-1)*100:.2f}" for p in future_prices]}))

        with t2:
            st.subheader("🤖 Model Doğruluk Analizi")
            kiyas_df = pd.DataFrame({'Gerçek': train_data['Target_1d'].tail(20), 'AI': train_data['AI_Pred_1d'].tail(20)})
            st.bar_chart(kiyas_df)
            st.info(f"💡 MAE: %{mean_absolute_error(train_data['Target_1d'], train_data['AI_Pred_1d'])*100:.4f}")

        with t3:
            st.subheader("🔍 Teknik Analiz Kanalları")
            fig1, ax1 = plt.subplots(figsize=(12, 5))
            ax1.plot(train_data.index[-100:], train_data['Close'].tail(100), label="Fiyat", color='white', linewidth=2)
            ax1.plot(train_data.index[-100:], train_data['SMA_20'].tail(100), label="SMA 20", color='orange', alpha=0.7)
            ax1.fill_between(train_data.index[-100:], train_data['BB_Lower'].tail(100), train_data['BB_Upper'].tail(100), color='gray', alpha=0.2, label="Bollinger")
            ax1.set_facecolor('#0E1117'); fig1.patch.set_facecolor('#0E1117'); ax1.tick_params(colors='white'); plt.legend()
            st.pyplot(fig1)
            st.area_chart(train_data['RSI'].tail(100))
            st.bar_chart(train_data['Volume'].tail(100))

        with t4:
            if financials is not None and not financials.empty: st.bar_chart(financials.loc[['Total Revenue', 'Net Income']].T)

        with t5:
            st.subheader("🛡️ Risk ve Getiri Kıyaslaması (Backtest)")
            train_data['Sinyal'] = np.where((train_data['AI_Pred_1d'] > 0), 1, 0)
            train_data['Strategy_Cum'] = (1 + (train_data['Target_1d'] * train_data['Sinyal'])).cumprod() * nakit
            train_data['Market_Cum'] = (1 + train_data['Target_1d']).cumprod() * nakit
            st.line_chart(train_data[['Market_Cum', 'Strategy_Cum']])
            peak = train_data['Strategy_Cum'].cummax()
            dd = ((train_data['Strategy_Cum'] - peak) / peak)
            st.area_chart(dd)

        with t6:
            st.subheader("📊 Liste İçi AI Kapışması")
            if len(izleme_listesi) > 0:
                results = []
                with st.spinner("Analiz ediliyor..."):
                    for s_hisse in izleme_listesi:
                        try:
                            t_d, c_d, _, _ = v20_veri_hazirla(s_hisse, "1y")
                            m_1, _, m_p, sc, f_l = model_merkezi(t_d)
                            l_r = sc.transform(c_d[f_l])
                            results.append({"Hisse": s_hisse, "Güven (%)": round(m_p.predict_proba(l_r)[0,1]*100,2), "Beklenti (%)": round(m_1.predict(l_r)[0]*100,2)})
                        except: continue
                df_res = pd.DataFrame(results).sort_values("Güven (%)", ascending=False)
                st.dataframe(df_res, use_container_width=True)
                st.bar_chart(df_res.set_index("Hisse")["Güven (%)"])

except Exception as e:
    st.error(f"Sistem Hatası: {e}")

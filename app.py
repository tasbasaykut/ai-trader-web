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
st.set_page_config(page_title="A.S.T. Ultra Terminal v19", layout="wide")

# Telegram API (Bilgilerin korunuyor)
TELEGRAM_TOKEN = "8438099476:AAHWz26Y0bnInuskr_Qjgno4TjjiHOpJ7ao"
CHAT_ID = "5026797450"

# BIST10 Listesi (Hız için optimize edildi)
BIST10 = ['AKBNK.IS', 'BIMAS.IS', 'EREGL.IS', 'FROTO.IS', 'GARAN.IS', 'ISCTR.IS', 'KCHOL.IS', 'THYAO.IS', 'TUPRS.IS', 'YKBNK.IS']

def telegram_sinyal_gonder(mesaj):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": mesaj, "parse_mode": "Markdown"}
    try: requests.post(url, json=payload)
    except: pass

st.markdown("""
    <style>
    .stApp { background-color: #0E1117; }
    div[data-testid="stMetric"] { background-color: #1E1E26; border: 1px solid #3E3E4E; padding: 15px; border-radius: 12px; }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] { color: #FFFFFF !important; font-weight: 700; }
    .stTabs [data-baseweb="tab"] { color: #BFC5D3; }
    .stTabs [aria-selected="true"] { color: #FFFFFF !important; border-bottom-color: #FF4B4B !important; }
    </style>
    """, unsafe_allow_html=True)

st.title("🛡️ A.S.T. Ultra v19: BIST10 & Veri Derinliği")

# --- 2. VERİ VE MODEL MOTORU ---
@st.cache_data
def v19_veri_hazirla(ticker, period):
    df = yf.download(ticker, period=period, interval="1d")
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    if df.empty: return None, None, None, None
    s = yf.Ticker(ticker)
    
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

# --- 3. SIDEBAR ---
st.sidebar.header("📁 Portföy & İzleme")
izleme_listesi = st.sidebar.multiselect("Kıyaslanacak BIST10 Hisseleri", BIST10, default=BIST10[:5])
hisse = st.sidebar.selectbox("Detaylı Analiz Odağı", BIST10, index=7) # Varsayılan THYAO
nakit = st.sidebar.number_input("Bakiye (TL)", value=1000)

# --- 4. ANA AKIŞ ---
try:
    train_data, current_data, financials, info = v19_veri_hazirla(hisse, "2y")

    if train_data is not None:
        m1, m7, mp, scaler, f_list = model_merkezi(train_data)
        last_row_scaled = scaler.transform(current_data[f_list])
        p_1d, p_7d, prob = m1.predict(last_row_scaled)[0], m7.predict(last_row_scaled)[0], mp.predict_proba(last_row_scaled)[0, 1]
        son_tarih, hedef_tarih = current_data.index[-1].strftime('%d.%m.%Y'), (current_data.index[-1] + pd.Timedelta(days=1)).strftime('%d.%m.%Y')

        st.subheader(f"🏁 {hisse} Analiz Paneli")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Son Kapanış", f"{current_data['Close'].iloc[-1]:.2f} TL")
        c2.metric("AI Yarın Hedefi", hedef_tarih)
        c3.metric("Yükseliş Güveni", f"%{prob*100:.1f}")
        c4.metric("Beklenen Getiri", f"%{p_1d*100:.2f}")

        if st.sidebar.button("Telegram'a Raporla"):
            mesaj = f"🚀 *A.S.T. v19 Sinyal*\n\n📈 *Hisse:* {hisse}\n🔥 *Güven:* %{prob*100:.1f}\n💰 *Beklenti:* %{p_1d*100:.2f}"
            telegram_sinyal_gonder(mesaj)
            st.sidebar.success("Mesaj gönderildi!")

        t1, t2, t3, t4, t5, t6 = st.tabs(["🔮 Gelecek Kahini", "🤖 AI Performans", "📈 Teknik Analiz", "🏢 Şirket Röntgeni", "🛡️ Risk Yönetimi", "📊 BIST10 Kıyaslama"])
        
        with t1:
            st.subheader("📍 7 Günlük Fiyat ve Risk Projeksiyonu")
            future_dates = pd.date_range(start=current_data.index[-1] + pd.Timedelta(days=1), periods=7)
            current_price = current_data['Close'].iloc[-1]
            future_prices = [current_price * (1 + (p_7d/7) * i) for i in range(1, 8)]
            vol = train_data['Volatility'].iloc[-1]
            
            # Güven aralığı formülü: $Bound = Price_{t+n} \times (1 \pm Volatility \times \sqrt{n})$
            upper_bound = [p * (1 + vol * np.sqrt(i)) for i, p in enumerate(future_prices, 1)]
            lower_bound = [p * (1 - vol * np.sqrt(i)) for i, p in enumerate(future_prices, 1)]
            
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(train_data.index[-30:], train_data['Close'].tail(30), label="Gerçek Fiyat", color='#00d4ff', marker='o', markersize=4)
            ax.plot(future_dates, future_prices, label="AI Tahmin Yolu", linestyle='--', color='#ff4b4b', marker='s', markersize=5)
            ax.fill_between(future_dates, lower_bound, upper_bound, color='#ff4b4b', alpha=0.15, label="Olası Sapma Bölgesi")
            
            for i, txt in enumerate(future_prices):
                ax.annotate(f"{txt:.1f}", (future_dates[i], future_prices[i]), textcoords="offset points", xytext=(0,10), ha='center', color='white', fontsize=8)

            ax.set_facecolor('#0E1117'); fig.patch.set_facecolor('#0E1117')
            ax.tick_params(colors='white'); plt.grid(color='#3E3E4E', alpha=0.3); plt.legend(facecolor='#1E1E26', labelcolor='white')
            st.pyplot(fig)
            
            st.write("**Gelecek Periyodu Sayısal Verileri**")
            tahmin_tablosu = pd.DataFrame({
                "Tarih": future_dates.strftime('%d.%m.%Y'),
                "AI Hedef Fiyat": [f"{p:.2f} TL" for p in future_prices],
                "Olası En Düşük": [f"{l:.2f} TL" for l in lower_bound],
                "Olası En Yüksek": [f"{u:.2f} TL" for u in upper_bound],
                "Günlük Tahmini Getiri (%)": [f"%{((p/current_price)-1)*100:.2f}" for p in future_prices]
            })
            st.table(tahmin_tablosu)

        with t2:
            st.subheader("Model Doğruluk Analizi")
            kiyas_df = pd.DataFrame({'Gerçek': train_data['Target_1d'].tail(20), 'AI': train_data['AI_Pred_1d'].tail(20)})
            st.bar_chart(kiyas_df)
            st.info(f"Ortalama Mutlak Hata (MAE): %{mean_absolute_error(train_data['Target_1d'], train_data['AI_Pred_1d'])*100:.4f}")

        with t3:
            st.subheader("Tam Teknik Analiz Seti")
            # 1. Fiyat ve Bollinger
            fig1, ax1 = plt.subplots(figsize=(12, 4))
            ax1.plot(train_data.index[-100:], train_data['Close'].tail(100), label="Fiyat", color='white')
            ax1.plot(train_data.index[-100:], train_data['SMA_20'].tail(100), label="SMA 20", color='orange', alpha=0.7)
            ax1.fill_between(train_data.index[-100:], train_data['BB_Lower'].tail(100), train_data['BB_Upper'].tail(100), color='gray', alpha=0.2, label="Bollinger")
            ax1.set_facecolor('#0E1117'); fig1.patch.set_facecolor('#0E1117'); ax1.tick_params(colors='white'); plt.legend()
            st.pyplot(fig1)
            
            # 2. RSI
            st.write("**RSI (Relative Strength Index)**")
            st.area_chart(train_data['RSI'].tail(100))
            
            # 3. Hacim
            st.write("**İşlem Hacmi**")
            st.bar_chart(train_data['Volume'].tail(100))

        with t4:
            if financials is not None and not financials.empty:
                st.bar_chart(financials.loc[['Total Revenue', 'Net Income']].T)

        with t5:
            train_data['Sinyal'] = np.where((train_data['AI_Pred_1d'] > 0), 1, 0)
            train_data['Strategy_Cum'] = (1 + (train_data['Target_1d'] * train_data['Sinyal'])).cumprod() * nakit
            st.line_chart(train_data['Strategy_Cum'])

        with t6:
            st.subheader("📊 BIST10 AI Kapışması")
            if len(izleme_listesi) > 0:
                karsilastirma = []
                with st.spinner("BIST10 verileri optimize edilerek çekiliyor..."):
                    for s_hisse in izleme_listesi:
                        try:
                            t_d, c_d, _, _ = v19_veri_hazirla(s_hisse, "1y") # Hızlı analiz için 1y
                            m_1, _, m_p, sc, f_l = model_merkezi(t_d)
                            l_r = sc.transform(c_d[f_l])
                            s_p = m_1.predict(l_r)[0]
                            s_pr = m_p.predict_proba(l_r)[0, 1]
                            karsilastirma.append({"Hisse": s_hisse, "Güven (%)": round(s_pr*100,2), "Beklenti (%)": round(s_p*100,2), "RSI": round(t_d['RSI'].iloc[-1],2)})
                        except: continue
                
                df_comp = pd.DataFrame(karsilastirma).sort_values(by="Güven (%)", ascending=False)
                st.dataframe(df_comp, use_container_width=True)
                st.bar_chart(df_comp.set_index("Hisse")["Güven (%)"])
            else: st.info("Hisse seçiniz.")

except Exception as e:
    st.error(f"Hata: {e}")

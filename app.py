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
st.set_page_config(page_title="A.S.T. Ultra Terminal v18", layout="wide")

# Telegram API (Bilgilerin aynen korunuyor)
TELEGRAM_TOKEN = "8438099476:AAHWz26Y0bnInuskr_Qjgno4TjjiHOpJ7ao"
CHAT_ID = "5026797450"

# BIST30 Listesi (Otomatik Seçim İçin)
BIST30 = [
    'AKBNK.IS', 'ARCLK.IS', 'ASELS.IS', 'ASTOR.IS', 'BIMAS.IS', 'EKGYO.IS', 'ENJSA.IS', 
    'ENKAI.IS', 'EREGL.IS', 'FROTO.IS', 'GARAN.IS', 'GUBRF.IS', 'HALKB.IS', 'HEKTS.IS', 
    'ISCTR.IS', 'KCHOL.IS', 'KONTR.IS', 'KOZAL.IS', 'KRDMD.IS', 'PETKM.IS', 'PGSUS.IS', 
    'SAHOL.IS', 'SASA.IS', 'SISE.IS', 'TAVHL.IS', 'TCELL.IS', 'THYAO.IS', 'TOASO.IS', 
    'TUPRS.IS', 'YKBNK.IS'
]

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

st.title("🛡️ A.S.T. Ultra v18: BIST30 Kıyaslama Terminali")

# --- 2. VERİ VE MODEL MOTORU ---
@st.cache_data
def v18_veri_hazirla(ticker, period):
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

# --- 3. SIDEBAR (Kayıt ve Seçim Alanı) ---
st.sidebar.header("📁 Takip & Kıyaslama")
# Birden fazla hisseyi kayıt etmek için multiselect
izleme_listesi = st.sidebar.multiselect("Kıyaslanacak BIST30 Hisseleri", BIST30, default=['THYAO.IS', 'ASELS.IS', 'EREGL.IS'])

hisse = st.sidebar.selectbox("Detaylı Analiz Odak Noktası", izleme_listesi if izleme_listesi else ["THYAO.IS"])
nakit = st.sidebar.number_input("Bakiye", value=1000)

# --- 4. ANA AKIŞ ---
try:
    # --- ANA HİSSE ANALİZİ (Eski Fonksiyonların Tamamı) ---
    train_data, current_data, financials, info = v18_veri_hazirla(hisse, "2y")

    if train_data is not None:
        m1, m7, mp, scaler, f_list = model_merkezi(train_data)
        last_row_scaled = scaler.transform(current_data[f_list])
        p_1d, p_7d, prob = m1.predict(last_row_scaled)[0], m7.predict(last_row_scaled)[0], mp.predict_proba(last_row_scaled)[0, 1]
        son_tarih, hedef_tarih = current_data.index[-1].strftime('%d.%m.%Y'), (current_data.index[-1] + pd.Timedelta(days=1)).strftime('%d.%m.%Y')

        st.subheader(f"🏁 {hisse} Stratejik Özet ({son_tarih})")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Son Fiyat", f"{current_data['Close'].iloc[-1]:.2f} TL")
        c2.metric("Tahmin Hedefi", hedef_tarih)
        c3.metric("Yükseliş Güveni", f"%{prob*100:.1f}")
        c4.metric("Beklenen Değişim", f"%{p_1d*100:.2f}")

        # Telegram Butonu (Korundu)
        if st.sidebar.button("Telegram Sinyali Gönder"):
            if prob >= 0.70:
                mesaj = f"🚀 *A.S.T. SİNYALİ*\n\n📈 *Hisse:* {hisse}\n🔥 *Güven:* %{prob*100:.1f}\n💰 *Hedef:* %{p_1d*100:.2f}"
                telegram_sinyal_gonder(mesaj)
                st.sidebar.success("Sinyal Gönderildi!")
            else: st.sidebar.warning("Güven yetersiz.")

        # --- SEKMELER (Yeni Karşılaştırma Sekmesi Eklendi) ---
        t1, t2, t3, t4, t5, t6 = st.tabs(["🔮 Gelecek Kahini", "🤖 AI Performans", "📈 Teknik Analiz", "🏢 Şirket Röntgeni", "🛡️ Risk Yönetimi", "📊 BIST30 Kıyaslama"])
        
        # t1-t5 Arası Kodlar (Aynen Korundu)
        with t1:
            st.subheader("7 Günlük Fiyat Projeksiyonu")
            future_dates = pd.date_range(start=current_data.index[-1] + pd.Timedelta(days=1), periods=7)
            future_prices = [current_data['Close'].iloc[-1] * (1 + (p_7d/7) * i) for i in range(1, 8)]
            vol = current_data['Volatility'].iloc[-1]
            upper_bound = [p * (1 + vol * np.sqrt(i)) for i, p in enumerate(future_prices, 1)]
            lower_bound = [p * (1 - vol * np.sqrt(i)) for i, p in enumerate(future_prices, 1)]
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(train_data.index[-20:], train_data['Close'].tail(20), label="Geçmiş", color='#00d4ff')
            ax.plot(future_dates, future_prices, label="AI Tahmini", linestyle='--', color='#ff4b4b')
            ax.fill_between(future_dates, lower_bound, upper_bound, color='#ff4b4b', alpha=0.1)
            ax.set_facecolor('#0E1117'); fig.patch.set_facecolor('#0E1117'); ax.tick_params(colors='white')
            st.pyplot(fig)

        with t2:
            kiyas_df = pd.DataFrame({'Gerçek': train_data['Target_1d'].tail(15), 'AI': train_data['AI_Pred_1d'].tail(15)})
            st.bar_chart(kiyas_df)
            st.info(f"MAE: %{mean_absolute_error(train_data['Target_1d'], train_data['AI_Pred_1d'])*100:.4f}")

        with t3:
            col1, col2 = st.columns(2)
            col1.area_chart(train_data['RSI'].tail(100))
            col2.line_chart(train_data[['Close', 'BB_Upper', 'BB_Lower']].tail(100))

        with t4:
            if financials is not None and not financials.empty: st.bar_chart(financials.loc[['Total Revenue', 'Net Income']].T)

        with t5:
            train_data['Sinyal'] = np.where((train_data['AI_Pred_1d'] > 0), 1, 0)
            train_data['Strategy_Cum'] = (1 + (train_data['Target_1d'] * train_data['Sinyal'])).cumprod() * nakit
            st.line_chart(train_data[['Strategy_Cum']])

        # --- YENİ: t6 - BIST30 KIYASLAMA EKRANI ---
        with t6:
            st.subheader("📊 Seçili Hisseler Arası AI Kapışması")
            if len(izleme_listesi) > 1:
                karsilastirma_verisi = []
                with st.spinner("Hisseler kapıştırılıyor, lütfen bekleyiniz..."):
                    for s_hisse in izleme_listesi:
                        try:
                            t_data, c_data, _, _ = v18_veri_hazirla(s_hisse, "1y") # Hızlı analiz için 1y
                            m_1, _, m_p, sc, f_l = model_merkezi(t_data)
                            l_row = sc.transform(c_data[f_l])
                            s_p_1d = m_1.predict(l_row)[0]
                            s_prob = m_p.predict_proba(l_row)[0, 1]
                            karsilastirma_verisi.append({
                                "Hisse": s_hisse,
                                "Son Fiyat": f"{c_data['Close'].iloc[-1]:.2f}",
                                "Yarın Beklenti": round(s_p_1d * 100, 2),
                                "Yükseliş Güveni (%)": round(s_prob * 100, 2),
                                "RSI": round(t_data['RSI'].iloc[-1], 2)
                            })
                        except: continue
                
                compare_df = pd.DataFrame(karsilastirma_verisi)
                if not compare_df.empty:
                    # Tabloyu göster
                    st.dataframe(compare_df.sort_values(by="Yükseliş Güveni (%)", ascending=False), use_container_width=True)
                    
                    # Grafiksel Karşılaştırma
                    st.subheader("🔥 AI Güven Puanı Kıyaslaması")
                    st.bar_chart(compare_df.set_index("Hisse")["Yükseliş Güveni (%)"])
                else: st.warning("Karşılaştırma verisi çekilemedi.")
            else: st.info("Karşılaştırma yapmak için sidebar'dan en az 2 hisse seçmelisiniz.")

except Exception as e:
    st.error(f"Sistem Hatası: {e}")

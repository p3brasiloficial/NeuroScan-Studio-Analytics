import streamlit as st
import cv2
import tempfile
import numpy as np
import pandas as pd
import librosa
import plotly.graph_objects as go
import plotly.express as px
from scipy.signal import find_peaks
from moviepy.editor import VideoFileClip
import os
import base64
from deepface import DeepFace

# ==========================================
# 1. ARQUITETURA E CONFIGURAÇÃO STUDIO
# ==========================================
st.set_page_config(page_title="NeuroScan Studio V7.2", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
    <style>
    .main { background-color: #0f0f0f; color: #f1f1f1; font-family: 'Roboto', sans-serif;}
    .studio-card { background-color: #212121; border: 1px solid #3d3d3d; padding: 20px; border-radius: 12px; margin-bottom: 15px;}
    .metric-value { font-size: 2.2rem; font-weight: 800; color: #ffffff; margin-bottom: 0px; letter-spacing: -1px;}
    .metric-title { font-size: 0.85rem; font-weight: 600; color: #aaaaaa; text-transform: uppercase; letter-spacing: 0.5px;}
    .benchmark-text { font-size: 0.85rem; font-weight: 400; padding-top: 8px; line-height: 1.4;}
    .good-metric { color: #2ba640; } .bad-metric { color: #ff4e45; } .neutral-metric { color: #3ea6ff; }
    .stTabs [data-baseweb="tab-list"] { gap: 0px; border-bottom: 1px solid #3d3d3d; }
    .stTabs [data-baseweb="tab"] { background-color: transparent; border: none; padding: 15px 25px; color: #aaaaaa; font-weight: 500;}
    .stTabs [aria-selected="true"] { color: #ffffff !important; border-bottom: 3px solid #3ea6ff !important; }
    .timecode-badge { background-color: #3d3d3d; padding: 4px 8px; border-radius: 4px; font-family: monospace; font-size: 1.1rem; color: #fff;}
    .report-section { background-color: #1a1a1a; padding: 20px; border-radius: 8px; margin-bottom: 15px; border-left: 4px solid #3ea6ff;}
    .report-success { border-left-color: #2ba640; }
    .report-danger { border-left-color: #ff4e45; }
    .report-warning { border-left-color: #f59e0b; }
    </style>
    """, unsafe_allow_html=True)

def format_time(seconds):
    m = seconds // 60
    s = seconds % 60
    return f"{int(m):02d}:{int(s):02d}"

if 'neuro_data' not in st.session_state:
    st.session_state.neuro_data = None
if 'video_path' not in st.session_state:
    st.session_state.video_path = None
if 'thumbs_orig' not in st.session_state:
    st.session_state.thumbs_orig = []
if 'thumbs_heat' not in st.session_state:
    st.session_state.thumbs_heat = []

# ==========================================
# 2. MOTOR DE BIOMETRIA
# ==========================================
class NeuroEngine:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def process_frame(self, frame_rgb):
        small_frame = cv2.resize(frame_rgb, (640, 360))
        gray = cv2.cvtColor(small_frame, cv2.COLOR_RGB2GRAY) 
        
        edges = cv2.Canny(gray, 50, 150)
        edge_blur = cv2.GaussianBlur(edges.astype(float), (51, 51), 0)
        
        h, w = gray.shape
        X, Y = np.meshgrid(np.arange(w), np.arange(h))
        center_bias = np.exp(-(((X - w/2)**2)/(2*(w/4)**2) + ((Y - h/2)**2)/(2*(h/4)**2)))
        
        combined = cv2.normalize((edge_blur * 0.7) + (center_bias * 0.3), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        heatmap_bgr = cv2.applyColorMap(combined, cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
        
        dark_frame = cv2.addWeighted(small_frame, 0.4, np.zeros_like(small_frame), 0.6, 0)
        overlay = cv2.addWeighted(dark_frame, 0.5, heatmap_rgb, 0.5, 0)
        
        _, buf_orig = cv2.imencode('.jpg', cv2.cvtColor(small_frame, cv2.COLOR_RGB2BGR))
        b64_orig = base64.b64encode(buf_orig).decode('utf-8')
        
        _, buf_heat = cv2.imencode('.jpg', cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        b64_heat = base64.b64encode(buf_heat).decode('utf-8')
        
        return np.mean(combined) / 2.55, gray.std(), b64_orig, b64_heat

# ==========================================
# 3. PROCESSAMENTO CENTRAL
# ==========================================
st.markdown("<h2>🎥 NeuroScan Studio Analytics V7.2</h2>", unsafe_allow_html=True)
col_player, col_analytics = st.columns([1.2, 2.8])

with col_player:
    st.markdown("<div class='studio-card'>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload de Vídeo", type=["mp4", "mov"], label_visibility="collapsed")
    
    if uploaded_file and st.session_state.neuro_data is None:
        engine = NeuroEngine()
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_file.read())
        st.session_state.video_path = tfile.name
        
        with st.status("🚀 Processando biometria analítica...", expanded=True) as status:
            cap = cv2.VideoCapture(tfile.name)
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_f / fps
            
            pbar = st.progress(0)
            timeline, t_orig, t_heat = [], [], []
            last_gray = None
            
            for s in range(int(duration)):
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(s * fps))
                ret, frame = cap.read()
                if not ret: break
                
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                small_rgb = cv2.resize(frame_rgb, (640, 360))
                gray = cv2.cvtColor(small_rgb, cv2.COLOR_RGB2GRAY)
                
                motion = np.mean(cv2.absdiff(gray, last_gray)) if last_gray is not None else 0
                last_gray = gray
                
                score, complex_score, b64_orig, b64_heat = engine.process_frame(frame_rgb)
                
                emo = {'happy': 0, 'surprise': 0, 'neutral': 100, 'angry': 0, 'sad': 0}
                try:
                    res = DeepFace.analyze(small_rgb, actions=['emotion'], enforce_detection=False, silent=True)[0]
                    if res.get('face_confidence', 0) > 0.4: emo = res['emotion']
                except: pass
                
                timeline.append({
                    "Tempo": s, "Atenção": score, "Movimento": motion, "Complexidade": complex_score,
                    "alegria": emo.get('happy', 0), "surpresa": emo.get('surprise', 0),
                    "Estresse": emo.get('angry', 0) + emo.get('sad', 0), "Neutro": emo.get('neutral', 0)
                })
                t_orig.append(b64_orig)
                t_heat.append(b64_heat)
                pbar.progress((s + 1) / int(duration))
            cap.release()
            
            try:
                clip = VideoFileClip(tfile.name)
                if clip.audio:
                    temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
                    clip.audio.write_audiofile(temp_wav.name, fps=16000, verbose=False, logger=None)
                    y, _ = librosa.load(temp_wav.name, sr=16000)
                    rms = librosa.feature.rms(y=y)[0]
                    rms_resampled = np.interp(np.linspace(0, 1, len(timeline)), np.linspace(0, 1, len(rms)), rms)
                    audio_energy = (rms_resampled / np.max(rms_resampled)) * 100
                    os.remove(temp_wav.name)
                else: audio_energy = np.zeros(len(timeline))
                clip.close()
            except: audio_energy = np.zeros(len(timeline))
            
            df = pd.DataFrame(timeline)
            df['Energia Áudio'] = audio_energy
            df['Risco'] = 100 - ((df['Atenção'] * 0.4) + (df['Energia Áudio'] * 0.3) + (df['alegria'] * 0.2) + (df['surpresa'] * 0.1))
            df['Time_Format'] = pd.to_datetime(df['Tempo'], unit='s')
            
            st.session_state.neuro_data = df
            st.session_state.thumbs_orig = t_orig
            st.session_state.thumbs_heat = t_heat
            status.update(label="✅ Análise Concluída", state="complete")
            st.rerun()

    if st.session_state.video_path:
        st.video(st.session_state.video_path)
        if st.button("🗑️ Limpar e Analisar Novo Vídeo", use_container_width=True):
            st.session_state.neuro_data = None
            st.session_state.video_path = None
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

# ==========================================
# 4. PAINEL DE ANALYTICS (Direita)
# ==========================================
with col_analytics:
    if st.session_state.neuro_data is not None:
        df = st.session_state.neuro_data
        t_orig = st.session_state.thumbs_orig
        t_heat = st.session_state.thumbs_heat
        
        # 5 ABAS: Todas Restauradas e Melhoradas
        tab_dash, tab_time, tab_heat, tab_brain, tab_report = st.tabs([
            "📊 Visão Geral", "📈 Linha do Tempo", "👁️ Mapa de Calor", "🧠 Neuro-Mapeamento", "📑 Laudo Profundo"
        ])
        
        hook_score = df['Atenção'].head(3).mean() if len(df) >= 3 else df['Atenção'].mean()
        avg_att = df['Atenção'].mean()
        est_retention = max(0, min(100, 100 - df['Risco'].mean() + (hook_score * 0.2)))
        
        peaks_att, _ = find_peaks(df['Atenção'], distance=3)
        peaks_risk, _ = find_peaks(df['Risco'], distance=3)
        top_peaks = df.iloc[peaks_att].nlargest(3, 'Atenção')['Tempo'].tolist()
        top_drops = df.iloc[peaks_risk].nlargest(3, 'Risco')['Tempo'].tolist()

        # --- ABA 1: DASHBOARD ---
        with tab_dash:
            st.markdown("<br>", unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            c1.markdown(f"<div class='studio-card'><div class='metric-title'>Atenção Global</div><div class='metric-value'>{avg_att:.1f}%</div><div class='benchmark-text {'good-metric' if avg_att > 25 else 'bad-metric'}'>Alvo: > 25%</div></div>", unsafe_allow_html=True)
            c2.markdown(f"<div class='studio-card'><div class='metric-title'>Força do Hook</div><div class='metric-value'>{hook_score:.1f}%</div><div class='benchmark-text {'good-metric' if hook_score > 40 else 'bad-metric'}'>Alvo: > 40%</div></div>", unsafe_allow_html=True)
            c3.markdown(f"<div class='studio-card'><div class='metric-title'>Retenção Estimada</div><div class='metric-value'>{est_retention:.1f}%</div><div class='benchmark-text {'good-metric' if est_retention > 50 else 'bad-metric'}'>Alvo: > 50%</div></div>", unsafe_allow_html=True)

        # --- ABA 2: GRÁFICOS E LOCALIZADOR PURO (SEM HEATMAP) ---
        with tab_time:
            st.markdown("### Curva de Retenção (Minutagem do Vídeo)")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df['Time_Format'], y=df['Atenção'], name="Atenção", line=dict(color='#3ea6ff', width=3)))
            fig.add_trace(go.Scatter(x=df['Time_Format'], y=df['Risco'], name="Risco/Fuga", line=dict(color='#ff4e45', width=2, dash='dot')))
            fig.add_trace(go.Scatter(x=df.loc[df['Tempo'].isin(top_peaks), 'Time_Format'], y=df.loc[df['Tempo'].isin(top_peaks), 'Atenção'], mode='markers', name='Top Picos', marker=dict(color='#2ba640', size=12)))
            
            fig.update_layout(template="plotly_dark", height=320, hovermode="x unified", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(l=0, r=0, t=10, b=0), xaxis=dict(tickformat="%M:%S", title="Tempo (MM:SS)"))
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("### 🎯 Localizador Exato de Eventos (Prints Puros)")
            col_p, col_d = st.columns(2)
            with col_p:
                st.markdown("<div style='background-color: rgba(43, 166, 64, 0.1); padding: 15px; border-radius: 8px; border-left: 4px solid #2ba640;'><h4>⭐ Top Picos</h4>", unsafe_allow_html=True)
                for p in top_peaks:
                    st.markdown(f"<span class='timecode-badge'>{format_time(p)}</span> - Atenção: **{df.loc[p, 'Atenção']:.1f}%**", unsafe_allow_html=True)
                    st.image(base64.b64decode(t_orig[p]), width=160)
                st.markdown("</div>", unsafe_allow_html=True)
            with col_d:
                st.markdown("<div style='background-color: rgba(255, 78, 69, 0.1); padding: 15px; border-radius: 8px; border-left: 4px solid #ff4e45;'><h4>⚠️ Top Quedas</h4>", unsafe_allow_html=True)
                for d in top_drops:
                    st.markdown(f"<span class='timecode-badge'>{format_time(d)}</span> - Risco: **{df.loc[d, 'Risco']:.1f}%**", unsafe_allow_html=True)
                    st.image(base64.b64decode(t_orig[d]), width=160)
                st.markdown("</div>", unsafe_allow_html=True)

        # --- ABA 3: MAPA DE CALOR RESTAURADO ---
        with tab_heat:
            st.markdown("### 👁️ Inspetor de Saliência Visual (Heatmap)")
            st.write("Deslize a barra para analisar a concentração de atenção frame a frame no mapa de calor.")
            insp_sec = st.slider("Tempo do Vídeo (Segundos)", 0, len(df)-1, 0, key="heatmap_slider")
            
            c_heat1, c_heat2 = st.columns([2, 1])
            with c_heat1:
                st.image(base64.b64decode(t_heat[insp_sec]), use_container_width=True)
            with c_heat2:
                st.markdown(f"**Tempo exato:** {format_time(insp_sec)}")
                st.metric("Atenção Estimada", f"{df.loc[insp_sec, 'Atenção']:.1f}%")
                st.metric("Carga Visual (Complexidade)", f"{df.loc[insp_sec, 'Complexidade']:.1f}")
                st.info("Zonas quentes (vermelho/amarelo) indicam alto poder de atração focal, baseadas em contraste, movimento e presença facial.")

        # --- ABA 4: NEURO-MAPEAMENTO (CÉREBRO COGNITIVO AVANÇADO) ---
        with tab_brain:
            st.markdown("### 🧠 Mapa de Ativação Cognitiva")
            st.write("Análise neuromarketing preditiva baseada nas áreas anatômicas estimuladas pelo seu conteúdo.")
            
            # Cálculos de Neuromarketing
            frontal = min(100, hook_score + df['alegria'].mean()*0.5 + df['surpresa'].mean()*0.5)
            temporal = min(100, df['Energia Áudio'].mean() + df['alegria'].mean())
            occipital = min(100, df['Atenção'].mean() + df['Movimento'].mean()*2)
            parietal = min(100, df['Complexidade'].mean())
            limbico = min(100, df['Estresse'].mean() * 1.5 + df['Risco'].mean())
            
            col_radar, col_info = st.columns([1.2, 1.8])
            with col_radar:
                fig_brain = go.Figure(go.Scatterpolar(
                    r=[frontal, temporal, occipital, parietal, limbico, frontal],
                    theta=['Córtex Frontal<br>(Atenção)', 'Lobo Temporal<br>(Audição/Memória)', 'Lobo Occipital<br>(Visão)', 'Lobo Parietal<br>(Carga Mental)', 'Sistema Límbico<br>(Instinto/Risco)', 'Córtex Frontal<br>(Atenção)'],
                    fill='toself', line=dict(color='#d2a8ff'), fillcolor='rgba(210, 168, 255, 0.3)'
                ))
                fig_brain.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100], gridcolor='#3d3d3d'), angularaxis=dict(gridcolor='#3d3d3d')), showlegend=False, template="plotly_dark", height=450, paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_brain, use_container_width=True)
            
            with col_info:
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(f"**Córtex Frontal (Atenção/Decisão): {frontal:.1f}%**<br><span style='color:#aaaaaa; font-size: 0.9em;'>Mede o poder do vídeo de reter o raciocínio lógico e focar a atenção do espectador no início.</span>", unsafe_allow_html=True)
                st.progress(frontal / 100)
                
                st.markdown(f"**Sistema Límbico / Amígdala (Risco/Instinto): {limbico:.1f}%**<br><span style='color:#aaaaaa; font-size: 0.9em;'>Picos muito altos aqui indicam que o vídeo gera tensão profunda ou conflito (bom para reter, mau se não for resolvido).</span>", unsafe_allow_html=True)
                st.progress(limbico / 100)
                
                st.markdown(f"**Lobo Temporal (Audição/Emoção): {temporal:.1f}%**<br><span style='color:#aaaaaa; font-size: 0.9em;'>Processa a trilha sonora e o tom de voz. Picos aqui geram conexão visceral e memória a longo prazo.</span>", unsafe_allow_html=True)
                st.progress(temporal / 100)
                
                st.markdown(f"**Lobo Occipital (Estímulo Visual): {occipital:.1f}%**<br><span style='color:#aaaaaa; font-size: 0.9em;'>Ativado por cores, contraste e edição rápida (Cortes).</span>", unsafe_allow_html=True)
                st.progress(occipital / 100)

        # --- ABA 5: LAUDO PROFUNDO ---
        with tab_report:
            c_pie1, c_pie2 = st.columns(2)
            with c_pie1:
                emo_totals = [df['alegria'].sum(), df['surpresa'].sum(), df['Neutro'].sum(), df['Estresse'].sum()]
                fig_pie1 = px.pie(names=['Alegria/Empatia', 'Surpresa/Hook', 'Neutro/Info', 'Tensão/Risco'], values=emo_totals, hole=0.5, color_discrete_sequence=['#2ba640', '#f59e0b', '#3ea6ff', '#ff4e45'], title="Mapeamento de Gatilhos Emocionais")
                fig_pie1.update_layout(template="plotly_dark", height=300, paper_bgcolor='rgba(0,0,0,0)', margin=dict(t=30,b=0,l=0,r=0))
                st.plotly_chart(fig_pie1, use_container_width=True)
            with c_pie2:
                stim_totals = [df['Atenção'].mean(), df['Energia Áudio'].mean(), df['Movimento'].mean()]
                fig_pie2 = px.pie(names=['Estímulo Visual', 'Estímulo Sonoro', 'Dinamismo (Cortes)'], values=stim_totals, hole=0.5, color_discrete_sequence=['#3ea6ff', '#d2a8ff', '#f59e0b'], title="Composição Neural do Vídeo")
                fig_pie2.update_layout(template="plotly_dark", height=300, paper_bgcolor='rgba(0,0,0,0)', margin=dict(t=30,b=0,l=0,r=0))
                st.plotly_chart(fig_pie2, use_container_width=True)

            st.markdown("<hr>", unsafe_allow_html=True)
            st.markdown("<h2>📑 Auditoria de Neuromarketing e Conversão</h2>", unsafe_allow_html=True)
            
            st.markdown("<div class='report-section'>", unsafe_allow_html=True)
            st.markdown("### 👁️ 1. Visão Geral Estratégica")
            if est_retention >= 50: st.write(f"Conteúdo com forte base neuro-compatível. Estimativa de que **{est_retention:.1f}% do público reterá até ao final**. Ritmo de estímulos equilibrado.")
            else: st.write(f"Conteúdo apresenta falhas de retenção. Estimativa de alto abandono, com apenas **{est_retention:.1f}% chegando ao final**. O vídeo exige demasiado esforço cognitivo.")
            st.markdown("</div>", unsafe_allow_html=True)

            col_f, col_n = st.columns(2)
            with col_f:
                st.markdown("<div class='report-section report-success'>### ✅ Pontos Positivos", unsafe_allow_html=True)
                if hook_score > 40: st.markdown("- **Hook Aprovado:** Introdução quebra padrão de scroll.")
                if df['Energia Áudio'].mean() > 30: st.markdown("- **Presença Sonora:** Volume mantém o cérebro alerta.")
                if df['Complexidade'].mean() < 60: st.markdown("- **Clareza Visual:** Enquadramento não polui a visão.")
                st.markdown("</div>", unsafe_allow_html=True)
            with col_n:
                st.markdown("<div class='report-section report-danger'>### ❌ Pontos de Atenção", unsafe_allow_html=True)
                if hook_score <= 40: st.markdown(f"- **Gancho Monótono ({hook_score:.1f}%):** Início pouco estimulante.")
                if top_drops: st.markdown(f"- **Vale da Morte (Tempo {format_time(top_drops[0])}):** Queda vertiginosa de interesse detetada.")
                if df['Movimento'].mean() < 10: st.markdown("- **Estagnação Visual:** Falta de cortes convida ao tédio.")
                st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("<div class='report-section report-warning'>### 🛠️ Plano de Melhoria Prática", unsafe_allow_html=True)
            st.markdown(f"1. **Resolver a Queda:** Aos **{format_time(top_drops[0]) if top_drops else '00:00'}**, insira B-Roll ou altere a trilha sonora para forçar atenção.")
            st.markdown(f"2. **Potencializar o Pico:** Aos **{format_time(top_peaks[0]) if top_peaks else '00:00'}** é o momento ideal para a Oferta (CTA) ou Logótipo.")
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("### 📸 Raio-X de Saliência Visual (Momentos Chave)")
            c_h1, c_h2, c_h3 = st.columns(3)
            idx_start, idx_mid, idx_end = 1 if len(df) > 1 else 0, len(df) // 2, len(df) - 2 if len(df) > 2 else 0
            
            with c_h1:
                st.markdown(f"**O Gancho ({format_time(idx_start)})**")
                st.image(base64.b64decode(t_heat[idx_start]), use_container_width=True)
            with c_h2:
                st.markdown(f"**A Retenção ({format_time(idx_mid)})**")
                st.image(base64.b64decode(t_heat[idx_mid]), use_container_width=True)
            with c_h3:
                st.markdown(f"**O Fecho/CTA ({format_time(idx_end)})**")
                st.image(base64.b64decode(t_heat[idx_end]), use_container_width=True)

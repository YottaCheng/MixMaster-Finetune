import streamlit as st
import numpy as np
import librosa
import matplotlib.pyplot as plt
from scipy import signal
import io
import base64
from PIL import Image
import tempfile
import os
import matplotlib as mpl
from matplotlib.font_manager import FontProperties

# 设置页面配置 - 必须是第一个Streamlit命令
st.set_page_config(
    page_title="音频频率分析器 (Pro-Q3风格)",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 设置中文字体支持
def set_chinese_font():
    # 尝试设置中文字体，根据操作系统设置合适的字体
    try:
        # Windows系统
        if os.name == 'nt':
            font_paths = [
                'C:/Windows/Fonts/simhei.ttf',  # 黑体
                'C:/Windows/Fonts/simsun.ttc',  # 宋体
                'C:/Windows/Fonts/msyh.ttc'     # 微软雅黑
            ]
            for font_path in font_paths:
                if os.path.exists(font_path):
                    font = FontProperties(fname=font_path)
                    return font
        
        # macOS系统
        elif os.name == 'posix' and os.uname().sysname == 'Darwin':
            font_paths = [
                '/System/Library/Fonts/PingFang.ttc',
                '/Library/Fonts/Arial Unicode.ttf'
            ]
            for font_path in font_paths:
                if os.path.exists(font_path):
                    font = FontProperties(fname=font_path)
                    return font
        
        # Linux系统
        else:
            font_paths = [
                '/usr/share/fonts/truetype/arphic/uming.ttc',
                '/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf'
            ]
            for font_path in font_paths:
                if os.path.exists(font_path):
                    font = FontProperties(fname=font_path)
                    return font
    except:
        pass
    
    # 如果没有找到合适的中文字体，使用系统默认sans-serif字体
    return FontProperties(family='sans-serif')

# 获取中文字体
chinese_font = set_chinese_font()

# 配置matplotlib使用中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 自定义CSS
st.markdown("""
<style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1000px;
    }
    h1 {
        color: #00AAFF;
    }
    .stButton > button {
        background-color: #00AAFF;
        color: white;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    .stButton > button:hover {
        background-color: #0088CC;
        color: white;
    }
    .info-box {
        background-color: #222222;
        border-radius: 5px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .stImage {
        background-color: #1E1E1E;
        border-radius: 5px;
        padding: 5px;
    }
    
    /* 自定义数据指标样式 */
    .metric-container {
        background-color: #1E1E1E;
        border-radius: 10px;
        padding: 10px;
        margin: 5px 0;
    }
    .metric-label {
        font-size: 14px;
        color: #888;
        margin-bottom: 5px;
    }
    .metric-value {
        font-size: 18px;
        font-weight: bold;
        color: #00AAFF;
    }
</style>
""", unsafe_allow_html=True)


def analyze_audio(audio_file):
    """
    分析音频文件并提取频率特征
    """
    # 加载音频文件
    y, sr = librosa.load(audio_file, sr=None)
    
    # 计算短时傅里叶变换
    n_fft = 4096
    hop_length = 1024
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    
    # 转换为功率谱
    S = np.abs(D) ** 2
    S_db = librosa.power_to_db(S, ref=np.max)
    
    # 获取频率轴
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    
    # 计算平均频谱
    mean_spectrum = np.mean(S, axis=1)
    
    # 频谱平滑处理
    window_size = min(51, len(mean_spectrum) - 1)
    if window_size % 2 == 0:
        window_size -= 1
    
    if window_size >= 3:
        mean_spectrum_smooth = signal.savgol_filter(mean_spectrum, window_size, 3)
    else:
        mean_spectrum_smooth = mean_spectrum
    
    # 转换为分贝
    spectrum_db = librosa.amplitude_to_db(mean_spectrum_smooth)
    
    # 标准化到[-12, 12]范围内,模拟EQ调整范围
    max_db = np.max(spectrum_db)
    min_db = np.min(spectrum_db)
    spectrum_normalized = (spectrum_db - min_db) / (max_db - min_db) * 24 - 12
    
    # 定义频段
    bands = [
        {"name": "低频", "range": (20, 250), "color": "#4080FF"},
        {"name": "中频", "range": (250, 4000), "color": "#40CC40"},
        {"name": "高频", "range": (4000, 20000), "color": "#FF8040"}
    ]
    
    # 计算各频段能量
    total_energy = np.sum(mean_spectrum)
    band_energies = {}
    
    for band in bands:
        low_freq, high_freq = band["range"]
        low_idx = np.argmin(np.abs(freqs - low_freq)) if low_freq > 0 else 0
        high_idx = np.argmin(np.abs(freqs - high_freq)) if high_freq < sr/2 else len(freqs)-1
        
        band_energy = np.sum(mean_spectrum[low_idx:high_idx+1])
        energy_percent = (band_energy / total_energy) * 100 if total_energy > 0 else 0
        band_energies[band["name"]] = energy_percent
    
    # 确定主导频段
    dominant_band = max(band_energies.items(), key=lambda x: x[1])[0]
    
    # 创建频率曲线图
    plt.figure(figsize=(12, 6))
    
    # 设置黑色背景
    plt.style.use('dark_background')
    ax = plt.gca()
    ax.set_facecolor('#1E1E1E')
    plt.gcf().set_facecolor('#1E1E1E')
    
    # 绘制网格
    plt.grid(True, which="both", ls="-", color='#333333', alpha=0.7)
    
    # 绘制频率响应曲线
    plt.semilogx(freqs, spectrum_normalized, linewidth=3, color='#00AAFF')
    
    # 填充曲线下方区域
    plt.fill_between(freqs, -12, spectrum_normalized, alpha=0.3, color='#00AAFF')
    
    # 设置x轴范围和刻度(对数刻度)
    plt.xlim(20, 20000)
    plt.xticks([20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000],
             ['20', '50', '100', '200', '500', '1k', '2k', '5k', '10k', '20k'])
    
    # 设置y轴范围和刻度
    plt.ylim(-12, 12)
    plt.yticks([-12, -9, -6, -3, 0, 3, 6, 9, 12])
    
    # 设置标签，使用中文字体
    plt.xlabel('频率 (Hz)', fontproperties=chinese_font, fontsize=12, color='white')
    plt.ylabel('增益 (dB)', fontproperties=chinese_font, fontsize=12, color='white')
    plt.title('频率响应曲线 (Pro-Q3 风格)', fontproperties=chinese_font, fontsize=14, color='white')
    
    # 标记频段
    for band in bands:
        plt.axvspan(band["range"][0], band["range"][1], alpha=0.1, color=band["color"])
        x_pos = np.sqrt(band["range"][0] * band["range"][1])  # 取几何平均值作为标签位置
        plt.annotate(band["name"], xy=(x_pos, 11), color=band["color"], 
                     ha='center', fontsize=10, fontproperties=chinese_font,
                     bbox=dict(boxstyle="round,pad=0.3", fc='#1E1E1E', ec=band["color"], alpha=0.7))
    
    # 将图像转换为PIL图像
    curve_buffer = io.BytesIO()
    plt.savefig(curve_buffer, format='png', dpi=100, facecolor='#1E1E1E')
    curve_buffer.seek(0)
    curve_image = Image.open(curve_buffer)
    plt.close()
    
    # 创建波形图
    plt.figure(figsize=(12, 2))
    plt.style.use('dark_background')
    ax = plt.gca()
    ax.set_facecolor('#1E1E1E')
    plt.gcf().set_facecolor('#1E1E1E')
    
    plt.plot(np.linspace(0, len(y)/sr, len(y)), y, color='#00AAFF', linewidth=0.8, alpha=0.7)
    plt.xlim(0, len(y)/sr)
    plt.title('波形图', fontproperties=chinese_font, color='white')
    plt.xlabel('时间 (秒)', fontproperties=chinese_font, color='white')
    plt.grid(False)
    
    # 将波形图转换为PIL图像
    wave_buffer = io.BytesIO()
    plt.savefig(wave_buffer, format='png', dpi=100, facecolor='#1E1E1E')
    wave_buffer.seek(0)
    wave_image = Image.open(wave_buffer)
    plt.close()
    
    return {
        "curve_image": curve_image,
        "wave_image": wave_image,
        "low": round(band_energies['低频'], 1),
        "mid": round(band_energies['中频'], 1),
        "high": round(band_energies['高频'], 1),
        "sr": sr,
        "duration": round(len(y)/sr, 2),
        "dominant": dominant_band,
        "filename": os.path.basename(audio_file) if isinstance(audio_file, str) else "上传的音频"
    }

def process_audio(audio_data):
    """处理上传的音频数据"""
    if audio_data is None:
        return None
    
    try:
        # 创建临时文件
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            # 写入音频数据
            tmp_file.write(audio_data)
            tmp_path = tmp_file.name
        
        # 执行分析
        result = analyze_audio(tmp_path)
        
        # 删除临时文件
        os.unlink(tmp_path)
        
        return result
    except Exception as e:
        st.error(f"处理音频时出错: {str(e)}")
        return None

# 创建一个缓存函数来避免重复分析
@st.cache_data
def cached_process_audio(audio_bytes):
    """缓存音频处理结果以提高性能"""
    # 使用字节内容作为缓存键
    return process_audio(audio_bytes)

# 主页面布局
st.title("音频频率分析器 (Pro-Q3风格)")
st.write("上传音频文件，分析其频率特性并生成可视化结果")

# 创建两列布局
col1, col2 = st.columns([1, 2])

with col1:
    # 音频上传
    uploaded_file = st.file_uploader("上传音频文件", type=['wav', 'mp3', 'ogg', 'flac'])
    
    # 分析按钮
    analyze_button = st.button("分析音频", use_container_width=True)
    
    # 创建音频预览
    if uploaded_file is not None:
        st.audio(uploaded_file, format=f"audio/{uploaded_file.name.split('.')[-1]}")
    
    # 信息显示区域
    st.markdown("### 频段能量分布")
    
    # 预留结果显示区域
    low_freq_container = st.empty()
    mid_freq_container = st.empty()
    high_freq_container = st.empty()
    
    st.markdown("### 音频信息")
    sample_rate_container = st.empty()
    duration_container = st.empty()
    dominant_band_container = st.empty()
    filename_container = st.empty()

with col2:
    # 预留曲线图显示区域
    freq_curve_container = st.empty()
    wave_container = st.empty()

# 使用说明
with st.expander("使用说明", expanded=False):
    st.markdown("""
    ## 使用说明
    
    1. 点击上传按钮或拖放音频文件到上传区域
    2. 点击"分析"按钮处理音频
    3. 查看生成的频率响应曲线和波形图
    4. 频段能量分布显示低频(20-250Hz)、中频(250-4000Hz)和高频(4000-20000Hz)的能量比例
    
    频率响应曲线模拟了专业音频均衡器FabFilter Pro-Q3的显示风格，可以直观地看出音频的频率特性。
    """)

# 处理按钮点击事件
if uploaded_file is not None and analyze_button:
    with st.spinner('正在分析音频...'):
        # 读取上传的文件
        audio_bytes = uploaded_file.getvalue()
        
        # 处理音频
        result = cached_process_audio(audio_bytes)
        
        if result:
            # 显示频率曲线图 - 使用use_container_width替代已弃用的use_column_width
            freq_curve_container.image(result["curve_image"], caption="频率响应曲线", use_container_width=True)
            wave_container.image(result["wave_image"], caption="波形图", use_container_width=True)
            
            # 使用自定义HTML显示频段分析
            low_freq_container.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">低频 (%)</div>
                <div class="metric-value">{result["low"]}</div>
            </div>
            """, unsafe_allow_html=True)
            
            mid_freq_container.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">中频 (%)</div>
                <div class="metric-value">{result["mid"]}</div>
            </div>
            """, unsafe_allow_html=True)
            
            high_freq_container.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">高频 (%)</div>
                <div class="metric-value">{result["high"]}</div>
            </div>
            """, unsafe_allow_html=True)
            
            # 显示音频信息
            sample_rate_container.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">采样率 (Hz)</div>
                <div class="metric-value">{result["sr"]}</div>
            </div>
            """, unsafe_allow_html=True)
            
            duration_container.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">持续时间 (秒)</div>
                <div class="metric-value">{result["duration"]}</div>
            </div>
            """, unsafe_allow_html=True)
            
            dominant_band_container.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">主导频段</div>
                <div class="metric-value">{result["dominant"]}</div>
            </div>
            """, unsafe_allow_html=True)
            
            filename_container.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">文件名</div>
                <div class="metric-value" style="font-size: 14px;">{result["filename"]}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.error("无法处理音频文件。请确保上传的是有效的音频文件。")
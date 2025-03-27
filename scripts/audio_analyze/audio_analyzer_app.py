import gradio as gr
import numpy as np
import librosa
import matplotlib.pyplot as plt
from scipy import signal
import io
import base64
from PIL import Image

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
    
    # 设置标签
    plt.xlabel('频率 (Hz)', color='white', fontsize=12)
    plt.ylabel('增益 (dB)', color='white', fontsize=12)
    plt.title('频率响应曲线 (Pro-Q3 风格)', color='white', fontsize=14)
    
    # 标记频段
    for band in bands:
        plt.axvspan(band["range"][0], band["range"][1], alpha=0.1, color=band["color"])
        x_pos = np.sqrt(band["range"][0] * band["range"][1])  # 取几何平均值作为标签位置
        plt.annotate(band["name"], xy=(x_pos, 11), color=band["color"], 
                     ha='center', fontsize=10, bbox=dict(boxstyle="round,pad=0.3", 
                                                    fc='#1E1E1E', ec=band["color"], alpha=0.7))
    
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
    plt.title('波形图', color='white')
    plt.xlabel('时间 (秒)', color='white')
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
        "filename": audio_file.split("/")[-1] if "/" in audio_file else audio_file.split("\\")[-1] if "\\" in audio_file else audio_file
    }

def process_audio(audio):
    if audio is None:
        return None, None, 0, 0, 0, 0, 0, "无数据", ""
    
    try:
        # 执行分析
        result = analyze_audio(audio)
        
        return (
            result["curve_image"], 
            result["wave_image"],
            result["low"],
            result["mid"],
            result["high"],
            result["sr"],
            result["duration"],
            result["dominant"],
            result["filename"]
        )
    except Exception as e:
        return None, None, 0, 0, 0, 0, 0, f"错误: {str(e)}", ""

# 创建Gradio界面
with gr.Blocks() as demo:
    gr.Markdown("# 音频频率分析器 (Pro-Q3风格)")
    gr.Markdown("上传音频文件，分析其频率特性并生成可视化结果")
    
    with gr.Row():
        with gr.Column(scale=1):
            audio_input = gr.Audio(type="filepath", label="上传音频文件")
            analyze_button = gr.Button("分析")
            
            # 使用普通分组代替Box
            with gr.Column():
                gr.Markdown("### 频段能量分布")
                with gr.Row():
                    low_freq = gr.Number(label="低频 (%)", precision=1)
                    mid_freq = gr.Number(label="中频 (%)", precision=1)
                    high_freq = gr.Number(label="高频 (%)", precision=1)
                
                with gr.Row():
                    sample_rate = gr.Number(label="采样率 (Hz)")
                    duration = gr.Number(label="持续时间 (秒)", precision=2)
                
                with gr.Row():
                    dominant_band = gr.Textbox(label="主导频段")
                    filename = gr.Textbox(label="文件名")
        
        with gr.Column(scale=2):
            freq_image = gr.Image(label="频率响应曲线", type="pil")
            wave_image = gr.Image(label="波形图", type="pil")
    
    # 绑定分析按钮
    analyze_button.click(
        process_audio,
        inputs=audio_input,
        outputs=[freq_image, wave_image, low_freq, mid_freq, high_freq, 
                 sample_rate, duration, dominant_band, filename]
    )
    
    # 使用说明
    with gr.Accordion("使用说明", open=False):
        gr.Markdown("""
        ## 使用说明
        
        1. 点击上传按钮或拖放音频文件到上传区域
        2. 点击"分析"按钮处理音频
        3. 查看生成的频率响应曲线和波形图
        4. 频段能量分布显示低频(20-250Hz)、中频(250-4000Hz)和高频(4000-20000Hz)的能量比例
        
        频率响应曲线模拟了专业音频均衡器FabFilter Pro-Q3的显示风格，可以直观地看出音频的频率特性。
        """)

if __name__ == "__main__":
    demo.launch(
        share=True,              # 创建可共享链接
        server_name="0.0.0.0",   # 绑定到所有网络接口
        server_port=7860         # 指定端口
    )
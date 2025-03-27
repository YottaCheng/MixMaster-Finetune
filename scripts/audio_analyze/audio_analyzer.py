import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from scipy import signal
import io
import base64

def analyze_audio(audio_file):
    """
    分析音频文件并提取频率特征
    
    参数:
        audio_file: 音频文件的路径
        
    返回:
        包含频率分析结果的字典
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
    
    # 创建频率曲线图
    curve_img = create_frequency_curve(freqs, spectrum_normalized, bands, band_energies)
    
    # 创建波形图
    waveform_img = create_waveform_image(y, sr)
    
    # 返回分析结果
    return {
        "audio_info": {
            "sample_rate": sr,
            "duration": len(y) / sr,
            "channels": 1 if y.ndim == 1 else y.shape[0]
        },
        "frequency_data": {
            "frequencies": freqs.tolist(),
            "spectrum": spectrum_normalized.tolist()
        },
        "band_energies": band_energies,
        "dominant_band": max(band_energies.items(), key=lambda x: x[1])[0],
        "curve_image": curve_img,
        "waveform_image": waveform_img
    }

def create_frequency_curve(freqs, spectrum, bands, band_energies):
    """生成Pro-Q3风格的频率响应曲线图像"""
    plt.figure(figsize=(12, 8))
    
    # 设置黑色背景
    plt.style.use('dark_background')
    ax = plt.gca()
    ax.set_facecolor('#1E1E1E')
    plt.gcf().set_facecolor('#1E1E1E')
    
    # 绘制网格
    plt.grid(True, which="both", ls="-", color='#333333', alpha=0.7)
    
    # 绘制频率响应曲线
    plt.semilogx(freqs, spectrum, linewidth=3, color='#00AAFF')
    
    # 填充曲线下方区域
    plt.fill_between(freqs, -12, spectrum, alpha=0.3, color='#00AAFF')
    
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
    
    # 在图表右侧添加能量分布信息
    energy_text = "\n".join([
        f"{band}: {energy:.1f}%" for band, energy in band_energies.items()
    ])
    plt.figtext(0.92, 0.5, energy_text, color='white', fontsize=10,
              bbox=dict(boxstyle="round,pad=0.5", fc='#333333', ec='#666666', alpha=0.8),
              verticalalignment='center')
    
    # 将图像转换为base64
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=100, facecolor='#1E1E1E')
    buffer.seek(0)
    img_str = base64.b64encode(buffer.read()).decode('utf-8')
    plt.close()
    
    return img_str

def create_waveform_image(y, sr):
    """创建音频波形图"""
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
    
    # 将波形图转换为base64
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=100, facecolor='#1E1E1E')
    buffer.seek(0)
    img_str = base64.b64encode(buffer.read()).decode('utf-8')
    plt.close()
    
    return img_str

if __name__ == "__main__":
    import sys
    import json
    
    if len(sys.argv) < 2:
        print("使用方法: python audio_analyzer.py <音频文件路径>")
        sys.exit(1)
    
    audio_path = sys.argv[1]
    
    try:
        result = analyze_audio(audio_path)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        print(f"\n分析完成! 主导频段: {result['dominant_band']}")
        print(f"低频: {result['band_energies']['低频']:.1f}%, 中频: {result['band_energies']['中频']:.1f}%, 高频: {result['band_energies']['高频']:.1f}%")
    except Exception as e:
        print(f"分析过程中发生错误: {str(e)}")
# SpeechPulse - Project Specification

## 1. Project Overview

### 1.1 Name & Tagline
**SpeechPulse** - 基于Qwen2-Audio的语音情感理解MCP Server

### 1.2 Mission
填补OpenClaw/Agent生态中语音情感分析与语义理解的空白，让AI Agent能"听懂"语音消息中的情绪、意图和紧急程度。

### 1.3 Value Proposition

**核心差异化：**
- 🏆 **这是Agent/MCP生态中第一个语音情感分析工具** —— 填补OpenClaw/Agent生态的空白
- 🔬 **现有GitHub上的SER项目全部是独立学术代码**，没有封装成MCP Server或Agent Skill，无法直接集成到Agent工作流
- 🏗️ **不是用librosa提特征+规则的拼凑方案**，而是分层架构，从轻量规则到端到端AudioLLM，满足不同场景需求

**功能对比：**
- **现有方案**：仅支持ASR（语音转文字）和TTS（文字转语音）
- **SpeechPulse**：在转写基础上增加情感标签、紧急程度评估、讽刺检测

---

## 2. Architecture Overview

### 2.1 System Architecture (ASCII)

```
┌─────────────────────────────────────────────────────────────────┐
│                        MCP Client                               │
│                    (OpenClaw/Agent)                             │
│              Claude / Cline / Continue.dev                      │
└───────────────────────┬─────────────────────────────────────────┘
                        │ MCP Protocol (stdio/sse)
                        │ tools/list, tools/call
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│                     SpeechPulse MCP Server                      │
│                     (mcp SDK: FastMCP)                          │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              Registered MCP Tools                       │   │
│  │  ┌─────────────┐ ┌─────────────┐ ┌───────────────────┐ │   │
│  │  │analyze_audio│ │assess_urgency│ │  detect_sarcasm   │ │   │
│  │  └─────────────┘ └─────────────┘ └───────────────────┘ │   │
│  │  ┌───────────────────────────────────────────────────┐ │   │
│  │  │              full_analysis                        │ │   │
│  │  └───────────────────────────────────────────────────┘ │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                  │
│                              ▼                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              Analysis Engine Router                     │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐ │   │
│  │  │  Tier Lite  │  │Tier Standard│  │    Tier Pro     │ │   │
│  │  │ (stdlib)    │  │(emotion2vec │  │(Qwen2-Audio-7B) │ │   │
│  │  │             │  │ + Whisper)  │  │                 │ │   │
│  │  └─────────────┘  └─────────────┘  └─────────────────┘ │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                  │
│                              ▼                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  GET /health (Additional HTTP endpoint for monitoring)  │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Data Flow

```
Audio File (wav/mp3/flac)
         │
         ▼
┌─────────────────┐
│  Audio Loader   │ ──► Format validation, resampling (16kHz)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Feature Extract │ ──► Pitch, Energy, MFCC, Zero-crossing
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Tier Router    │ ──► Select Lite/Standard/Pro based on config
└────────┬────────┘
         │
    ┌────┴────┬────────┐
    ▼         ▼        ▼
┌───────┐ ┌───────┐ ┌───────────┐
│ Lite  │ │ Std   │ │ Pro       │
│ Rules │ │ ML    │ │ LLM       │
└───┬───┘ └───┬───┘ └─────┬─────┘
    │         │           │
    └─────────┴───────────┘
              │
              ▼
┌─────────────────────────┐
│    Result Aggregator    │ ──► Merge/Normalize outputs
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│    JSON Response        │ ──► MCP Tool Result
└─────────────────────────┘
```

---

## 3. File Structure & Responsibilities

```
speechpulse/
├── README.md                    # GitHub展示文档（中英双语）
├── pyproject.toml               # Python包配置，仅需mcp SDK，零ML依赖
├── LICENSE                      # MIT License
├── .gitignore
├── specs/
│   ├── spec.md                  # 本文件：完整项目规格
│   ├── tasks.md                 # 任务分解与排期
│   └── checklist.md             # 验收清单
├── src/
│   └── speechpulse/             # 主包目录 (src-layout)
│       ├── __init__.py          # Package init, exports main classes
│       ├── types.py             # 核心数据类型定义 (dataclasses)
│       ├── server.py            # MCP Server主入口
│       ├── audio_features.py    # 音频特征提取（Lite/Std tier共用）
│       ├── emotion.py           # 情感分析引擎（多tier实现）
│       ├── urgency.py           # 紧急程度评估模块
│       ├── sarcasm.py           # 讽刺检测（文本-语音情感对比）
│       ├── analyzer.py          # 完整分析管道（SpeechAnalyzer统一入口）
│       ├── config.py            # 配置管理（环境变量/JSON配置）
│       └── utils.py             # 工具函数（音频加载、格式转换等）
├── skills/
│   └── SKILL.md                 # OpenClaw Skill定义文件
├── examples/
│   └── demo.py                  # 生成测试音频+运行分析演示
└── tests/
    └── test_all.py              # pytest测试套件
```

### 3.1 File Details

#### `src/speechpulse/__init__.py`
```python
# Exports for package users
from .types import AudioFeatures, EmotionResult, UrgencyResult, SarcasmResult
from .analyzer import SpeechAnalyzer
from .emotion import EmotionAnalyzer
from .urgency import UrgencyAssessor
from .sarcasm import SarcasmDetector

__version__ = "0.1.0"
__all__ = [
    "AudioFeatures", "EmotionResult", "UrgencyResult", "SarcasmResult",
    "SpeechAnalyzer", "EmotionAnalyzer", "UrgencyAssessor", "SarcasmDetector"
]
```

#### `src/speechpulse/types.py`
- **职责**：核心数据类型定义，集中管理所有dataclass
- **内容**：
  - `AudioFeatures`: 音频特征数据类
  - `EmotionResult`: 情感分析结果
  - `UrgencyResult`: 紧急程度评估结果
  - `SarcasmResult`: 讽刺检测结果
- **设计原则**：所有模块共享同一类型定义，避免循环导入

#### `src/speechpulse/server.py`
- **职责**：MCP Server主入口，实现标准MCP协议
- **实现**：使用`mcp` Python SDK (FastMCP)
- **协议**：标准MCP协议 (Model Context Protocol)
  - `tools/list` - 列出可用工具
  - `tools/call` - 调用指定工具
- **注册的Tools**：
  - `analyze_audio` - 转写+情感分析
  - `assess_urgency` - 紧急程度评估
  - `detect_sarcasm` - 讽刺检测
  - `full_analysis` - 完整分析
- **额外端点**：
  - `GET /health` - HTTP健康检查（用于监控）
- **依赖**：`pip install mcp`
- **代码示例**：
```python
from mcp.server.fastmcp import FastMCP
from .types import AudioFeatures, EmotionResult

mcp = FastMCP("speechpulse")

@mcp.tool()
def analyze_audio(audio_path: str, tier: str = "lite", text: str = None) -> dict:
    """
    Analyze audio emotion and transcription.
    
    Args:
        audio_path: Path to audio file
        tier: Analysis tier (lite, standard, pro)
        text: Optional transcription text (required for lite tier sarcasm detection)
    
    Returns:
        Dict containing transcription, emotion, and features
    """
    analyzer = get_analyzer(tier)
    return analyzer.analyze(audio_path, text)

@mcp.tool()
def assess_urgency(audio_path: str, tier: str = "lite") -> dict:
    """Assess urgency level from audio."""
    analyzer = get_analyzer(tier)
    return analyzer.assess_urgency(audio_path)

@mcp.tool()
def detect_sarcasm(audio_path: str, tier: str = "lite", text: str = None) -> dict:
    """
    Detect sarcasm by comparing text and audio emotion.
    
    Args:
        audio_path: Path to audio file
        tier: Analysis tier
        text: Transcription text (required for lite tier)
    """
    analyzer = get_analyzer(tier)
    return analyzer.detect_sarcasm(audio_path, text)

@mcp.tool()
def full_analysis(audio_path: str, tier: str = "lite", text: str = None) -> dict:
    """Perform full analysis including emotion, urgency, and sarcasm."""
    analyzer = get_analyzer(tier)
    return analyzer.full_analysis(audio_path, text)

if __name__ == "__main__":
    mcp.run()
```

#### `src/speechpulse/audio_features.py`
- **职责**：音频特征提取
- **Lite Tier**：纯stdlib实现（wave/struct/math）
  - 提取：基频轮廓、能量包络、过零率、静音比例
- **Std/Pro Tier**：librosa增强
  - 额外：MFCC、频谱质心、韵律特征

#### `src/speechpulse/emotion.py`
- **职责**：情感分析引擎，支持多tier
- **Lite**：规则引擎（使用相对阈值，避免性别差异）
  - 高能量+高基频 → 兴奋/愤怒
  - 低能量+低基频 → 悲伤/疲惫
  - 能量波动大 → 焦虑/紧张
- **Standard**：emotion2vec (FunASR) + 分类器
- **Pro**：Qwen2-Audio-7B-Instruct (4-bit)

#### `src/speechpulse/urgency.py`
- **职责**：紧急程度评估 (0-1分数)
- **输入**：音频特征 + 转写文本（如有）
- **Lite规则**：
  - 语速快 + 音量高 + 音调高 → 高紧急
  - 停顿少 + 连续说话 → 高紧急
- **ML增强**：基于历史数据训练的评分模型

#### `src/speechpulse/sarcasm.py`
- **职责**：讽刺/反话检测
- **原理**：对比文本情感极性 vs 语音情感
  - 文本正面 + 语音负面 → 可能讽刺
  - 语调平坦 + 文本夸张 → 可能讽刺
- **Lite**：依赖用户传入的text参数进行对比
- **Pro**：LLM推理

#### `src/speechpulse/analyzer.py`
- **职责**：完整分析管道，提供统一入口
- **类**：`SpeechAnalyzer`
- **方法**：
  - `analyze(audio_path: str, text: str = None) -> Dict` - 情感分析
  - `assess_urgency(audio_path: str) -> UrgencyResult` - 紧急程度
  - `detect_sarcasm(audio_path: str, text: str = None) -> SarcasmResult` - 讽刺检测
  - `full_analysis(audio_path: str, text: str = None) -> Dict` - 完整分析
- **设计原则**：整合audio_features、emotion、urgency、sarcasm四个模块

#### `src/speechpulse/config.py`
- **职责**：配置管理模块
- **功能**：
  - 从环境变量读取配置（SPEECHPULSE_TIER, SPEECHPULSE_PORT等）
  - 从JSON配置文件读取
  - 配置验证和默认值
  - 全局配置单例

#### `src/speechpulse/utils.py`
- **职责**：通用工具函数（含音频加载）
- **设计原则**：Lite模式使用标准库`array.array`或`List[float]`，不依赖numpy
- **功能**：
  - WAV音频加载（纯stdlib: wave/struct，支持16-bit/24-bit PCM）
  - 音频格式检测与转换
  - 重采样到16kHz（使用array.array，不依赖numpy）
  - 路径安全检查（防目录遍历）
  - 音频分段/切片
- **函数签名示例**：
```python
from array import array
from typing import List, Tuple, Union

def load_audio_stdlib(path: str) -> Tuple[array, int]:
    """
    Load audio using only Python standard library.
    
    Returns:
        Tuple of (audio_samples, sample_rate)
        audio_samples is array.array('f') - array of floats
    """
    ...

def resample_audio(
    audio: Union[array, List[float]], 
    orig_sr: int, 
    target_sr: int
) -> array:
    """
    Resample audio to target sample rate.
    
    Args:
        audio: Audio samples as array.array or List[float]
        orig_sr: Original sample rate
        target_sr: Target sample rate
    
    Returns:
        Resampled audio as array.array('f')
    """
    ...

def ensure_16khz(audio: Union[array, List[float]], sr: int) -> array:
    """Ensure audio is at 16kHz sample rate."""
    if sr != 16000:
        return resample_audio(audio, sr, 16000)
    return audio
```

#### `skills/SKILL.md`
- **职责**：OpenClaw Skill定义
- **内容**：工具描述、参数schema、使用示例

#### `examples/demo.py`
- **职责**：演示脚本
- **功能**：
  - 生成合成测试音频（不同情感/紧急度）
  - 调用MCP Server进行分析
  - 打印格式化结果

---

## 4. Interface Definitions

### 4.1 MCP Tool Schemas

#### Tool: `analyze_audio`

**Request:**
```json
{
  "jsonrpc": "2.0",
  "method": "analyze_audio",
  "params": {
    "audio_path": "/path/to/audio.wav",
    "tier": "lite",
    "language": "zh",
    "text": "可选：用户提供的转写文本（Lite tier需要此参数获取转写）"
  },
  "id": 1
}
```

**Response (Lite Tier - 无ASR能力):**
```json
{
  "jsonrpc": "2.0",
  "result": {
    "transcription": null,
    "note": "Lite tier does not include ASR. Use 'text' param to provide transcription, or upgrade to Standard/Pro tier.",
    "emotion": {
      "primary": "happy",
      "confidence": 0.87,
      "secondary": "excited"
    },
    "speaker_state": {
      "energy_level": "high",
      "stress_indicator": "low"
    },
    "features": {
      "duration_sec": 3.5,
      "pitch_mean": 220,
      "energy_mean": 0.65
    }
  },
  "id": 1
}
```

**Response (Standard/Pro Tier - 含ASR):**
```json
{
  "jsonrpc": "2.0",
  "result": {
    "transcription": "我今天真的很开心",
    "emotion": {
      "primary": "happy",
      "confidence": 0.87,
      "secondary": "excited"
    },
    "speaker_state": {
      "energy_level": "high",
      "stress_indicator": "low"
    },
    "features": {
      "duration_sec": 3.5,
      "pitch_mean": 220,
      "energy_mean": 0.65
    }
  },
  "id": 1
}
```

#### Tool: `assess_urgency`

**Request:**
```json
{
  "jsonrpc": "2.0",
  "method": "assess_urgency",
  "params": {
    "audio_path": "/path/to/audio.wav",
    "tier": "lite",
    "context": "customer_support"
  },
  "id": 2
}
```

**Response:**
```json
{
  "jsonrpc": "2.0",
  "result": {
    "urgency_score": 0.82,
    "level": "high",
    "reasoning": [
      "语速比平时快40%",
      "音量持续偏高",
      "停顿时间极短"
    ],
    "recommended_action": "优先处理"
  },
  "id": 2
}
```

#### Tool: `detect_sarcasm`

**Request:**
```json
{
  "jsonrpc": "2.0",
  "method": "detect_sarcasm",
  "params": {
    "audio_path": "/path/to/audio.wav",
    "tier": "lite",
    "text": "需要检测的文本内容（Lite tier必需，Standard/Pro可选）"
  },
  "id": 3
}
```

**Response (Lite Tier - 依赖用户传入text):**
```json
{
  "jsonrpc": "2.0",
  "result": {
    "is_sarcastic": true,
    "confidence": 0.73,
    "indicators": [
      "文本情感极性为正面，语音情感为负面",
      "语调异常平坦"
    ],
    "text_emotion": "positive",
    "audio_emotion": "negative",
    "note": "Lite tier requires 'text' parameter for sarcasm detection"
  },
  "id": 3
}
```

**Response (Standard/Pro Tier - 自动转写):**
```json
{
  "jsonrpc": "2.0",
  "result": {
    "is_sarcastic": true,
    "confidence": 0.73,
    "indicators": [
      "文本情感极性为正面，语音情感为负面",
      "语调异常平坦"
    ],
    "text_emotion": "positive",
    "audio_emotion": "negative",
    "transcription": "这真是太棒了"  // 自动转写结果
  },
  "id": 3
}
```

#### Tool: `full_analysis`

**Request:**
```json
{
  "jsonrpc": "2.0",
  "method": "full_analysis",
  "params": {
    "audio_path": "/path/to/audio.wav",
    "tier": "lite",
    "include_interpretation": true
  },
  "id": 4
}
```

**Response (Lite Tier - 无ASR，需用户传入text):**
```json
{
  "jsonrpc": "2.0",
  "result": {
    "summary": "说话者表现出明显的焦虑和紧迫感",
    "transcription": null,
    "note": "Lite tier does not include ASR. Provide 'text' param for transcription and sarcasm detection.",
    "emotion_analysis": {
      "primary": "anxious",
      "confidence": 0.79
    },
    "urgency_assessment": {
      "score": 0.85,
      "level": "high"
    },
    "sarcasm_detection": {
      "is_sarcastic": null,
      "note": "Sarcasm detection requires 'text' parameter in Lite tier"
    },
    "raw_features": { ... }
  },
  "id": 4
}
```

**Response (Standard/Pro Tier - 含ASR与完整分析):**
```json
{
  "jsonrpc": "2.0",
  "result": {
    "summary": "说话者表现出明显的焦虑和紧迫感，可能在投诉或求助",
    "transcription": "这个订单已经三天了还没发货",
    "emotion_analysis": {
      "primary": "anxious",
      "confidence": 0.79
    },
    "urgency_assessment": {
      "score": 0.85,
      "level": "high"
    },
    "sarcasm_detection": {
      "is_sarcastic": false,
      "confidence": 0.91
    },
    "interpretation": "用户语气急促且带有焦虑情绪，建议立即联系处理发货问题",
    "raw_features": { ... }
  },
  "id": 4
}
```

### 4.2 Internal Interfaces (defined in `src/speechpulse/types.py`)

All core data types are centralized in `types.py` to avoid circular imports and ensure consistency across modules.

#### `AudioFeatures` (Dataclass)
```python
@dataclass
class AudioFeatures:
    """Audio prosodic features extracted from waveform."""
    duration_sec: float
    sample_rate: int
    pitch_mean: float          # Mean fundamental frequency (Hz)
    pitch_std: float           # Pitch variation (standard deviation)
    energy_mean: float         # Mean energy (normalized 0-1)
    energy_std: float          # Energy variation
    zero_crossing_rate: float  # Rate of sign changes
    silence_ratio: float       # Proportion of silent frames
    mfcc: Optional[List[float]] = None  # MFCC features (Std/Pro only)
    
    def __repr__(self) -> str:
        return f"AudioFeatures(duration={self.duration_sec:.2f}s, pitch={self.pitch_mean:.1f}Hz)"
```

#### `EmotionResult` (Dataclass)
```python
@dataclass
class EmotionResult:
    """Emotion analysis result."""
    primary: str  # happy, sad, angry, anxious, neutral, excited, tired
    confidence: float          # Confidence score (0-1)
    secondary: Optional[str] = None    # Secondary emotion label
    scores: Dict[str, float] = field(default_factory=dict)  # All emotion scores
    
    def __repr__(self) -> str:
        return f"EmotionResult({self.primary}, confidence={self.confidence:.2f})"
```

#### `UrgencyResult` (Dataclass)
```python
@dataclass
class UrgencyResult:
    """Urgency assessment result."""
    score: float              # Urgency score (0-1)
    level: str                # low, medium, high, critical
    reasoning: List[str]      # Human-readable reasoning
    recommended_action: str   # Suggested action
    
    def __repr__(self) -> str:
        return f"UrgencyResult({self.level}, score={self.score:.2f})"
```

#### `SarcasmResult` (Dataclass)
```python
@dataclass
class SarcasmResult:
    """Sarcasm detection result."""
    is_sarcastic: bool
    confidence: float         # Confidence score (0-1)
    indicators: List[str]     # Detection indicators
    text_emotion: Optional[str] = None   # Text sentiment polarity
    audio_emotion: Optional[str] = None  # Audio emotion detected
    
    def __repr__(self) -> str:
        return f"SarcasmResult(is_sarcastic={self.is_sarcastic}, confidence={self.confidence:.2f})"
```

#### Usage Example
```python
from speechpulse.types import AudioFeatures, EmotionResult

# Create feature instance
features = AudioFeatures(
    duration_sec=3.5,
    sample_rate=16000,
    pitch_mean=220.0,
    pitch_std=45.0,
    energy_mean=0.65,
    energy_std=0.15,
    zero_crossing_rate=0.08,
    silence_ratio=0.12
)

# Create result instance
emotion = EmotionResult(
    primary="happy",
    confidence=0.87,
    secondary="excited"
)
```

---

## 5. Tier Specifications

### 5.1 Tier Comparison Table

| Feature | Lite | Standard | Pro |
|---------|------|----------|-----|
| Dependencies | mcp SDK only (零ML依赖) | librosa, torch, funasr | transformers, torch, dashscope |
| Model Size | N/A | ~2GB | ~6GB (4-bit) |
| GPU Required | No | Optional | Recommended |
| Transcription | No (用户可传入text) | Whisper-base | Qwen2-Audio |
| Emotion Labels | 7 basic | 8 detailed | 12+ nuanced |
| Sarcasm Detection | Rule-based | ML-based | LLM-based |
| Latency | <100ms | 1-3s | 3-10s |
| Accuracy | 60-70% | 75-85% | 85-95% |

### 5.2 Lite Tier Rules

#### Design Decision: Relative Thresholds

**Why relative thresholds instead of absolute values?**

Traditional emotion recognition systems use absolute thresholds (e.g., "pitch > 200Hz = happy"), which have significant limitations:

1. **Gender differences**: Adult male voices typically range 85-180Hz, while female voices range 165-255Hz. Absolute thresholds would misclassify gender.
2. **Individual variation**: Personal speaking habits vary widely.
3. **Recording conditions**: Microphone distance and volume settings affect absolute measurements.

**Solution**: Use relative thresholds based on deviation from personal baseline:
- Calculate mean and standard deviation from the audio sample itself
- Use z-scores (standard deviations from mean) for classification
- Example: "pitch > mean + 1.5σ" instead of "pitch > 200Hz"

#### Emotion Rules (Relative Thresholds)

```python
# Relative thresholds using z-scores (standard deviations from mean)
# This avoids gender bias and individual variation issues

EMOTION_RULES = {
    "happy": {
        "conditions": [
            "pitch_zscore > 1.0",        # Pitch above mean + 1σ
            "energy_zscore > 0.5",       # Energy above mean + 0.5σ
            "pitch_std_zscore > 0.8"     # High pitch variation
        ],
        "weight": 0.8,
        "description": "Elevated pitch with moderate-high energy and variation"
    },
    "excited": {
        "conditions": [
            "pitch_zscore > 1.5",
            "energy_zscore > 1.0",
            "pitch_std_zscore > 1.2"
        ],
        "weight": 0.85,
        "description": "Very high pitch and energy with large variation"
    },
    "angry": {
        "conditions": [
            "energy_zscore > 1.2",       # Very high energy
            "pitch_std_zscore > 1.0",    # High variation
            "zero_crossing_rate > baseline * 1.3"  # Harsh voice quality
        ],
        "weight": 0.9,
        "description": "High energy with harsh voice quality and variation"
    },
    "sad": {
        "conditions": [
            "pitch_zscore < -0.8",       # Pitch below mean
            "energy_zscore < -0.5",      # Low energy
            "pitch_std_zscore < -0.5"    # Low variation (monotone)
        ],
        "weight": 0.85,
        "description": "Low pitch, low energy, monotone"
    },
    "tired": {
        "conditions": [
            "energy_zscore < -0.8",
            "pitch_zscore < -0.3",
            "silence_ratio > baseline * 1.5"  # More pauses
        ],
        "weight": 0.8,
        "description": "Low energy with increased pauses"
    },
    "anxious": {
        "conditions": [
            "pitch_std_zscore > 1.5",    # Very high variation
            "energy_std_zscore > 1.0",   # Unstable energy
            "silence_ratio < baseline * 0.7"  # Fewer pauses (rushed)
        ],
        "weight": 0.75,
        "description": "Unstable pitch and energy, rushed speech"
    },
    "neutral": {
        "conditions": [
            "abs(pitch_zscore) < 0.5",
            "abs(energy_zscore) < 0.5",
            "abs(pitch_std_zscore) < 0.5"
        ],
        "weight": 0.6,
        "description": "All features near baseline"
    }
}

# Z-score calculation (per audio sample)
def calculate_zscore(value: float, mean: float, std: float) -> float:
    """Calculate z-score (standard deviations from mean)."""
    if std == 0:
        return 0.0
    return (value - mean) / std

# Feature normalization (within each audio sample)
def normalize_features(frame_pitches: List[float], frame_energies: List[float],
                       features: AudioFeatures) -> Dict[str, float]:
    """
    Convert absolute features to relative z-scores.
    This makes the system robust to gender and individual differences.
    
    Baseline is calculated from the same audio's frame-level statistics,
    so no external reference or pre-computed baseline is needed.
    """
    import statistics
    
    # Calculate baseline from frame-level data of the same audio
    pitch_baseline = statistics.mean(frame_pitches) if frame_pitches else 0.0
    pitch_baseline_std = statistics.stdev(frame_pitches) if len(frame_pitches) > 1 else 1.0
    
    energy_baseline = statistics.mean(frame_energies) if frame_energies else 0.0
    energy_baseline_std = statistics.stdev(frame_energies) if len(frame_energies) > 1 else 1.0
    
    # Z-scores: how much does each segment deviate from the full-audio baseline
    pitch_zscore = calculate_zscore(features.pitch_mean, pitch_baseline, pitch_baseline_std)
    energy_zscore = calculate_zscore(features.energy_mean, energy_baseline, energy_baseline_std)
    
    return {
        "pitch_zscore": pitch_zscore,
        "energy_zscore": energy_zscore,
        "pitch_std_zscore": calculate_zscore(features.pitch_std, pitch_baseline_std, pitch_baseline_std),
        "energy_std_zscore": calculate_zscore(features.energy_std, energy_baseline_std, energy_baseline_std),
    }
```

#### Urgency Scoring
```python
URGENCY_FACTORS = {
    "speaking_rate": {
        "fast": 0.3,    # > 4 chars/sec (Chinese) or > 150 wpm (English)
        "normal": 0,
        "slow": -0.1
    },
    "volume_level": {
        "high": 0.25,   # energy_mean > 0.6
        "normal": 0,
        "low": -0.05
    },
    "pitch_variation": {
        "high": 0.2,    # pitch_std > 50
        "normal": 0,
        "low": -0.05
    },
    "pause_pattern": {
        "few_pauses": 0.15,  # silence_ratio < 0.15
        "normal": 0,
        "many_pauses": -0.1
    },
    "energy_trend": {
        "increasing": 0.1,
        "stable": 0,
        "decreasing": -0.05
    }
}
# Final score = sigmoid(sum(factors))
```

---

## 6. Configuration

### 6.1 Environment Variables

```bash
# Server Config
SPEECHPULSE_HOST=0.0.0.0
SPEECHPULSE_PORT=8080
SPEECHPULSE_TIER=lite  # lite, standard, pro

# Model Paths (for Standard/Pro tiers)
SPEECHPULSE_WHISPER_MODEL=base
SPEECHPULSE_EMOTION2VEC_MODEL=iic/emotion2vec_base
SPEECHPULSE_QWEN_MODEL=Qwen/Qwen2-Audio-7B-Instruct

# API Keys (for Pro tier with DashScope)
DASHSCOPE_API_KEY=your_key_here

# GPU Config
SPEECHPULSE_USE_GPU=false
SPEECHPULSE_GPU_DEVICE=0
```

### 6.2 Config File (JSON)

```json
{
  "server": {
    "host": "0.0.0.0",
    "port": 8080,
    "workers": 1
  },
  "tier": "lite",
  "models": {
    "whisper": {
      "model_size": "base",
      "language": "zh"
    },
    "emotion2vec": {
      "model_name": "iic/emotion2vec_base",
      "granularity": "utterance"
    },
    "qwen": {
      "model_path": "Qwen/Qwen2-Audio-7B-Instruct",
      "quantization": "4bit",
      "max_length": 512
    }
  },
  "analysis": {
    "default_language": "zh",
    "min_audio_duration": 0.5,
    "max_audio_duration": 300
  }
}
```

---

## 7. Error Handling

### 7.1 Error Codes

| Code | Meaning | Description |
|------|---------|-------------|
| -32700 | Parse error | Invalid JSON |
| -32600 | Invalid Request | Invalid JSON-RPC 2.0 Request |
| -32601 | Method not found | Unknown tool name |
| -32602 | Invalid params | Invalid method parameters |
| -32603 | Internal error | Internal server error |
| -32001 | Audio not found | Audio file does not exist |
| -32002 | Invalid format | Unsupported audio format |
| -32003 | Audio too short | Duration < 0.5s |
| -32004 | Audio too long | Duration > 300s |
| -32005 | Tier unavailable | Requested tier not available |
| -32006 | Model load failed | Failed to load ML model |

### 7.2 Error Response Format

```json
{
  "jsonrpc": "2.0",
  "error": {
    "code": -32001,
    "message": "Audio file not found",
    "data": {
      "path": "/path/to/missing.wav"
    }
  },
  "id": 1
}
```

---

## 8. Security & Privacy

### 8.1 Privacy Guarantees
- 所有分析在本地运行，音频数据不离开用户设备
- 不存储任何音频文件或分析结果
- 支持内存中处理，不写入临时文件（可选）

### 8.2 Security Measures
- 输入路径验证（防止目录遍历）
- 音频格式白名单（wav, mp3, flac, ogg, m4a）
- 文件大小限制（默认50MB）
- 可选：API Key认证

---

## 9. Performance Targets

| Metric | Lite | Standard | Pro |
|--------|------|----------|-----|
| Cold Start | <1s | <5s | <30s |
| Per-Request | <100ms | 1-3s | 3-10s |
| Throughput | 100 req/s | 10 req/s | 2 req/s |
| Memory | <50MB | <3GB | <8GB |

---

## 10. Future Enhancements

1. **Speaker Diarization**: 多说话人情感分析
2. **Real-time Streaming**: 流式音频分析
3. **Custom Models**: 用户自定义情感标签
4. **Multilingual**: 支持更多语言
5. **Voice Biometrics**: 说话人识别+情感关联

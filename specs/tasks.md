# SpeechPulse - Task Breakdown

## 任务总览

| Phase | 名称 | 任务数 | 预计工时 |
|-------|------|--------|----------|
| 1 | 基础设施 | 5 | 4h |
| 2 | Lite Tier核心 | 6 | 8h |
| 3 | MCP Server | 4 | 6h |
| 4 | 高级Tier | 3 | 6h |
| 5 | 集成与测试 | 4 | 4h |
| 6 | 文档与发布 | 3 | 4h |
| **总计** | | **25** | **32h** |

---

## Phase 1: 基础设施 (Foundation)

### Task 1.1: 项目初始化
**优先级**: P0 | **预计工时**: 30min

**描述**:
初始化Python项目结构，配置pyproject.toml，设置.gitignore

**验收标准**:
- [ ] pyproject.toml包含正确的metadata
- [ ] 包名正确：speechpulse
- [ ] 版本号：0.1.0
- [ ] 零必需依赖（extras定义optional依赖）
- [ ] .gitignore包含Python标准忽略项

**产出物**:
- `pyproject.toml`
- `.gitignore`
- `LICENSE` (MIT)

---

### Task 1.2: 目录结构创建
**优先级**: P0 | **预计工时**: 15min

**描述**:
创建完整的项目目录结构

**目录清单**:
```
speechpulse/
├── src/
│   └── speechpulse/          # 主包目录 (src-layout)
│       ├── __init__.py
│       ├── types.py          # 核心数据类型定义
│       ├── server.py         # MCP Server (mcp SDK)
│       ├── audio_features.py
│       ├── emotion.py
│       ├── urgency.py
│       ├── sarcasm.py
│       └── utils.py
├── skills/
│   └── SKILL.md
├── examples/
│   └── demo.py
├── tests/
│   └── test_all.py
└── specs/
    ├── spec.md
    ├── tasks.md
    └── checklist.md
```

**产出物**:
- 所有目录和空文件创建完成

---

### Task 1.3: 类型定义与数据类
**优先级**: P0 | **预计工时**: 45min

**描述**:
定义核心数据类：AudioFeatures, EmotionResult, UrgencyResult, SarcasmResult

**文件**: `src/speechpulse/types.py`

**要求**:
- 使用@dataclass装饰器
- 完整的类型注解
- 可选字段使用Optional
- 包含__repr__方法

**代码模板**:
```python
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

@dataclass
class AudioFeatures:
    duration_sec: float
    sample_rate: int
    pitch_mean: float
    pitch_std: float
    energy_mean: float
    energy_std: float
    zero_crossing_rate: float
    silence_ratio: float
    mfcc: Optional[List[float]] = None
    
    def __repr__(self) -> str:
        return f"AudioFeatures(duration={self.duration_sec:.2f}s, pitch={self.pitch_mean:.1f}Hz)"
```

---

### Task 1.4: 配置管理模块
**优先级**: P1 | **预计工时**: 1h

**描述**:
实现配置加载模块，支持环境变量和配置文件

**文件**: `src/speechpulse/config.py`

**功能**:
- 从环境变量读取配置
- 从JSON配置文件读取
- 配置验证和默认值
- 全局配置单例

**配置项**:
- SPEECHPULSE_HOST, SPEECHPULSE_PORT
- SPEECHPULSE_TIER
- SPEECHPULSE_USE_GPU
- 各tier模型路径

---

### Task 1.5: 工具函数模块
**优先级**: P1 | **预计工时**: 1.5h

**描述**:
实现通用工具函数

**文件**: `src/speechpulse/utils.py`

**函数清单**:
- `validate_audio_path(path: str) -> bool` - 路径安全检查
- `get_audio_format(path: str) -> str` - 检测音频格式
- `load_audio_stdlib(path: str) -> Tuple[array.array, int]` - 纯stdlib加载
- `resample_audio(audio: Union[array.array, List[float]], orig_sr: int, target_sr: int) -> array.array`
- `ensure_16khz(audio: Union[array.array, List[float]], sr: int) -> array.array`

**约束**:
- Lite模式下不使用numpy，统一使用array.array或List[float]

---

## Phase 2: Lite Tier核心 (Core Engine)

### Task 2.1: 纯stdlib音频加载
**优先级**: P0 | **预计工时**: 1.5h

**描述**:
使用Python标准库（wave/struct）实现WAV文件加载

**文件**: `src/speechpulse/utils.py`（音频加载功能合并到工具模块中）

**功能**:
- 读取WAV文件头
- 支持PCM 16-bit, 24-bit
- 转换为float数组（-1.0 ~ 1.0）
- 返回 (samples, sample_rate)

**不使用**:
- numpy
- scipy
- librosa
- soundfile

---

### Task 2.2: 音频特征提取（stdlib版）
**优先级**: P0 | **预计工时**: 2.5h

**描述**:
使用纯Python实现音频特征提取

**文件**: `src/speechpulse/audio_features.py`

**特征列表**:
1. **Duration**: 音频时长
2. **Energy**: 短时能量均值和标准差
3. **Zero Crossing Rate**: 过零率
4. **Pitch**: 基频估计（自相关法）
5. **Silence Ratio**: 静音帧比例

**算法**:
- 基频：自相关函数 + 峰值检测
- 能量：分帧计算RMS
- 过零率：统计符号变化

**输出**: AudioFeatures对象

---

### Task 2.3: 情感分析规则引擎
**优先级**: P0 | **预计工时**: 2h

**描述**:
实现基于规则的情感分析

**文件**: `src/speechpulse/emotion.py`

**情感标签**: happy, sad, angry, anxious, neutral, excited, tired

**规则引擎（使用相对阈值/z-score）**:
```python
# 使用z-score（标准差倍数）而非绝对值，避免性别差异
RULES = {
    "happy": {
        "pitch_zscore": (">", 1.0),      # 高于均值1个标准差
        "energy_zscore": (">", 0.5),
        "pitch_std_zscore": (">", 0.8)
    },
    "angry": {
        "energy_zscore": (">", 1.2),
        "pitch_std_zscore": (">", 1.0),
        "zero_crossing_rate": (">", "baseline * 1.3")
    },
    "sad": {
        "pitch_zscore": ("<", -0.8),
        "energy_zscore": ("<", -0.5),
        "pitch_std_zscore": ("<", -0.5)
    },
    # ... more rules
}

# Z-score计算
def calculate_zscore(value: float, mean: float, std: float) -> float:
    if std == 0:
        return 0.0
    return (value - mean) / std
```

**评分逻辑**:
- 基于z-score的相对阈值匹配
- 加权平均计算各情感得分
- 返回primary + secondary + confidence
- 优势：避免性别差异（男女音高基线不同）

---

### Task 2.4: 紧急程度评估
**优先级**: P0 | **预计工时**: 1.5h

**描述**:
实现紧急程度评分算法

**文件**: `src/speechpulse/urgency.py`

**评分因子**:
- Speaking rate (语速)
- Volume level (音量)
- Pitch variation (音调变化)
- Pause pattern (停顿模式)
- Energy trend (能量趋势)

**算法**:
```python
def calculate_urgency(features: AudioFeatures) -> UrgencyResult:
    score = 0.0
    reasoning = []
    
    # 语速因子
    if speaking_rate > FAST_THRESHOLD:
        score += 0.3
        reasoning.append("语速比平时快40%")
    
    # ... 其他因子
    
    # sigmoid归一化到0-1
    final_score = sigmoid(score)
    level = score_to_level(final_score)
    
    return UrgencyResult(score=final_score, level=level, reasoning=reasoning)
```

---

### Task 2.5: 讽刺检测
**优先级**: P1 | **预计工时**: 1.5h

**描述**:
实现基于文本-语音情感对比的讽刺检测

**文件**: `src/speechpulse/sarcasm.py`

**Lite Tier逻辑**:
- 依赖用户传入的`text`参数（Lite无ASR能力）
- 文本情感极性（简单关键词匹配）
- 语音情感（来自emotion模块）
- 对比：
  - 文本positive + 语音negative → 讽刺可能
  - 语调平坦 + 文本夸张 → 讽刺可能
- 注意：text参数在Lite tier下为必需

**Standard/Pro Tier逻辑**:
- 自动转写音频获取文本
- 进行同样的对比分析

**输出**: SarcasmResult

---

### Task 2.6: 完整分析管道
**优先级**: P1 | **预计工时**: 1h

**描述**:
整合所有分析模块，提供统一入口

**文件**: `src/speechpulse/analyzer.py`

**类**: `SpeechAnalyzer`

**方法**:
- `analyze(audio_path: str) -> Dict` - 情感分析
- `assess_urgency(audio_path: str) -> UrgencyResult`
- `detect_sarcasm(audio_path: str) -> SarcasmResult`
- `full_analysis(audio_path: str) -> Dict` - 完整分析

---

## Phase 3: MCP Server实现

### Task 3.1: MCP Server基础框架
**优先级**: P0 | **预计工时**: 1.5h

**描述**:
使用mcp Python SDK实现标准MCP Server

**文件**: `src/speechpulse/server.py`

**依赖**: `pip install mcp`

**实现**:
```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("speechpulse")

@mcp.tool()
def analyze_audio(audio_path: str, tier: str = "lite", text: str = None) -> dict:
    ...
```

**功能**:
- 使用FastMCP创建MCP Server
- 注册4个tools
- 支持stdio和sse传输

**约束**:
- 不使用Flask/FastAPI/http.server
- 使用标准MCP协议

---

### Task 3.2: MCP Tools注册
**优先级**: P0 | **预计工时**: 1.5h

**描述**:
实现并注册4个MCP Tools

**文件**: `src/speechpulse/server.py`

**Tools**:
1. `analyze_audio` - 转写+情感分析
   - 参数: audio_path, tier, text(optional)
   - Lite tier返回transcription=null
2. `assess_urgency` - 紧急程度评估
   - 参数: audio_path, tier
3. `detect_sarcasm` - 讽刺检测
   - 参数: audio_path, tier, text(Lite必需)
4. `full_analysis` - 完整分析
   - 参数: audio_path, tier, text

**参数验证**:
- audio_path: 必须存在，格式支持
- tier: lite/standard/pro，默认lite
- text: Lite tier下detect_sarcasm必需

---

### Task 3.3: 健康检查端点
**优先级**: P1 | **预计工时**: 1h

**描述**:
添加HTTP健康检查端点（额外功能，非MCP标准）

**文件**: `src/speechpulse/server.py`

**实现**:
- 在MCP Server基础上添加HTTP路由
- GET /health 返回状态
- 用于Docker/K8s健康检查

---

### Task 3.4: Server启动脚本
**优先级**: P1 | **预计工时**: 1h

**描述**:
实现命令行启动脚本

**文件**: `src/speechpulse/__main__.py`

**功能**:
```bash
# stdio模式（默认，用于MCP客户端）
python -m speechpulse

# sse模式（HTTP传输）
python -m speechpulse --transport sse --port 8080

# 指定tier
python -m speechpulse --tier lite
```

**参数**:
- --transport: stdio或sse
- --port, -p: HTTP端口（sse模式）
- --tier, -t: 默认tier
- --config, -c: 配置文件路径

---

## Phase 4: 高级Tier实现

### Task 4.1: Standard Tier集成
**优先级**: P2 | **预计工时**: 3h

**描述**:
集成emotion2vec + Whisper

**依赖** (extras):
```toml
[project.optional-dependencies]
standard = ["torch", "librosa", "funasr", "openai-whisper"]
```

**文件**: `src/speechpulse/tiers/standard.py`

**功能**:
- Whisper转写
- emotion2vec情感识别
- 特征融合

**类**: `StandardAnalyzer`

---

### Task 4.2: Pro Tier集成
**优先级**: P2 | **预计工时**: 2h

**描述**:
集成Qwen2-Audio-7B-Instruct

**依赖** (extras):
```toml
pro = ["torch", "transformers", "dashscope", "bitsandbytes"]
```

**文件**: `src/speechpulse/tiers/pro.py`

**支持**:
- 本地4-bit量化模型
- DashScope API调用

**类**: `ProAnalyzer`

---

### Task 4.3: Tier路由与自动选择
**优先级**: P2 | **预计工时**: 1h

**描述**:
实现Tier自动检测和路由

**文件**: `src/speechpulse/tiers/__init__.py`

**逻辑**:
```python
def get_analyzer(tier: str) -> BaseAnalyzer:
    if tier == "lite":
        return LiteAnalyzer()
    elif tier == "standard":
        if not has_standard_deps():
            raise TierUnavailable("Standard tier requires: pip install speechpulse[standard]")
        return StandardAnalyzer()
    elif tier == "pro":
        # ...
```

---

## Phase 5: 集成与测试

### Task 5.1: 单元测试
**优先级**: P1 | **预计工时**: 1.5h

**描述**:
编写pytest测试套件

**文件**: `tests/test_all.py`

**测试覆盖**:
- AudioFeatures提取
- 情感规则引擎（z-score相对阈值）
- 紧急程度评分
- 讽刺检测（含text参数处理）
- MCP协议解析
- 配置加载

**测试音频**:
- 使用合成音频（numpy生成）
- 或小型真实音频文件

---

### Task 5.2: 演示脚本
**优先级**: P1 | **预计工时**: 1h

**描述**:
创建演示脚本展示功能

**文件**: `examples/demo.py`

**功能**:
- 生成合成测试音频（不同情感）
- 调用MCP Server进行分析
- 打印格式化结果
- 展示不同tier对比

---

### Task 5.3: OpenClaw Skill定义
**优先级**: P1 | **预计工时**: 1h

**描述**:
编写SKILL.md文件

**文件**: `skills/SKILL.md`

**内容**:
- Skill名称和描述
- 工具定义（4个tools）
- 参数schema
- 使用示例
- 配置说明

---

### Task 5.4: 集成测试
**优先级**: P1 | **预计工时**: 30min

**描述**:
端到端测试

**测试流程**:
1. 启动Server
2. 发送analyze_audio请求
3. 验证响应格式
4. 测试错误处理

---

## Phase 6: 文档与发布

### Task 6.1: README编写
**优先级**: P0 | **预计工时**: 2h

**描述**:
编写中英双语README

**文件**: `README.md`

**内容**:
- 项目介绍（What & Why）
- 架构图（ASCII）
- 快速开始
- Demo输出示例
- Tier对比表
- 安装说明
- API文档链接
- 贡献指南

**格式**:
- 英文为主
- 关键段落中英双语

---

### Task 6.2: 代码注释与文档字符串
**优先级**: P1 | **预计工时**: 1h

**描述**:
完善代码文档

**要求**:
- 所有公共函数添加docstring
- 复杂算法添加行内注释
- 注释使用英文

**示例**:
```python
def extract_pitch(audio: array.array, sr: int) -> float:
    """
    Extract fundamental frequency (pitch) using autocorrelation.
    
    Args:
        audio: Audio samples as array of floats (-1.0 to 1.0)
        sr: Sample rate in Hz
        
    Returns:
        Mean pitch value in Hz
        
    Note:
        Uses pure Python implementation without numpy.
    """
```

---

### Task 6.3: 发布准备
**优先级**: P2 | **预计工时**: 1h

**描述**:
准备PyPI发布

**检查清单**:
- [ ] 版本号确认
- [ ] pyproject.toml完整
- [ ] LICENSE文件
- [ ] README渲染测试
- [ ] 构建测试：`python -m build`
- [ ] 安装测试：`pip install -e .`

---

## 依赖关系图

```
Task 1.1 ───┬─── Task 1.2 ───┬─── Task 1.3 ───┬─── Task 1.4 ───┬─── Task 1.5
            │                │                │                │
            ▼                ▼                ▼                ▼
        Task 2.1 ───┬─── Task 2.2 ───┬─── Task 2.3 ───┬─── Task 2.4 ───┬─── Task 2.5 ───┬─── Task 2.6
                    │                │                │                │                │
                    ▼                ▼                ▼                ▼                ▼
                Task 3.1 ───┬─── Task 3.2 ───┬─── Task 3.3 ───┬─── Task 3.4
                            │                │                │
                            ▼                ▼                ▼
                        Task 4.1 ───┬─── Task 4.2 ───┬─── Task 4.3
                                        │                │
                                        ▼                ▼
                                    Task 5.1 ───┬─── Task 5.2 ───┬─── Task 5.3 ───┬─── Task 5.4
                                                    │                │                │
                                                    ▼                ▼                ▼
                                                Task 6.1 ───┬─── Task 6.2 ───┬─── Task 6.3
```

---

## 里程碑

| 里程碑 | 包含任务 | 交付物 |
|--------|----------|--------|
| M1: Lite MVP | 1.1-1.5, 2.1-2.6, 3.1-3.4 | 可运行的Lite Tier + MCP Server |
| M2: Full Tiers | 4.1-4.3 | 完整3-tier支持 |
| M3: Release Ready | 5.1-5.4, 6.1-6.3 | 测试通过 + 文档完整 |

---

## 风险与缓解

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| 纯stdlib音频处理性能差 | 中 | 优化算法，使用array.array而非list |
| 基频估计不准确 | 中 | 实现多种算法，取最可靠结果 |
| Qwen2-Audio模型过大 | 低 | 优先完成Lite，Pro作为可选 |
| emotion2vec依赖复杂 | 低 | 使用funasr简化，提供fallback |

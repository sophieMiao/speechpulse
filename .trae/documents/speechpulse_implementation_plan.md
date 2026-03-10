# SpeechPulse 实现计划

## 项目概述
SpeechPulse - 基于Qwen2-Audio的语音情感理解MCP Server

## 执行规则
1. 严格按照tasks.md的Phase顺序执行：Phase 1 → Phase 2 → Phase 3（先做到M1里程碑：Lite MVP）
2. Phase 4（Standard/Pro Tier）暂不实现，留空或写stub
3. 每完成一个Task后，告知用户完成了什么，然后继续下一个Task
4. 遇到spec中有歧义的地方，优先按照spec.md的定义执行

## 关键技术约束
- MCP Server 使用 `mcp` Python SDK 的 FastMCP，不要用http.server/Flask/FastAPI
- Lite Tier 的音频处理使用纯Python标准库（wave/struct/math/array），不使用numpy/librosa
- 情感规则引擎使用z-score相对阈值，不使用绝对值
- 包结构是 src-layout：src/speechpulse/
- pyproject.toml 的 dependencies 包含 mcp>=1.0

---

## Phase 1: 基础设施 (Foundation)

### Task 1.1: 项目初始化
**优先级**: P0 | **预计工时**: 30min

**产出物**:
- `pyproject.toml` - 包含正确的metadata，包名speechpulse，版本0.1.0，零必需依赖
- `.gitignore` - 包含Python标准忽略项
- `LICENSE` - MIT License

### Task 1.2: 目录结构创建
**优先级**: P0 | **预计工时**: 15min

**产出物**:
创建以下目录结构：
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

### Task 1.3: 类型定义与数据类
**优先级**: P0 | **预计工时**: 45min

**文件**: `src/speechpulse/types.py`

**要求**:
- 使用@dataclass装饰器
- 完整的类型注解
- 可选字段使用Optional
- 包含__repr__方法

**数据类**:
1. `AudioFeatures` - 音频特征数据类
2. `EmotionResult` - 情感分析结果
3. `UrgencyResult` - 紧急程度评估结果
4. `SarcasmResult` - 讽刺检测结果

### Task 1.4: 配置管理模块
**优先级**: P1 | **预计工时**: 1h

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

### Task 1.5: 工具函数模块
**优先级**: P1 | **预计工时**: 1.5h

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

**文件**: `src/speechpulse/utils.py`

**功能**:
- 读取WAV文件头
- 支持PCM 16-bit, 24-bit
- 转换为float数组（-1.0 ~ 1.0）
- 返回 (samples, sample_rate)

**不使用**:
- numpy, scipy, librosa, soundfile

### Task 2.2: 音频特征提取（stdlib版）
**优先级**: P0 | **预计工时**: 2.5h

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

### Task 2.3: 情感分析规则引擎
**优先级**: P0 | **预计工时**: 2h

**文件**: `src/speechpulse/emotion.py`

**情感标签**: happy, sad, angry, anxious, neutral, excited, tired

**规则引擎（使用相对阈值/z-score）**:
- 使用z-score（标准差倍数）而非绝对值，避免性别差异
- 基于z-score的相对阈值匹配
- 加权平均计算各情感得分
- 返回primary + secondary + confidence

### Task 2.4: 紧急程度评估
**优先级**: P0 | **预计工时**: 1.5h

**文件**: `src/speechpulse/urgency.py`

**评分因子**:
- Speaking rate (语速)
- Volume level (音量)
- Pitch variation (音调变化)
- Pause pattern (停顿模式)
- Energy trend (能量趋势)

**算法**:
- 各因子加权求和
- sigmoid归一化到0-1
- 返回level (low/medium/high/critical) 和 reasoning列表

### Task 2.5: 讽刺检测
**优先级**: P1 | **预计工时**: 1.5h

**文件**: `src/speechpulse/sarcasm.py`

**Lite Tier逻辑**:
- 依赖用户传入的`text`参数（Lite无ASR能力）
- 文本情感极性（简单关键词匹配）
- 语音情感（来自emotion模块）
- 对比：文本positive + 语音negative → 讽刺可能
- 语调平坦 + 文本夸张 → 讽刺可能

**输出**: SarcasmResult

### Task 2.6: 完整分析管道
**优先级**: P1 | **预计工时**: 1h

**文件**: `src/speechpulse/analyzer.py`

**类**: `SpeechAnalyzer`

**方法**:
- `analyze(audio_path: str, text: str = None) -> Dict` - 情感分析
- `assess_urgency(audio_path: str) -> UrgencyResult`
- `detect_sarcasm(audio_path: str, text: str = None) -> SarcasmResult`
- `full_analysis(audio_path: str, text: str = None) -> Dict` - 完整分析

---

## Phase 3: MCP Server实现

### Task 3.1: MCP Server基础框架
**优先级**: P0 | **预计工时**: 1.5h

**文件**: `src/speechpulse/server.py`

**依赖**: `pip install mcp`

**实现**:
- 使用FastMCP创建MCP Server
- 注册4个tools
- 支持stdio和sse传输

**约束**:
- 不使用Flask/FastAPI/http.server
- 使用标准MCP协议

### Task 3.2: MCP Tools注册
**优先级**: P0 | **预计工时**: 1.5h

**文件**: `src/speechpulse/server.py`

**Tools**:
1. `analyze_audio` - 转写+情感分析
2. `assess_urgency` - 紧急程度评估
3. `detect_sarcasm` - 讽刺检测
4. `full_analysis` - 完整分析

**参数验证**:
- audio_path: 必须存在，格式支持
- tier: lite/standard/pro，默认lite
- text: Lite tier下detect_sarcasm必需

### Task 3.3: 健康检查端点
**优先级**: P1 | **预计工时**: 1h

**文件**: `src/speechpulse/server.py`

**实现**:
- 在MCP Server基础上添加HTTP路由
- GET /health 返回状态
- 用于Docker/K8s健康检查

### Task 3.4: Server启动脚本
**优先级**: P1 | **预计工时**: 1h

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

## Phase 4: 高级Tier实现（暂不实现，留空或stub）

### Task 4.1: Standard Tier集成 - STUB ONLY
**优先级**: P2 | **预计工时**: 30min

**文件**: `src/speechpulse/tiers/standard.py`
- 创建空文件或stub类
- 依赖在extras中定义

### Task 4.2: Pro Tier集成 - STUB ONLY
**优先级**: P2 | **预计工时**: 30min

**文件**: `src/speechpulse/tiers/pro.py`
- 创建空文件或stub类
- 依赖在extras中定义

### Task 4.3: Tier路由与自动选择 - STUB ONLY
**优先级**: P2 | **预计工时**: 30min

**文件**: `src/speechpulse/tiers/__init__.py`
- 基础路由框架
- 依赖检测逻辑

---

## Phase 5: 集成与测试

### Task 5.1: 单元测试
**优先级**: P1 | **预计工时**: 1.5h

**文件**: `tests/test_all.py`

**测试覆盖**:
- AudioFeatures提取
- 情感规则引擎（z-score相对阈值）
- 紧急程度评分
- 讽刺检测（含text参数处理）
- MCP协议解析
- 配置加载

### Task 5.2: 演示脚本
**优先级**: P1 | **预计工时**: 1h

**文件**: `examples/demo.py`

**功能**:
- 生成合成测试音频（不同情感）
- 调用MCP Server进行分析
- 打印格式化结果

### Task 5.3: OpenClaw Skill定义
**优先级**: P1 | **预计工时**: 1h

**文件**: `skills/SKILL.md`

**内容**:
- Skill名称和描述
- 工具定义（4个tools）
- 参数schema
- 使用示例
- 配置说明

### Task 5.4: 集成测试
**优先级**: P1 | **预计工时**: 30min

**测试流程**:
1. 启动Server
2. 发送analyze_audio请求
3. 验证响应格式
4. 测试错误处理

---

## Phase 6: 文档与发布

### Task 6.1: README编写
**优先级**: P0 | **预计工时**: 2h

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

### Task 6.2: 代码注释与文档字符串
**优先级**: P1 | **预计工时**: 1h

**要求**:
- 所有公共函数添加docstring
- 复杂算法添加行内注释
- 注释使用英文

### Task 6.3: 发布准备
**优先级**: P2 | **预计工时**: 1h

**检查清单**:
- [ ] 版本号确认
- [ ] pyproject.toml完整
- [ ] LICENSE文件
- [ ] README渲染测试
- [ ] 构建测试：`python -m build`
- [ ] 安装测试：`pip install -e .`

---

## 里程碑

| 里程碑 | 包含任务 | 交付物 |
|--------|----------|--------|
| M1: Lite MVP | 1.1-1.5, 2.1-2.6, 3.1-3.4 | 可运行的Lite Tier + MCP Server |
| M2: Full Tiers | 4.1-4.3 | 完整3-tier支持（stub） |
| M3: Release Ready | 5.1-5.4, 6.1-6.3 | 测试通过 + 文档完整 |

---

## 当前状态

- [ ] Phase 1: 基础设施 - 待开始
- [ ] Phase 2: Lite Tier核心 - 待开始
- [ ] Phase 3: MCP Server实现 - 待开始
- [ ] Phase 4: 高级Tier实现 - 待开始（stub only）
- [ ] Phase 5: 集成与测试 - 待开始
- [ ] Phase 6: 文档与发布 - 待开始

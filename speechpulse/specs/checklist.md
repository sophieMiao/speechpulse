# SpeechPulse - Acceptance Checklist

## 验收总览

| 类别 | 检查项 | 状态 |
|------|--------|------|
| 项目结构 | 5项 | ⬜ |
| Lite Tier | 8项 | ⬜ |
| MCP Server | 6项 | ⬜ |
| Standard/Pro Tier | 4项 | ⬜ |
| 测试 | 5项 | ⬜ |
| 文档 | 5项 | ⬜ |
| **总计** | **33项** | **0/33** |

---

## 1. 项目结构验收

### 1.1 目录结构
- [ ] 目录结构符合spec.md定义（src-layout）
- [ ] `src/speechpulse/` 包目录存在
- [ ] `src/speechpulse/types.py` 存在
- [ ] `skills/` 目录存在
- [ ] `examples/` 目录存在
- [ ] `tests/` 目录存在
- [ ] `specs/` 目录存在

### 1.2 配置文件
- [ ] `pyproject.toml` 存在且格式正确
- [ ] 包名：`speechpulse`
- [ ] 版本：`0.1.0`
- [ ] 核心依赖仅含 `mcp>=1.0`（零ML依赖）
- [ ] extras定义完整：standard, pro
- [ ] `.gitignore` 包含Python标准忽略项
- [ ] `LICENSE` (MIT) 存在

### 1.3 包初始化
- [ ] `src/speechpulse/__init__.py` 存在
- [ ] 正确导出主要类
- [ ] `__version__` 定义

---

## 2. Lite Tier 验收

### 2.1 类型定义
- [ ] `AudioFeatures` dataclass 定义完整
- [ ] `EmotionResult` dataclass 定义完整
- [ ] `UrgencyResult` dataclass 定义完整
- [ ] `SarcasmResult` dataclass 定义完整
- [ ] 所有字段类型注解正确

### 2.2 音频处理（stdlib）
- [ ] 纯stdlib WAV加载实现
- [ ] 支持16-bit PCM
- [ ] 支持24-bit PCM
- [ ] 转换为float数组（-1.0 ~ 1.0）
- [ ] 不使用numpy/scipy/librosa

### 2.3 特征提取
- [ ] 时长计算正确
- [ ] 能量均值/标准差计算
- [ ] 过零率计算
- [ ] 基频估计（自相关法）
- [ ] 静音比例计算
- [ ] 输出AudioFeatures对象

### 2.4 情感分析
- [ ] 规则引擎实现
- [ ] 支持情感：happy, sad, angry, anxious, neutral, excited, tired
- [ ] 使用相对阈值（z-score）而非绝对值
- [ ] 规则匹配逻辑正确
- [ ] 置信度计算合理
- [ ] 返回primary + secondary emotion

### 2.5 紧急程度评估
- [ ] 语速因子计算
- [ ] 音量因子计算
- [ ] 音调变化因子计算
- [ ] 停顿模式因子计算
- [ ] 最终分数归一化到0-1
- [ ] 返回level (low/medium/high/critical)
- [ ] 返回reasoning列表

### 2.6 讽刺检测
- [ ] Lite tier正确处理text参数（必需）
- [ ] Standard/Pro tier可自动转写
- [ ] 文本情感极性分析（关键词匹配）
- [ ] 语音情感获取
- [ ] 对比逻辑实现
- [ ] 返回is_sarcastic布尔值
- [ ] 返回confidence分数
- [ ] 返回indicators列表

### 2.7 完整分析管道
- [ ] `SpeechAnalyzer` 类实现
- [ ] `analyze()` 方法
- [ ] `assess_urgency()` 方法
- [ ] `detect_sarcasm()` 方法
- [ ] `full_analysis()` 方法

### 2.8 配置管理
- [ ] 环境变量读取
- [ ] 配置文件读取（JSON）
- [ ] 默认值设置
- [ ] 配置验证

---

## 3. MCP Server 验收

### 3.1 MCP Server基础
- [ ] 使用`mcp` Python SDK实现
- [ ] 使用FastMCP创建Server
- [ ] 不使用Flask/FastAPI/http.server
- [ ] 支持stdio传输模式
- [ ] 支持sse传输模式
- [ ] 支持GET /health（额外监控端点）

### 3.2 MCP协议实现
- [ ] `tools/list` 端点正确返回工具列表
- [ ] `tools/call` 端点正确调用工具
- [ ] 请求解析正确
- [ ] 响应格式符合MCP协议
- [ ] 错误处理完善

### 3.3 MCP Tools
- [ ] `analyze_audio` tool实现
  - [ ] Lite tier返回transcription=null
  - [ ] Standard/Pro tier返回转写文本
  - [ ] 支持text参数（用户传入转写）
- [ ] `assess_urgency` tool实现
- [ ] `detect_sarcasm` tool实现
  - [ ] Lite tier必需text参数
  - [ ] Standard/Pro tier可自动转写
- [ ] `full_analysis` tool实现
- [ ] 参数验证完整
- [ ] 返回格式符合spec.md定义

### 3.4 启动脚本
- [ ] `__main__.py` 实现
- [ ] 支持`python -m speechpulse`启动
- [ ] --transport 参数支持（stdio/sse）
- [ ] --port 参数支持（sse模式）
- [ ] --tier 参数支持
- [ ] --config 参数支持

---

## 4. Standard/Pro Tier 验收

### 4.1 Standard Tier
- [ ] emotion2vec集成
- [ ] Whisper集成
- [ ] 依赖在extras中定义
- [ ] 优雅降级（依赖缺失时提示）

### 4.2 Pro Tier
- [ ] Qwen2-Audio集成
- [ ] 支持本地4-bit量化
- [ ] 支持DashScope API
- [ ] 依赖在extras中定义

### 4.3 Tier路由
- [ ] 自动检测依赖
- [ ] 根据tier参数路由
- [ ] 依赖缺失时友好错误提示

---

## 5. 测试验收

### 5.1 单元测试
- [ ] `tests/test_all.py` 存在
- [ ] 音频特征提取测试
- [ ] 情感规则引擎测试
- [ ] 紧急程度评分测试
- [ ] 讽刺检测测试
- [ ] MCP工具注册与调用测试
- [ ] 配置加载测试

### 5.2 集成测试
- [ ] Server启动测试
- [ ] API端点测试
- [ ] 错误处理测试
- [ ] 完整流程测试

### 5.3 演示脚本
- [ ] `examples/demo.py` 存在
- [ ] 可生成合成测试音频
- [ ] 可调用MCP Server
- [ ] 输出格式化结果

### 5.4 OpenClaw Skill
- [ ] `skills/SKILL.md` 存在
- [ ] Skill定义完整
- [ ] 4个tools定义
- [ ] 参数schema正确
- [ ] 使用示例

---

## 6. 文档验收

### 6.1 README
- [ ] `README.md` 存在
- [ ] 项目介绍（What & Why）
- [ ] 架构图（ASCII）
- [ ] 快速开始指南
- [ ] Demo输出示例
- [ ] Tier对比表
- [ ] 安装说明
- [ ] 中英双语

### 6.2 代码注释
- [ ] 所有公共函数有docstring
- [ ] 注释使用英文
- [ ] 复杂算法有行内注释
- [ ] 类型注解完整

### 6.3 规格文档
- [ ] `specs/spec.md` 完整
- [ ] `specs/tasks.md` 完整
- [ ] `specs/checklist.md` 完整

---

## 7. 性能验收

### 7.1 Lite Tier性能
- [ ] 冷启动 < 1s
- [ ] 单次请求 < 100ms
- [ ] 内存占用 < 50MB

### 7.2 Standard Tier性能
- [ ] 冷启动 < 5s
- [ ] 单次请求 1-3s
- [ ] 内存占用 < 3GB

### 7.3 Pro Tier性能
- [ ] 冷启动 < 30s
- [ ] 单次请求 3-10s
- [ ] 内存占用 < 8GB

---

## 8. 安全与隐私验收

### 8.1 隐私保护
- [ ] 本地处理，不上传音频
- [ ] 不存储音频文件
- [ ] 不存储分析结果

### 8.2 输入验证
- [ ] 路径安全检查（防目录遍历）
- [ ] 音频格式白名单
- [ ] 文件大小限制

---

## 验收流程

### 阶段1: Lite MVP验收
```bash
# 1. 安装
pip install -e .

# 2. 运行演示脚本（直接调用分析模块，无需启动Server）
python examples/demo.py

# 3. 测试MCP Server SSE模式
python -m speechpulse --transport sse --port 8080 &
sleep 2
curl http://localhost:8080/health
kill %1

# 4. 测试MCP Server stdio模式（用于MCP客户端集成）
echo '{"jsonrpc":"2.0","method":"tools/list","id":1}' | python -m speechpulse

# 5. 运行单元测试
pytest tests/test_all.py -v
```

### 阶段2: 完整功能验收
```bash
# 1. 安装Standard依赖
pip install -e ".[standard]"

# 2. 测试Standard Tier
python -m speechpulse --tier standard

# 3. 安装Pro依赖
pip install -e ".[pro]"

# 4. 测试Pro Tier
python -m speechpulse --tier pro
```

### 阶段3: 文档验收
- [ ] README渲染正常
- [ ] 所有链接可点击
- [ ] 代码示例可运行

---

## 验收签字

| 角色 | 签字 | 日期 |
|------|------|------|
| 技术负责人 | ⬜ | |
| 产品经理 | ⬜ | |
| 测试负责人 | ⬜ | |

---

## 附录: 快速验证命令

```bash
# 结构检查
ls -la speechpulse/
ls -la speechpulse/src/speechpulse/

# 依赖检查
cat speechpulse/pyproject.toml | grep -A 20 "dependencies"

# 导入测试
python -c "from speechpulse import SpeechAnalyzer; print('OK')"

# Server启动测试（stdio模式，用于MCP客户端）
python -m speechpulse --tier lite

# Server启动测试（sse模式，用于HTTP访问）
python -m speechpulse --transport sse --port 8080 &
sleep 2
curl http://localhost:8080/health

# Lite功能测试
python speechpulse/examples/demo.py

# 测试运行
pytest speechpulse/tests/test_all.py -v
```

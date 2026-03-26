# 时序预测脚本使用方式

## 核心功能

1. **0点预测** (`predict_at_midnight`): 根据用户预设的日期类型预测当天0-24时的value
2. **9点预测** (`predict_at_nine`): 根据0-9点实际数据自动推断日期类型，预测当天9-24时的value

## 预测逻辑

```
0点预测流程:
  输入: 历史数据 + calendar_day_type_input(预设日期类型)
  ↓
  使用预设日期类型
  ↓
  输出: 当天0-24时预测

9点预测流程:
  输入: 历史数据 + 当日0-9点数据
  ↓
  分析0-9点数据特征，自动推断日期类型
  ↓
  (覆盖0点的预设类型)
  ↓
  输出: 推断的日期类型 + 9-24时预测
```

## 支持的功能

- 支持2年以上历史数据（70,000+数据点）
- 支持15分钟和60分钟两种时间间隔
- 自动截取最近数据用于分析（控制内存和运行时间）
- Prophet模型支持day_type作为回归特征
- 典型运行时间：< 1分钟（模板预测）或 < 5分钟（Prophet预测）

## 快速开始

### 1. 安装依赖
```bash
pip install pandas numpy
# 可选：安装Prophet以获得更好的预测效果
pip install prophet
```

### 2. 准备数据

数据需要包含3列：`time`, `value`, `day_type`

| 列名 | 类型 | 说明 |
|------|------|------|
| `time` | datetime | 时间戳 |
| `value` | float | 数值 |
| `day_type` | int | 1=工作日, 0=休息日, 其他=异常 |

### 3. 使用示例

#### 生成测试数据
```python
from time_series_forecast import generate_sample_data

# 生成15分钟间隔测试数据
df = generate_sample_data(days=100, interval_minutes=15)

# 生成60分钟间隔测试数据
df = generate_sample_data(days=100, interval_minutes=60)
```

#### 0点预测
```python
import pandas as pd
from time_series_forecast import predict_at_midnight

# 加载数据
df = pd.read_csv('your_data.csv')

# 预测 - 输入1表示工作日
result = predict_at_midnight(df, interval_minutes=15, calendar_day_type_input=1)

# 预测 - 输入0表示休息日
result = predict_at_midnight(df, interval_minutes=15, calendar_day_type_input=0)

print(result)
```

#### 9点预测
```python
import pandas as pd
from time_series_forecast import predict_at_nine

# 加载数据
df = pd.read_csv('your_data.csv')

# 获取当日0-9点数据（36个点 for 15分钟间隔）
# 假设当天数据在 df_today 中
nine_am_data = df_today[df_today['time'].dt.hour < 9]

# 预测 - 自动推断日期类型
day_type, forecast = predict_at_nine(df, nine_am_data, interval_minutes=15)

print(f"推断的日期类型: {day_type} (1=工作日, 0=休息日, 2=异常)")
print(forecast)
```

#### 评估日期类型
```python
from time_series_forecast import evaluate_day_type_by_comparison

# 对比预测和实际数据，评估日期类型
actual = pd.read_csv('actual_data.csv')
day_type = evaluate_day_type_by_comparison(forecast_df, actual)

print(f"评估结果: {day_type}")  # 'weekday', 'holiday', 或 'anomaly'
```

## 输入数据格式

| 列名 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `time` | datetime | 是 | 时间戳 |
| `value` | float | 是 | 数值 |
| `day_type` | int | 是 | 1=工作日, 0=休息日, 其他=异常 |

## 输出数据格式

### predict_at_midnight 返回

| 列名 | 说明 |
|------|------|
| `time` | 预测时间点（0-24时） |
| `value_predicted` | 预测值 |
| `day_type` | 输入的日期类型 |
| `value_lower` | 预测区间下限（85%） |
| `value_upper` | 预测区间上限（115%） |

### predict_at_nine 返回

返回元组: `(day_type, forecast_df)`

- `day_type`: 推断的日期类型 (1=工作日, 0=休息日, 2=异常)
- `forecast_df`: DataFrame，列同上

## 高级参数

### 时间间隔
```python
# 15分钟间隔
result = predict_at_midnight(df, interval_minutes=15, calendar_day_type_input=1)

# 60分钟间隔
result = predict_at_midnight(df, interval_minutes=60, calendar_day_type_input=1)
```

### 日期类型说明

| calendar_day_type_input | 含义 |
|------------------------|------|
| 1 | 工作日 (weekday) |
| 0 | 休息日 (holiday) |
| 其他数字 | 异常日期 (anomaly) |

## 运行测试

```bash
python time_series_forecast.py
```

## 性能说明

| 数据量 | 间隔 | 典型运行时间 |
|--------|------|--------------|
| 2年+ (70,000+) | 15分钟 | < 1分钟（模板）/ < 5分钟（Prophet） |
| 2年+ (70,000+) | 60分钟 | < 30秒（模板）/ < 3分钟（Prophet） |

## 特性总结

- **双阶段预测**: 0点用预设类型，9点自动推断覆盖
- **day_type特征**: 历史数据包含日期类型，Prophet模型使用此特征
- **自适应间隔**: 支持15分钟和60分钟两种间隔
- **大数据支持**: 自动处理2年+历史数据
- **纯CPU运行**: 无GPU依赖
- **30分钟保障**: 即使使用Prophet，默认参数确保运行时间<5分钟
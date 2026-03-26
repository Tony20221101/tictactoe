"""
时序数据预测脚本 - 支持2年以上历史数据
输入：包含 time, value, day_type 三列的 DataFrame（支持15分钟或60分钟间隔）
      day_type: 1=工作日, 0=休息日, 其他=异常
输出：当天剩余时间的value预测

预测逻辑：
1. 0点预测：根据用户输入的预设日期类型进行预测
2. 9点预测：根据0-9时数据自动推断日期类型，覆盖预设类型
3. 使用动态模板匹配进行预测

性能优化：
- 支持大数据量（2年+），自动采样最近90天数据用于分析
- 支持多种时间间隔（15分钟/60分钟），自动调整样本数量
- 高效的日期类型判断算法
- 使用动态生成的模板，准确匹配数据特征

环境要求：无GPU，运行时间控制在30分钟内
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')



def get_interval_config(interval_minutes=15):
    """
    获取时间间隔配置参数

    参数:
        interval_minutes: 时间间隔分钟数（15或60）

    返回:
        配置参数字典，包含点数计算、频率字符串等
    """
    if interval_minutes == 15:
        return {
            'interval_minutes': 15,
            'intervals_per_day': 96,        # 24小时 * 4
            'intervals_per_hour': 4,
            'hours_for_detection': 9,
            'points_for_detection': 36,       # 9小时 * 4
            'freq': '15min',
            'min_daily_points': 48,           # 至少半天数据
            'stats_window_days': 90,          # 统计窗口：90天
            'train_window_days': 30           # 训练窗口：30天
        }
    elif interval_minutes == 60:
        return {
            'interval_minutes': 60,
            'intervals_per_day': 24,          # 24小时 * 1
            'intervals_per_hour': 1,
            'hours_for_detection': 9,
            'points_for_detection': 9,          # 9小时 * 1
            'freq': '60min',
            'min_daily_points': 12,            # 至少半天数据
            'stats_window_days': 180,          # 统计窗口：180天（样本较少，需要更长的窗口）
            'train_window_days': 60            # 训练窗口：60天
        }
    else:
        raise ValueError(f"不支持的时间间隔: {interval_minutes}分钟，仅支持15或60分钟")


def detect_interval(df):
    """
    自动检测数据的时间间隔

    参数:
        df: DataFrame，包含 'time' 列

    返回:
        时间间隔分钟数（15或60）
    """
    time_diff = df['time'].diff().dropna().median()
    interval_minutes = int(time_diff.total_seconds() / 60)

    # 处理接近15或60分钟的值
    if 10 <= interval_minutes <= 20:
        return 15
    elif 50 <= interval_minutes <= 70:
        return 60
    else:
        # 如果不符合标准间隔，返回最接近的标准值
        if interval_minutes < 35:
            return 15
        else:
            return 60


def generate_pattern_from_history(df, interval_minutes=15):
    """
    根据历史数据动态生成工作日和假期的模式模板

    参数:
        df: DataFrame，包含 'time', 'value', 'day_type' 列
        interval_minutes: 时间间隔（15或60）

    返回:
        dict: {'weekday': array, 'holiday': array}
            - weekday: 工作日平均模式数组 (96点 for 15min, 24点 for 60min)
            - holiday: 假期平均模式数组
    """
    df = df.copy()
    df['time'] = pd.to_datetime(df['time'])

    # 提取日期
    df['date'] = df['time'].dt.date

    # 根据interval计算每天的总点数
    points_per_day = int(24 * 60 / interval_minutes)

    # 按日期分组，计算每天每个时刻的平均值
    df['time_of_day'] = df['time'].dt.hour * 60 + df['time'].dt.minute

    # 将time_of_day转换为点序号 (0 到 points_per_day-1)
    df['point_index'] = (df['time_of_day'] / interval_minutes).astype(int) % points_per_day

    # 分离工作日和假期数据
    weekday_data = df[df['day_type'] == 1]
    holiday_data = df[df['day_type'] == 0]

    # 计算每个时间点的平均值
    weekday_pattern = weekday_data.groupby('point_index')['value'].mean().sort_index()
    holiday_pattern = holiday_data.groupby('point_index')['value'].mean().sort_index()

    # 填充缺失的时间点（如果有）
    all_indices = np.arange(points_per_day)
    weekday_pattern = weekday_pattern.reindex(all_indices)
    holiday_pattern = holiday_pattern.reindex(all_indices)

    # 填充NaN值（使用整体均值）
    weekday_pattern = weekday_pattern.fillna(weekday_pattern.mean())
    holiday_pattern = holiday_pattern.fillna(holiday_pattern.mean())

    weekday_pattern = weekday_pattern.values
    holiday_pattern = holiday_pattern.values

    print(f"\n从历史数据生成模板（间隔{interval_minutes}分钟，{points_per_day}点/天）:")
    print(f"  工作日模板: {len(weekday_pattern)}点, 均值={np.mean(weekday_pattern):.2f}")
    print(f"  假期模板: {len(holiday_pattern)}点, 均值={np.mean(holiday_pattern):.2f}")

    return {'weekday': weekday_pattern, 'holiday': holiday_pattern}


def load_and_preprocess_data(df, max_history_days=90, interval_minutes=None):
    """
    加载并预处理数据 - 优化大数据量处理，支持多种时间间隔

    参数:
        df: DataFrame，包含 'time', 'value', 'day_type' 列
            - time: 时间戳
            - value: 数值
            - day_type: 日期类型 (1=工作日, 0=休息日, 其他=异常)
        max_history_days: 最大历史天数（根据时间间隔自动调整）
        interval_minutes: 时间间隔分钟数（15或60），None表示自动检测

    返回:
        预处理后的DataFrame（仅包含最近max_history_days天的数据）
    """
    df = df.copy()

    # 确保time列是datetime类型
    df['time'] = pd.to_datetime(df['time'])

    # 按时间排序
    df = df.sort_values('time').reset_index(drop=True)

    original_count = len(df)
    original_start = df['time'].min()
    original_end = df['time'].max()

    # 检测或确认时间间隔
    if interval_minutes is None:
        interval_minutes = detect_interval(df)
        print(f"自动检测到时间间隔: {interval_minutes}分钟")

    # 获取配置
    config = get_interval_config(interval_minutes)

    # 对于大数据集，只保留最近max_history_days天的数据
    cutoff_date = df['time'].max() - timedelta(days=max_history_days)
    df = df[df['time'] >= cutoff_date].copy()

    # 计算实际时间间隔
    time_diff = df['time'].diff().dropna().median()

    print(f"时间间隔: {interval_minutes}分钟 ({config['intervals_per_day']}点/天)")
    print(f"原始数据点数: {original_count} ({original_start.date()} 至 {original_end.date()})")
    print(f"处理后数据点数: {len(df)} ({df['time'].min().date()} 至 {df['time'].max().date()})")
    print(f"每天数据点数: {config['intervals_per_day']}, 检测时长: {config['hours_for_detection']}小时 ({config['points_for_detection']}点)")

    if original_count > len(df):
        print(f"已自动截取最近 {max_history_days} 天数据用于分析")

    # 处理缺失值
    if df['value'].isnull().sum() > 0:
        print(f"检测到 {df['value'].isnull().sum()} 个缺失值，使用线性插值填充")
        df['value'] = df['value'].interpolate(method='linear')

    # 处理day_type列（如果存在）
    if 'day_type' in df.columns:
        # 处理day_type缺失值：使用前向填充
        if df['day_type'].isnull().sum() > 0:
            null_count = df['day_type'].isnull().sum()
            print(f"检测到 {null_count} 个day_type缺失值，使用前向填充")
            df['day_type'] = df['day_type'].ffill().bfill()
        # 确保day_type为数值类型
        df['day_type'] = pd.to_numeric(df['day_type'], errors='coerce').fillna(0).astype(int)

    return df


def get_first_9_hours_data(df, target_date=None, interval_minutes=15):
    """
    获取指定日期的前9小时数据

    参数:
        df: 完整的DataFrame
        target_date: 目标日期（默认使用最新日期）
        interval_minutes: 时间间隔分钟数（15或60）

    返回:
        前9小时数据的DataFrame, 当天完整数据
    """
    # 获取配置
    config = get_interval_config(interval_minutes)
    points_needed = config['points_for_detection']

    if target_date is None:
        target_date = df['time'].dt.date.iloc[-1]

    # 将目标日期转为datetime
    if isinstance(target_date, str):
        target_date = pd.to_datetime(target_date).date()

    # 筛选目标日期的数据
    day_data = df[df['time'].dt.date == target_date].copy()

    if len(day_data) == 0:
        raise ValueError(f"日期 {target_date} 没有数据")

    # 获取当天的前9小时数据（根据时间间隔计算点数）
    first_9_hours = day_data.head(points_needed)

    print(f"\n目标日期: {target_date}")
    print(f"时间间隔: {interval_minutes}分钟")
    print(f"当天数据点数: {len(day_data)} (预期{config['intervals_per_day']}点)")
    print(f"前{config['hours_for_detection']}小时数据点数: {len(first_9_hours)} (预期{points_needed}点)")
    if len(first_9_hours) > 0:
        print(f"前{config['hours_for_detection']}小时时间范围: {first_9_hours['time'].min()} 至 {first_9_hours['time'].max()}")

    return first_9_hours, day_data


def calculate_day_type_stats(df, anomaly_dates=None):
    """
    根据历史数据计算工作日和假期的统计特征
    用于辅助日期类型判断

    参数:
        df: DataFrame，包含 'time' 和 'value' 列
        anomaly_dates: 异常日期集合，这些日期会被排除在统计之外

    返回:
        包含工作日和假期统计特征的字典，同时包含样本数量信息
    """
    # 添加日期类型标记
    df = df.copy()
    df['date'] = df['time'].dt.date
    df['hour'] = df['time'].dt.hour
    df['is_weekend'] = df['time'].dt.weekday >= 5

    # 过滤掉异常日期
    if anomaly_dates:
        original_dates = df['date'].nunique()
        df = df[~df['date'].isin(anomaly_dates)].copy()
        remaining_dates = df['date'].nunique()
        print(f"  统计计算已排除 {len(anomaly_dates)} 个异常日期，剩余 {remaining_dates} 个有效日期")

    # 按日期聚合
    daily_stats = df.groupby('date').agg({
        'value': ['mean', 'std', 'min', 'max'],
        'is_weekend': 'first'
    }).reset_index()

    daily_stats.columns = ['date', 'mean', 'std', 'min', 'max', 'is_weekend']

    # 分离工作日和假期统计
    weekday_stats = daily_stats[daily_stats['is_weekend'] == False]
    holiday_stats = daily_stats[daily_stats['is_weekend'] == True]

    stats = {}
    if len(weekday_stats) > 0:
        stats['weekday'] = {
            'mean': weekday_stats['mean'].mean(),
            'std': weekday_stats['std'].mean(),
            'range': (weekday_stats['max'] - weekday_stats['min']).mean(),
            'count': len(weekday_stats)  # 添加样本数量
        }
    if len(holiday_stats) > 0:
        stats['holiday'] = {
            'mean': holiday_stats['mean'].mean(),
            'std': holiday_stats['std'].mean(),
            'range': (holiday_stats['max'] - holiday_stats['min']).mean(),
            'count': len(holiday_stats)  # 添加样本数量
        }

    # 输出数据分布情况
    print(f"  工作日样本数: {len(weekday_stats)}, 假期样本数: {len(holiday_stats)}")
    if len(weekday_stats) + len(holiday_stats) > 0:
        holiday_ratio = len(holiday_stats) / (len(weekday_stats) + len(holiday_stats))
        print(f"  假期占比: {holiday_ratio:.1%}")

    return stats


def detect_anomaly_days(df, z_threshold=3.0, iqr_multiplier=2.0, dynamic_patterns=None):
    """
    检测历史数据中的异常日期

    检测规则:
    1. Z-score 方法：单日均值偏离整体均值超过 z_threshold 个标准差
    2. IQR 方法：单日统计量超过四分位数的异常范围

    参数:
        df: DataFrame，包含 'time' 和 'value' 列
        z_threshold: Z-score 阈值（默认3.0，即99.7%置信区间外的为异常）
        iqr_multiplier: IQR 倍数（默认2.0）
        dynamic_patterns: 动态生成的模板 dict{'weekday': array, 'holiday': array}

    返回:
        set: 异常日期的集合 (date 对象)
    """
    df = df.copy()
    df['date'] = df['time'].dt.date

    # 按日期聚合统计
    daily_stats = df.groupby('date').agg({
        'value': ['mean', 'std', 'min', 'max', 'count']
    }).reset_index()
    daily_stats.columns = ['date', 'mean', 'std', 'min', 'max', 'count']

    # 过滤掉数据点过少的天数（至少需要有48个点，即12小时数据）
    daily_stats = daily_stats[daily_stats['count'] >= 48]

    if len(daily_stats) < 7:
        print("历史数据不足，跳过异常日期检测")
        return set()

    anomaly_dates = set()

    # 方法1: Z-score 检测
    for col in ['mean', 'std', 'max']:
        col_mean = daily_stats[col].mean()
        col_std = daily_stats[col].std()
        if col_std > 0:
            z_scores = np.abs((daily_stats[col] - col_mean) / col_std)
            anomaly_mask = z_scores > z_threshold
            anomalous = daily_stats[anomaly_mask]['date'].tolist()
            anomaly_dates.update(anomalous)

    # 方法2: IQR 检测
    for col in ['mean', 'std', 'max']:
        q1 = daily_stats[col].quantile(0.25)
        q3 = daily_stats[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - iqr_multiplier * iqr
        upper_bound = q3 + iqr_multiplier * iqr
        anomaly_mask = (daily_stats[col] < lower_bound) | (daily_stats[col] > upper_bound)
        anomalous = daily_stats[anomaly_mask]['date'].tolist()
        anomaly_dates.update(anomalous)

    # 方法3: 基于工作日/假期模板的相似度检测
    daily_stats['is_weekend'] = pd.to_datetime(daily_stats['date']).dt.weekday >= 5

    for _, day_row in daily_stats.iterrows():
        date = day_row['date']
        day_data = df[df['date'] == date]['value'].values

        if len(day_data) >= 36:  # 至少需要9小时数据
            # 必须使用动态模板
            if dynamic_patterns is None or not isinstance(dynamic_patterns, dict):
                print("警告: detect_anomaly_days 需要 dynamic_patterns，跳过模板相似度检测")
                break

            if day_row['is_weekend']:
                template = dynamic_patterns.get('holiday').copy()
            else:
                template = dynamic_patterns.get('weekday').copy()

            # 计算与模板的相似度（使用前9小时或更少）
            observed = day_data[:min(36, len(day_data))]
            template_segment = template[:len(observed)]

            # 标准化后计算差异
            obs_norm = (observed - np.mean(observed)) / (np.std(observed) + 1e-8)
            temp_norm = (template_segment - np.mean(template_segment)) / (np.std(template_segment) + 1e-8)

            # 计算欧氏距离
            distance = np.sqrt(np.sum((obs_norm - temp_norm) ** 2))

            # 距离过大认为是异常
            threshold_distance = 8.0  # 经验阈值
            if distance > threshold_distance:
                anomaly_dates.add(date)

    print(f"\n异常日期检测完成:")
    print(f"  共检测到 {len(anomaly_dates)} 个异常日期")
    if len(anomaly_dates) > 0:
        sorted_anomalies = sorted(list(anomaly_dates))[:5]  # 只显示前5个
        print(f"  前5个异常日期: {[str(d) for d in sorted_anomalies]}")
        if len(anomaly_dates) > 5:
            print(f"  ... 等共 {len(anomaly_dates)} 个")

    return anomaly_dates


def detect_day_type(first_9_hours_data, historical_stats=None, is_anomaly=False, mode='predict', dynamic_patterns=None):
    """
    根据前9小时数据特征判断日期类型

    判断依据：
    1. 异常检测：首先检查是否为异常日期（仅在evaluate模式下）
    2. 平均值：工作日通常有更高的平均活动量
    3. 方差：工作日通常有更大的波动（早晚高峰）
    4. 趋势：工作日早上通常有上升趋势
    5. 历史统计对比（如果有）

    参数:
        first_9_hours_data: 前9小时的数据
        historical_stats: 历史统计数据
        is_anomaly: 是否已标记为异常日期
        mode: 'predict' - 预测模式（只返回weekday/holiday）
              'evaluate' - 评估模式（返回weekday/holiday/anomaly）
        dynamic_patterns: 动态生成的模板 dict{'weekday': array, 'holiday': array}，优先于默认模板

    返回:
        'weekday', 'holiday' 或 'anomaly'（仅evaluate模式）
    """
    values = first_9_hours_data['value'].values

    # 计算统计特征
    mean_value = np.mean(values)
    std_value = np.std(values)
    min_value = np.min(values)
    max_value = np.max(values)
    range_value = max_value - min_value

    # 计算趋势（前一半vs后一半）
    mid = len(values) // 2
    first_half_mean = np.mean(values[:mid])
    second_half_mean = np.mean(values[mid:])
    trend = second_half_mean - first_half_mean

    print(f"\n前9小时数据特征:")
    print(f"  平均值: {mean_value:.2f}")
    print(f"  标准差: {std_value:.2f}")
    print(f"  最小值: {min_value:.2f}")
    print(f"  最大值: {max_value:.2f}")
    print(f"  极差: {range_value:.2f}")
    print(f"  趋势(后半-前半): {trend:.2f}")

    # 计算与模板的相似度（必须使用动态模板）
    if dynamic_patterns is None or not isinstance(dynamic_patterns, dict):
        raise ValueError("detect_day_type requires dynamic_patterns parameter from generate_pattern_from_history()")

    weekday_template = dynamic_patterns.get('weekday')[:len(values)]
    holiday_template = dynamic_patterns.get('holiday')[:len(values)]

    weekday_similarity = calculate_pattern_similarity(values, weekday_template)
    holiday_similarity = calculate_pattern_similarity(values, holiday_template)

    print(f"  与工作日模板相似度: {weekday_similarity:.4f}")
    print(f"  与假期模板相似度: {holiday_similarity:.4f}")

    # 仅在evaluate模式下进行异常检测
    if mode == 'evaluate':
        # 如果已标记为异常，直接返回
        if is_anomaly:
            print("\n判断结果: ANOMALY (预标记异常日期)")
            return 'anomaly'

        # 实时异常检测：基于当前数据与模板的偏离程度
        max_similarity = max(weekday_similarity, holiday_similarity)

        # 异常判断条件：
        # 1. 与两种模板的相似度都低于阈值
        # 2. 方差异常低（几乎无波动）或异常高（剧烈波动）
        # 3. 值域超出正常范围太多
        is_anomaly_detected = False
        anomaly_reasons = []

        if max_similarity < 0.3:  # 与任何模板都不相似
            is_anomaly_detected = True
            anomaly_reasons.append(f"与模板相似度过低({max_similarity:.3f})")

        if historical_stats:
            # 检查是否偏离历史统计太多
            if 'weekday' in historical_stats:
                wd_mean = historical_stats['weekday']['mean']
                wd_std = historical_stats['weekday']['std']
                if abs(mean_value - wd_mean) > 3 * wd_std:
                    is_anomaly_detected = True
                    anomaly_reasons.append(f"均值偏离工作日统计({abs(mean_value - wd_mean):.1f} > 3σ)")

            if 'holiday' in historical_stats:
                hl_mean = historical_stats['holiday']['mean']
                hl_std = historical_stats['holiday']['std']
                if abs(mean_value - hl_mean) > 3 * hl_std:
                    is_anomaly_detected = True
                    anomaly_reasons.append(f"均值偏离假期统计({abs(mean_value - hl_mean):.1f} > 3σ)")

        # 实时异常：极差异常
        if range_value < 5:  # 几乎无变化
            is_anomaly_detected = True
            anomaly_reasons.append("极差过小(几乎无波动)")
        elif range_value > 300:  # 变化过于剧烈
            is_anomaly_detected = True
            anomaly_reasons.append("极差过大(变化过于剧烈)")

        if is_anomaly_detected:
            print(f"\n判断结果: ANOMALY (检测到异常)")
            print(f"  异常原因: {', '.join(anomaly_reasons)}")
            return 'anomaly'

    # 综合判断规则（原有逻辑）
    weekday_score = 0
    holiday_score = 0

    # 规则1: 平均值判断
    threshold_mean = 80
    if historical_stats and 'weekday' in historical_stats and 'holiday' in historical_stats:
        threshold_mean = (historical_stats['weekday']['mean'] + historical_stats['holiday']['mean']) / 2

    if mean_value > threshold_mean:
        weekday_score += 1
        print(f"  -> 平均值({mean_value:.1f}) > 阈值({threshold_mean:.1f})，倾向工作日")
    else:
        holiday_score += 1
        print(f"  -> 平均值({mean_value:.1f}) <= 阈值({threshold_mean:.1f})，倾向假期")

    # 规则2: 波动性判断
    threshold_std = 30
    if historical_stats and 'weekday' in historical_stats and 'holiday' in historical_stats:
        threshold_std = (historical_stats['weekday']['std'] + historical_stats['holiday']['std']) / 2

    if std_value > threshold_std:
        weekday_score += 1
        print(f"  -> 波动性({std_value:.1f}) > 阈值({threshold_std:.1f})，倾向工作日")
    else:
        holiday_score += 1
        print(f"  -> 波动性({std_value:.1f}) <= 阈值({threshold_std:.1f})，倾向假期")

    # 规则3: 趋势判断（工作日早上通常上升）
    if trend > 5:
        weekday_score += 1
        print("  -> 有明显上升趋势，倾向工作日")
    elif trend < -5:
        holiday_score += 1
        print("  -> 有明显下降趋势，倾向假期")

    # 规则4: 模板相似度
    if weekday_similarity > holiday_similarity:
        weekday_score += 2
        print("  -> 与工作日模板更相似")
    else:
        holiday_score += 2
        print("  -> 与假期模板更相似")

    # 最终判断
    if weekday_score >= holiday_score:
        day_type = 'weekday'
        confidence = weekday_score / (weekday_score + holiday_score) if (weekday_score + holiday_score) > 0 else 0.5
    else:
        day_type = 'holiday'
        confidence = holiday_score / (weekday_score + holiday_score) if (weekday_score + holiday_score) > 0 else 0.5

    print(f"\n判断结果: {day_type.upper()} (置信度: {confidence:.2%})")
    print(f"评分 - 工作日: {weekday_score}, 假期: {holiday_score}")

    return day_type


def calculate_pattern_similarity(observed, template):
    """
    计算观察数据与模板的相似度（余弦相似度）
    """
    if len(observed) != len(template):
        template = template[:len(observed)]

    # 归一化
    obs_norm = (observed - np.mean(observed)) / (np.std(observed) + 1e-8)
    temp_norm = (template - np.mean(template)) / (np.std(template) + 1e-8)

    # 余弦相似度
    similarity = np.dot(obs_norm, temp_norm) / (np.linalg.norm(obs_norm) * np.linalg.norm(temp_norm) + 1e-8)

    return max(0, similarity)  # 确保非负


def predict_remaining_day(df, first_9_hours, day_type, day_data, historical_stats=None, interval_minutes=15, remaining_times=None, dynamic_patterns=None):
    """
    根据日期类型预测当天剩余时间

    参数:
        df: 历史数据
        first_9_hours: 前9小时数据（或根据时间间隔的前N小时数据）
        day_type: 'weekday', 'holiday', 'anomaly' 或对应的数值(0/1/2)
        day_data: 当天的完整数据
        historical_stats: 历史统计特征（用于异常日期预测）
        interval_minutes: 时间间隔分钟数（15或60）
        remaining_times: 预测时间点列表（可选）
        dynamic_patterns: 动态生成的模板 dict{'weekday': array, 'holiday': array}，优先于默认模板

    返回:
        预测结果DataFrame
    """
    # 获取配置
    config = get_interval_config(interval_minutes)

    print(f"\n开始预测当天剩余时间（基于{day_type}模式，{interval_minutes}分钟间隔）...")

    # 处理0点预测情况（first_9_hours为空）
    if len(first_9_hours) == 0:
        # 0点预测：使用历史数据计算最后一个值作为参考
        if len(df) > 0:
            # 获取历史最后一天最后几个点的平均值
            last_day = df['time'].max().date()
            last_day_data = df[df['time'].dt.date == last_day]
            if len(last_day_data) > 0:
                last_value = last_day_data['value'].iloc[-1]
            else:
                last_value = df['value'].iloc[-1]
        else:
            last_value = 100  # 默认值

        # 目标日期 - 使用day_data获取（如果有）
        if len(day_data) > 0:
            target_date = day_data['time'].min().date()
        else:
            # 使用历史数据的最后日期+1天
            target_date = df['time'].max().date() + timedelta(days=1)
        last_time = pd.Timestamp(target_date)

        print("0点预测模式：无当天数据，使用历史数据参考")
    else:
        # 获取前9小时的最后一个时间点
        last_time = first_9_hours['time'].iloc[-1]
        last_value = first_9_hours['value'].iloc[-1]

    # 确定当天剩余时间
    current_date = last_time.date()
    day_end = pd.Timestamp(current_date) + timedelta(days=1)

    # 生成剩余时间的预测时间点（根据时间间隔）
    # 优先使用传入的remaining_times参数
    if remaining_times is None or len(remaining_times) == 0:
        remaining_times = pd.date_range(
            start=last_time + timedelta(minutes=interval_minutes),
            end=day_end - timedelta(minutes=interval_minutes),
            freq=config['freq']
        )

    num_predictions = len(remaining_times)
    print(f"需要预测的时间点数: {num_predictions}")
    print(f"预测时间范围: {remaining_times[0]} 至 {remaining_times[-1]}")

    # 将数值day_type转换为字符串类型
    # 1=工作日(weekday), 0=休息日(holiday), 其他=异常(anomaly)
    if isinstance(day_type, (int, float)):
        if day_type == 1:
            day_type_str = 'weekday'
        elif day_type == 0:
            day_type_str = 'holiday'
        else:
            day_type_str = 'anomaly'
        print(f"日期类型转换: {day_type} -> {day_type_str}")
    else:
        day_type_str = day_type

    # 根据日期类型选择模板
    if day_type_str == 'anomaly':
        # 异常日期：使用更保守的预测策略
        # 基于前N小时的趋势进行简单线性外推
        print("异常日期模式：使用趋势外推预测")

        # 计算前N小时的平均趋势（每点的变化）
        values = first_9_hours['value'].values
        trend_per_point = (values[-1] - values[0]) / len(values) if len(values) > 1 else 0

        # 基于最后一个值和趋势进行预测
        predictions = []
        current_val = last_value
        for i in range(num_predictions):
            current_val += trend_per_point
            # 添加一些衰减，避免趋势无限延续
            if day_type == 'anomaly':
                # 异常日期预测值趋向历史均值
                if 'weekday' in historical_stats:
                    historical_mean = historical_stats['weekday']['mean']
                elif 'holiday' in historical_stats:
                    historical_mean = historical_stats['holiday']['mean']
                else:
                    historical_mean = last_value
                # 逐渐回归历史均值
                decay_factor = 0.02
                current_val = current_val * (1 - decay_factor) + historical_mean * decay_factor
            predictions.append(max(0, current_val))

        predictions = np.array(predictions)

        # 异常日期的置信区间更宽（不确定性更高）
        lower_factor = 0.70  # 30%下限
        upper_factor = 1.30  # 30%上限

    else:
        # 正常日期：使用动态生成的模板
        if dynamic_patterns is None or not isinstance(dynamic_patterns, dict):
            raise ValueError("predict_remaining_day requires dynamic_patterns parameter from generate_pattern_from_history()")

        if day_type_str == 'weekday':
            base_pattern = dynamic_patterns['weekday'].copy()
        else:
            base_pattern = dynamic_patterns['holiday'].copy()
        print(f"使用动态生成的模板（{day_type_str}）")

        # 根据前9小时数据调整模板幅度（如果有当天数据）
        if len(first_9_hours) > 0:
            observed_mean = first_9_hours['value'].mean()
            template_mean = base_pattern[:len(first_9_hours)].mean()
            scale_factor = observed_mean / (template_mean + 1e-8)

            # 根据前9小时最后一个值调整偏移
            template_end_value = base_pattern[len(first_9_hours) - 1]
            offset = last_value - template_end_value * scale_factor

            print(f"模板调整参数 - 缩放因子: {scale_factor:.2f}, 偏移量: {offset:.2f}")

            # 提取剩余时间的模板值
            start_idx = len(first_9_hours)
            remaining_template = base_pattern[start_idx:start_idx + num_predictions]

            # 应用调整
            predictions = remaining_template * scale_factor + offset
        else:
            # 0点预测：使用完整的模板
            scale_factor = 1.0
            offset = 0
            remaining_template = base_pattern[:num_predictions]
            predictions = remaining_template.copy()
            print("0点预测模式：使用完整模板，不做调整")

        # 正常日期的置信区间
        lower_factor = 0.85
        upper_factor = 1.15

    # 确保预测值非负
    predictions = np.maximum(predictions, 0)

    # 创建结果DataFrame
    forecast_df = pd.DataFrame({
        'time': remaining_times,
        'value_predicted': predictions,
        'day_type': [day_type] * len(remaining_times),
        'value_lower': predictions * lower_factor,
        'value_upper': predictions * upper_factor
    })

    return forecast_df


class TimeSeriesDataProcessor:
    """
    时序数据预处理器

    训练前的数据处理环节，包括：
    1. 异常日期排除
    2. 时序通用数据处理项
    """

    def __init__(self, anomaly_dates=None, interpolation_method='linear',
                 outlier_method='iqr', outlier_threshold=1.5,
                 smoothing_window=None, min_daily_points=48):
        """
        参数:
            anomaly_dates: 异常日期集合，这些日期的数据将被排除
            interpolation_method: 缺失值插值方法 ('linear', 'time', 'pad', 'polynomial')
            outlier_method: 异常值处理方法 ('iqr', 'zscore', 'none')
            outlier_threshold: 异常值阈值 (IQR倍数或Z-score阈值)
            smoothing_window: 平滑窗口大小 (None表示不平滑)
            min_daily_points: 每天最少数据点数，不足则视为数据缺失
        """
        self.anomaly_dates = anomaly_dates or set()
        self.interpolation_method = interpolation_method
        self.outlier_method = outlier_method
        self.outlier_threshold = outlier_threshold
        self.smoothing_window = smoothing_window
        self.min_daily_points = min_daily_points

        # 记录处理统计
        self.processing_stats = {}

    def process(self, df):
        """
        执行完整的数据处理流程

        处理流程：
        1. 基础检查与排序
        2. 异常日期排除
        3. 缺失值处理（插值）
        4. 异常值处理
        5. 数据平滑
        6. 数据质量检查

        参数:
            df: 输入DataFrame，包含 'time' 和 'value' 列

        返回:
            处理后的DataFrame
        """
        print("\n" + "=" * 60)
        print("时序数据预处理")
        print("=" * 60)

        df = df.copy()
        original_shape = df.shape
        original_start = df['time'].min()
        original_end = df['time'].max()

        print(f"\n输入数据: {original_shape[0]} 行, {original_shape[1]} 列")
        print(f"时间范围: {original_start} 至 {original_end}")

        # 步骤1: 基础检查与排序
        df = self._basic_check_and_sort(df)

        # 步骤2: 异常日期排除
        df = self._exclude_anomaly_dates(df)

        # 步骤3: 缺失值处理
        df = self._handle_missing_values(df)

        # 步骤4: 异常值处理
        df = self._handle_outliers(df)

        # 步骤5: 数据平滑
        df = self._smooth_data(df)

        # 步骤6: 数据质量检查
        df = self._quality_check(df)

        # 输出处理统计
        final_shape = df.shape
        print("\n" + "-" * 40)
        print("数据处理统计")
        print("-" * 40)
        print(f"原始数据: {original_shape[0]} 行")
        print(f"处理后:   {final_shape[0]} 行")
        print(f"删除:     {original_shape[0] - final_shape[0]} 行 ({(1 - final_shape[0]/original_shape[0])*100:.1f}%)")

        if self.processing_stats:
            print("\n详细统计:")
            for key, value in self.processing_stats.items():
                print(f"  {key}: {value}")

        print("=" * 60)

        return df

    def _basic_check_and_sort(self, df):
        """基础检查与排序"""
        print("\n[步骤1] 基础检查与排序...")

        # 确保time列是datetime类型
        if not pd.api.types.is_datetime64_any_dtype(df['time']):
            df['time'] = pd.to_datetime(df['time'])

        # 按时间排序
        df = df.sort_values('time').reset_index(drop=True)

        # 检查value列类型
        if not pd.api.types.is_numeric_dtype(df['value']):
            try:
                df['value'] = pd.to_numeric(df['value'], errors='coerce')
                print("  ✓ value列已转换为数值类型")
            except:
                raise ValueError("value列无法转换为数值类型")

        # 检查时间间隔
        time_diff = df['time'].diff().dropna()
        if len(time_diff) > 0:
            median_diff = time_diff.median()
            mode_diff = time_diff.mode()[0] if len(time_diff.mode()) > 0 else median_diff
            print(f"  ✓ 时间间隔: {mode_diff} (中位数)")

            # 检查间隔是否一致
            unique_diffs = time_diff.unique()
            if len(unique_diffs) > 3:  # 允许多种相近的间隔
                print(f"  ⚠ 检测到 {len(unique_diffs)} 种不同时间间隔，可能需要重采样")

        self.processing_stats['原始数据点数'] = len(df)
        return df

    def _exclude_anomaly_dates(self, df):
        """排除异常日期"""
        print(f"\n[步骤2] 排除异常日期...")

        if not self.anomaly_dates:
            print("  - 未设置异常日期，跳过此步骤")
            return df

        original_count = len(df)
        df['date'] = df['time'].dt.date

        # 排除异常日期
        df_filtered = df[~df['date'].isin(self.anomaly_dates)].copy()
        excluded_count = original_count - len(df_filtered)

        # 统计被排除的日期
        excluded_dates = df[df['date'].isin(self.anomaly_dates)]['date'].unique()

        print(f"  ✓ 排除 {len(excluded_dates)} 个异常日期")
        print(f"  ✓ 删除 {excluded_count} 个数据点")
        print(f"  ✓ 剩余 {len(df_filtered)} 个数据点")

        self.processing_stats['排除异常日期数'] = len(excluded_dates)
        self.processing_stats['排除数据点数'] = excluded_count

        df_filtered = df_filtered.drop(columns=['date'])
        return df_filtered

    def _handle_missing_values(self, df):
        """缺失值处理"""
        print(f"\n[步骤3] 缺失值处理 (方法: {self.interpolation_method})...")

        # 检查缺失值
        missing_count = df['value'].isnull().sum()
        missing_time_count = df['time'].isnull().sum()

        if missing_time_count > 0:
            print(f"  ⚠ 发现 {missing_time_count} 个缺失时间戳，已删除")
            df = df.dropna(subset=['time'])

        if missing_count == 0:
            print("  ✓ 未发现缺失值")
            return df

        print(f"  ⚠ 发现 {missing_count} 个缺失值 ({missing_count/len(df)*100:.2f}%)")

        # 设置time为索引进行插值
        df_indexed = df.set_index('time')

        # 根据方法选择插值方式
        if self.interpolation_method == 'linear':
            df_indexed['value'] = df_indexed['value'].interpolate(method='time')
        elif self.interpolation_method == 'time':
            df_indexed['value'] = df_indexed['value'].interpolate(method='time')
        elif self.interpolation_method == 'pad':
            df_indexed['value'] = df_indexed['value'].fillna(method='pad').fillna(method='bfill')
        elif self.interpolation_method == 'polynomial':
            df_indexed['value'] = df_indexed['value'].interpolate(method='polynomial', order=2)
        else:
            df_indexed['value'] = df_indexed['value'].interpolate(method='linear')

        # 重置索引
        df = df_indexed.reset_index()

        # 检查是否还有缺失值
        remaining_missing = df['value'].isnull().sum()
        if remaining_missing > 0:
            print(f"  ⚠ 插值后仍有 {remaining_missing} 个缺失值，使用前向/后向填充")
            df['value'] = df['value'].fillna(method='ffill').fillna(method='bfill')

        print(f"  ✓ 缺失值处理完成")
        self.processing_stats['缺失值处理数'] = missing_count

        return df

    def _handle_outliers(self, df):
        """异常值处理"""
        print(f"\n[步骤4] 异常值处理 (方法: {self.outlier_method})...")

        if self.outlier_method == 'none':
            print("  - 跳过异常值处理")
            return df

        values = df['value'].copy()
        outlier_count = 0

        if self.outlier_method == 'iqr':
            # IQR方法
            Q1 = values.quantile(0.25)
            Q3 = values.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - self.outlier_threshold * IQR
            upper_bound = Q3 + self.outlier_threshold * IQR

            outlier_mask = (values < lower_bound) | (values > upper_bound)
            outlier_count = outlier_mask.sum()

            if outlier_count > 0:
                # 使用中位数替换异常值
                median_value = values.median()
                values = values.where(~outlier_mask, median_value)
                print(f"  ✓ IQR方法: 检测到 {outlier_count} 个异常值")
                print(f"    正常范围: [{lower_bound:.2f}, {upper_bound:.2f}]")
                print(f"    异常值已替换为中位数: {median_value:.2f}")

        elif self.outlier_method == 'zscore':
            # Z-score方法
            mean = values.mean()
            std = values.std()
            z_scores = np.abs((values - mean) / (std + 1e-8))
            outlier_mask = z_scores > self.outlier_threshold
            outlier_count = outlier_mask.sum()

            if outlier_count > 0:
                # 使用截断值替换
                lower_bound = mean - self.outlier_threshold * std
                upper_bound = mean + self.outlier_threshold * std
                values = values.clip(lower_bound, upper_bound)
                print(f"  ✓ Z-score方法: 检测到 {outlier_count} 个异常值")
                print(f"    正常范围: [{lower_bound:.2f}, {upper_bound:.2f}]")

        df['value'] = values
        self.processing_stats['异常值处理数'] = outlier_count

        if outlier_count == 0:
            print("  ✓ 未发现异常值")

        return df

    def _smooth_data(self, df):
        """数据平滑"""
        if not self.smoothing_window:
            return df

        print(f"\n[步骤5] 数据平滑 (窗口: {self.smoothing_window})...")

        # 使用移动平均进行平滑
        df['value'] = df['value'].rolling(window=self.smoothing_window,
                                          min_periods=1,
                                          center=True).mean()

        print(f"  ✓ 移动平均平滑完成")
        self.processing_stats['平滑窗口大小'] = self.smoothing_window

        return df

    def _quality_check(self, df):
        """数据质量检查"""
        print(f"\n[步骤6] 数据质量检查...")

        issues = []

        # 检查1: 是否存在缺失值
        if df['value'].isnull().any():
            issues.append(f"存在 {df['value'].isnull().sum()} 个缺失值")

        # 检查2: 是否存在负值
        negative_count = (df['value'] < 0).sum()
        if negative_count > 0:
            issues.append(f"存在 {negative_count} 个负值")
            print(f"  ⚠ 检测到负值，已截断为0")
            df['value'] = df['value'].clip(lower=0)

        # 检查3: 时间是否连续
        df_sorted = df.sort_values('time')
        time_diff = df_sorted['time'].diff().dropna()
        if len(time_diff) > 1:
            expected_diff = time_diff.mode()[0]
            gaps = time_diff[time_diff > expected_diff * 2]
            if len(gaps) > 0:
                issues.append(f"存在 {len(gaps)} 个时间间隔异常")

        # 检查4: 每天数据点数
        df['date'] = df['time'].dt.date
        daily_counts = df.groupby('date').size()
        low_data_days = daily_counts[daily_counts < self.min_daily_points]
        if len(low_data_days) > 0:
            issues.append(f"{len(low_data_days)} 天数据点数不足 ({self.min_daily_points})")
            print(f"  ⚠ {len(low_data_days)} 天数据量不足")

        df = df.drop(columns=['date'], errors='ignore')

        if issues:
            print(f"  ⚠ 发现 {len(issues)} 个问题:")
            for issue in issues:
                print(f"    - {issue}")
        else:
            print("  ✓ 数据质量检查通过")

        self.processing_stats['数据质量警告'] = len(issues)

        return df


def preprocess_training_data(df, anomaly_dates=None, **kwargs):
    """
    训练数据预处理便捷函数

    参数:
        df: 输入DataFrame
        anomaly_dates: 异常日期集合
        **kwargs: 传递给TimeSeriesDataProcessor的其他参数

    返回:
        处理后的DataFrame
    """
    processor = TimeSeriesDataProcessor(
        anomaly_dates=anomaly_dates,
        **kwargs
    )
    return processor.process(df)


def predict_daily_remaining(input_df, target_date=None, max_history_days=None,
                           interval_minutes=15, detect_anomaly=True, anomaly_z_threshold=3.0,
                           preprocess=True, interpolation_method='linear',
                           outlier_method='iqr', outlier_threshold=1.5):
    """
    主预测函数 - 预测当天剩余时间，支持多种时间间隔

    参数:
        input_df: 输入DataFrame，包含 'time' 和 'value' 列
        target_date: 目标日期（默认使用最新日期）
        max_history_days: 最大历史天数（None则根据时间间隔自动设置）
        interval_minutes: 时间间隔分钟数（15或60），根据此参数自动调整样本数量
        detect_anomaly: 是否启用异常日期检测
        anomaly_z_threshold: 异常检测Z-score阈值
        preprocess: 是否启用训练前数据预处理
        interpolation_method: 缺失值插值方法 ('linear', 'time', 'pad')
        outlier_method: 异常值处理方法 ('iqr', 'zscore', 'none')
        outlier_threshold: 异常值阈值 (IQR倍数或Z-score阈值)

    返回:
        预测结果DataFrame，包含当天剩余时间的预测值
    """
    print("=" * 60)
    print("时序数据预测 - 当天剩余时间预测")
    print("=" * 60)

    # 获取时间间隔配置
    config = get_interval_config(interval_minutes)

    # 根据时间间隔自动设置窗口大小（如果未指定）
    if max_history_days is None:
        max_history_days = config['train_window_days']
    stats_window_days = config['stats_window_days']

    print(f"\n配置信息:")
    print(f"  时间间隔: {interval_minutes}分钟 ({config['intervals_per_day']}点/天)")
    print(f"  检测时长: {config['hours_for_detection']}小时 ({config['points_for_detection']}点)")
    print(f"  统计窗口: {stats_window_days}天")
    print(f"  训练窗口: {max_history_days}天")

    start_time = datetime.now()

    # 步骤1: 训练前数据预处理
    if preprocess:
        print("\n" + "=" * 60)
        print("阶段1: 数据预处理")
        print("=" * 60)

        # 先进行异常日期检测（用于数据预处理）
        temp_df = load_and_preprocess_data(input_df.copy(), max_history_days=max_history_days, interval_minutes=interval_minutes)
        anomaly_dates_for_preprocess = set()
        if detect_anomaly:
            anomaly_dates_for_preprocess = detect_anomaly_days(temp_df, z_threshold=anomaly_z_threshold)

        # 数据预处理（使用根据时间间隔调整的最小每日点数）
        processor = TimeSeriesDataProcessor(
            anomaly_dates=anomaly_dates_for_preprocess,
            interpolation_method=interpolation_method,
            outlier_method=outlier_method,
            outlier_threshold=outlier_threshold,
            min_daily_points=config['min_daily_points']
        )
        processed_df = processor.process(input_df)
    else:
        processed_df = input_df.copy()

    # 步骤2: 计算历史统计特征（使用根据时间间隔调整的窗口确保节假日样本充足）
    print("\n" + "-" * 40)
    print("步骤: 计算历史统计特征")
    print("-" * 40)
    print(f"  统计窗口: 最近 {stats_window_days} 天（确保节假日样本充足）")
    print(f"  训练窗口: 最近 {max_history_days} 天（控制训练时间和内存）")

    # 使用更长的窗口计算统计特征，确保节假日样本充足
    stats_df = load_and_preprocess_data(processed_df, max_history_days=stats_window_days, interval_minutes=interval_minutes)

    # 异常日期检测（基于统计窗口）
    anomaly_dates = set()
    if detect_anomaly:
        print("\n" + "-" * 40)
        print("步骤: 异常日期检测")
        print("-" * 40)
        anomaly_dates = detect_anomaly_days(stats_df, z_threshold=anomaly_z_threshold)

    # 计算历史统计特征（基于统计窗口，排除异常日期）
    historical_stats = calculate_day_type_stats(stats_df, anomaly_dates=anomaly_dates)
    if historical_stats:
        print(f"\n历史统计特征 (基于 {stats_window_days} 天数据，已排除异常日期):")
        weekday_count = len([d for d in pd.to_datetime(stats_df['time']).dt.date.unique()
                            if d.weekday() < 5 and d not in anomaly_dates])
        holiday_count = len([d for d in pd.to_datetime(stats_df['time']).dt.date.unique()
                            if d.weekday() >= 5 and d not in anomaly_dates])
        if 'weekday' in historical_stats:
            print(f"  工作日 - 均值: {historical_stats['weekday']['mean']:.1f}, "
                  f"标准差: {historical_stats['weekday']['std']:.1f} (样本数: {weekday_count})")
        if 'holiday' in historical_stats:
            print(f"  假期   - 均值: {historical_stats['holiday']['mean']:.1f}, "
                  f"标准差: {historical_stats['holiday']['std']:.1f} (样本数: {holiday_count})")

    # 步骤3: 数据预处理（自动截取最近max_history_days天，用于实际训练/预测）
    df = load_and_preprocess_data(processed_df, max_history_days=max_history_days, interval_minutes=interval_minutes)

    # 获取前9小时数据（根据时间间隔自动计算点数）
    print("\n" + "-" * 40)
    print("步骤: 提取前9小时数据")
    print("-" * 40)
    first_9_hours, day_data = get_first_9_hours_data(df, target_date, interval_minutes=interval_minutes)

    # 根据时间间隔调整最小点数要求
    min_points_required = max(5, config['points_for_detection'] // 3)
    if len(first_9_hours) < min_points_required:
        raise ValueError(f"前{config['hours_for_detection']}小时数据不足（只有{len(first_9_hours)}个点，需要至少{min_points_required}个点），无法进行预测")

    # 判断日期类型（预测模式：只返回 weekday/holiday）
    print("\n" + "-" * 40)
    print("步骤: 判断日期类型（预测模式）")
    print("-" * 40)

    # 生成动态模板
    dynamic_patterns = generate_pattern_from_history(df, interval_minutes)

    target_date_obj = first_9_hours['time'].iloc[0].date()
    is_target_anomaly = target_date_obj in anomaly_dates
    day_type = detect_day_type(first_9_hours, historical_stats, is_anomaly=is_target_anomaly, mode='predict', dynamic_patterns=dynamic_patterns)

    # 确定预测时间范围（根据时间间隔）
    last_time = first_9_hours['time'].iloc[-1]
    current_date = last_time.date()
    day_end = pd.Timestamp(current_date) + timedelta(days=1)
    remaining_times = pd.date_range(
        start=last_time + timedelta(minutes=interval_minutes),
        end=day_end - timedelta(minutes=interval_minutes),
        freq=config['freq']
    )

    # 预测当天剩余时间
    print("\n" + "-" * 40)
    print("步骤: 预测当天剩余时间")
    print("-" * 40)
    forecast_df = predict_remaining_day(df, first_9_hours, day_type, day_data, historical_stats, interval_minutes=interval_minutes, remaining_times=remaining_times, dynamic_patterns=dynamic_patterns)

    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"\n预测完成，总耗时: {elapsed:.2f} 秒")

    return forecast_df


# ============================================================================
# 双阶段预测函数（简化接口）
# ============================================================================

def predict_at_midnight(input_df, interval_minutes=15, calendar_day_type_input=0, max_history_days=None):
    """
    0点预测 - 根据手动输入的日期类型进行预测

    参数:
        input_df: 历史数据DataFrame，包含 'time', 'value', 'day_type' 列
            - day_type: 日期类型 (1=工作日, 0=休息日, 其他=异常)
        interval_minutes: 时间间隔（15或60分钟）
        calendar_day_type_input: 手动输入的当天日期类型
            - 1: 工作日 (weekday)
            - 0: 休息日 (holiday)
            - 其他数字: 异常日期 (anomaly)
        max_history_days: 最大历史天数
            - None: 默认使用30天（15分钟）或60天（60分钟）
            - 0: 使用全部历史数据
            - 正整数: 使用指定天数

    返回:
        预测结果DataFrame，包含当日0-24时的预测值
        - 包含 'time', 'value_predicted', 'day_type', 'value_lower', 'value_upper' 列
    """
    print("=" * 60)
    print("0点预测 - 基于手动输入的日期类型")
    print("=" * 60)

    # 参数转换: 1=工作日, 0=休息日
    if calendar_day_type_input == 1:
        day_type = 'weekday'
    elif calendar_day_type_input == 0:
        day_type = 'holiday'
    else:
        day_type = 'anomaly'

    day_type_display = {1: '工作日', 0: '休息日'}.get(calendar_day_type_input, '异常日期')
    print(f"\n输入日期类型: {calendar_day_type_input} ({day_type_display})")
    print(f"内部类型: {day_type}")

    # 获取配置
    config = get_interval_config(interval_minutes)

    # 处理max_history_days参数
    # None: 使用默认值, 0: 使用全部数据, 正整数: 使用指定天数
    if max_history_days is None:
        use_max_days = config['train_window_days']
    elif max_history_days == 0:
        # 计算实际可用天数
        date_range = (input_df['time'].max() - input_df['time'].min()).days
        use_max_days = date_range
    else:
        use_max_days = max_history_days

    print(f"使用历史数据: 最近 {use_max_days} 天")

    # 数据预处理
    processed_df = preprocess_for_prediction(input_df, interval_minutes=interval_minutes, max_history_days=use_max_days)

    # 获取历史统计
    historical_stats = processed_df.get('historical_stats')
    anomaly_dates = processed_df.get('anomaly_dates', set())
    df = processed_df['df']

    # 确定目标日期（数据中最新日期的第二天）
    latest_date = df['time'].max().date()
    target_date = latest_date + timedelta(days=1)
    print(f"目标日期: {target_date}")

    # 生成预测时间范围（0-24点）
    day_start = pd.Timestamp(target_date)
    day_end = day_start + timedelta(days=1)
    remaining_times = pd.date_range(
        start=day_start,
        end=day_end - timedelta(minutes=interval_minutes),
        freq=config['freq']
    )

    print(f"预测时间范围: {remaining_times[0]} 至 {remaining_times[-1]}")
    print(f"预测点数: {len(remaining_times)}")

    # 生成动态模板
    dynamic_patterns = generate_pattern_from_history(df, interval_minutes)

    # 执行预测
    start_time = datetime.now()

    # 空的前9小时数据（0点预测没有当天数据）
    first_9_hours = pd.DataFrame(columns=['time', 'value'])
    day_data = pd.DataFrame(columns=['time', 'value'])

    # 使用模板预测
    forecast_df = predict_remaining_day(df, first_9_hours, day_type, day_data,
                                        historical_stats, interval_minutes=interval_minutes,
                                        remaining_times=remaining_times,
                                        dynamic_patterns=dynamic_patterns)

    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"\n预测完成，耗时: {elapsed:.2f} 秒")

    return forecast_df


def predict_at_nine(input_df, first_9_hours_df, interval_minutes=15, max_history_days=None):
    """
    9点预测 - 根据0-9点实际数据判断日期类型并预测剩余时间

    参数:
        input_df: 历史数据DataFrame，包含 'time', 'value', 'day_type' 列
            - day_type: 日期类型 (1=工作日, 0=休息日, 其他=异常)
        first_9_hours_df: 当日0-9点实际数据DataFrame，包含 'time' 和 'value' 列
        interval_minutes: 时间间隔（15或60分钟）
        max_history_days: 最大历史天数
            - None: 默认使用30天（15分钟）或60天（60分钟）
            - 0: 使用全部历史数据
            - 正整数: 使用指定天数

    返回:
        tuple: (日期类型, 预测结果DataFrame)
            - 日期类型: 1(工作日) / 0(休息日) / 2(异常)
            - 预测结果: 当日9-24时的预测值DataFrame
    """
    print("=" * 60)
    print("9点预测 - 基于0-9点实际数据")
    print("=" * 60)

    # 获取配置
    config = get_interval_config(interval_minutes)

    # 处理max_history_days参数
    if max_history_days is None:
        use_max_days = config['train_window_days']
    elif max_history_days == 0:
        date_range = (input_df['time'].max() - input_df['time'].min()).days
        use_max_days = date_range
    else:
        use_max_days = max_history_days

    print(f"使用历史数据: 最近 {use_max_days} 天")

    # 数据预处理
    processed_df = preprocess_for_prediction(input_df, interval_minutes=interval_minutes, max_history_days=use_max_days)

    # 获取历史统计
    historical_stats = processed_df.get('historical_stats')
    anomaly_dates = processed_df.get('anomaly_dates', set())
    df = processed_df['df']

    # 获取目标日期
    target_date = first_9_hours_df['time'].min().date()
    print(f"\n目标日期: {target_date}")
    print(f"0-9点数据点数: {len(first_9_hours_df)}")

    # 检查0-9点数据是否足够
    min_points = config['points_for_detection'] // 3
    if len(first_9_hours_df) < min_points:
        print(f"警告: 0-9点数据不足({len(first_9_hours_df)}点)，使用全部数据判断")

    # 生成动态模板
    dynamic_patterns = generate_pattern_from_history(df, interval_minutes)

    # 根据0-9点数据判断日期类型
    print("\n--- 日期类型判断 ---")
    is_target_anomaly = target_date in anomaly_dates
    day_type = detect_day_type(first_9_hours_df, historical_stats,
                                is_anomaly=is_target_anomaly, mode='predict',
                                dynamic_patterns=dynamic_patterns)

    day_type_display = {'weekday': '工作日', 'holiday': '节假日', 'anomaly': '异常日期'}
    print(f"推断的日期类型: {day_type} ({day_type_display.get(day_type, day_type)})")

    # 获取当天已有数据（0-9点）
    day_data = first_9_hours_df.copy()

    # 确定预测时间范围（9-24点）
    last_time = first_9_hours_df['time'].max()
    current_date = last_time.date()
    day_end = pd.Timestamp(current_date) + timedelta(days=1)
    remaining_times = pd.date_range(
        start=last_time + timedelta(minutes=interval_minutes),
        end=day_end - timedelta(minutes=interval_minutes),
        freq=config['freq']
    )

    print(f"\n预测时间范围: {remaining_times[0]} 至 {remaining_times[-1]}")
    print(f"预测点数: {len(remaining_times)}")

    # 执行预测（动态模板已在上面生成）
    start_time = datetime.now()

    # 使用模板预测
    forecast_df = predict_remaining_day(df, first_9_hours_df, day_type, day_data,
                                        historical_stats, interval_minutes=interval_minutes,
                                        remaining_times=remaining_times,
                                        dynamic_patterns=dynamic_patterns)

    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"\n预测完成，耗时: {elapsed:.2f} 秒")

    # 将字符串日期类型转换为数值类型返回
    # weekday=0, holiday=1, anomaly=2
    day_type_numeric = 1 if day_type == 'weekday' else (0 if day_type == 'holiday' else 2)

    return day_type_numeric, forecast_df


def evaluate_forecast_accuracy(forecast_df, actual_df, interval_minutes=15):
    """
    评估预测准确性 - 对比预测数据与实际数据

    参数:
        forecast_df: 当天预测数据DataFrame，包含 'time' 和 'value_predicted' 列
        actual_df: 当天实际数据DataFrame，包含 'time' 和 'value' 列
        interval_minutes: 时间间隔（15或60）

    返回:
        dict: 包含各种评估指标的字典
            - mape: 平均绝对百分比误差 (%)
            - mae: 平均绝对误差
            - rmse: 均方根误差
            - r2: R²决定系数
            - points_compared: 对比的数据点数
    """
    print("=" * 60)
    print("预测准确性评估")
    print("=" * 60)

    if len(actual_df) == 0:
        raise ValueError("实际数据为空")
    if len(forecast_df) == 0:
        raise ValueError("预测数据为空")

    # 确保有预测值列
    if 'value_predicted' in forecast_df.columns:
        pred_col = 'value_predicted'
    elif 'value' in forecast_df.columns:
        pred_col = 'value'
    else:
        raise ValueError("预测数据缺少value列")

    actual_df = actual_df.copy()
    forecast_df = forecast_df.copy()

    print(f"\n时间间隔: {interval_minutes}分钟")
    print(f"预测数据点数: {len(forecast_df)}")
    print(f"实际数据点数: {len(actual_df)}")

    # 对齐时间点
    actual_df = actual_df.set_index('time')
    forecast_df = forecast_df.set_index('time')

    # 找到共同的时间点
    common_times = actual_df.index.intersection(forecast_df.index)
    print(f"共同时间点数: {len(common_times)}")

    if len(common_times) == 0:
        print("警告: 预测与实际数据没有时间重叠")
        return None

    # 提取对齐后的数据
    actual_values = actual_df.loc[common_times, 'value'].values
    forecast_values = forecast_df.loc[common_times, pred_col].values

    # 计算各项评估指标
    # MAE (Mean Absolute Error)
    mae = np.mean(np.abs(actual_values - forecast_values))

    # MAPE (Mean Absolute Percentage Error)
    mask = actual_values != 0
    if np.any(mask):
        mape = np.mean(np.abs((actual_values[mask] - forecast_values[mask]) / actual_values[mask])) * 100
    else:
        mape = 0.0

    # RMSE (Root Mean Squared Error)
    rmse = np.sqrt(np.mean((actual_values - forecast_values) ** 2))

    # R² (Coefficient of Determination)
    ss_res = np.sum((actual_values - forecast_values) ** 2)
    ss_tot = np.sum((actual_values - np.mean(actual_values)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

    # 偏差分析
    bias = np.mean(forecast_values - actual_values)
    bias_pct = np.mean((forecast_values - actual_values) / (actual_values + 1e-8)) * 100

    # 输出结果
    print(f"\n评估指标:")
    print(f"  MAE   (平均绝对误差): {mae:.2f}")
    print(f"  MAPE  (平均绝对百分比误差): {mape:.2f}%")
    print(f"  RMSE  (均方根误差): {rmse:.2f}")
    print(f"  R²    (决定系数): {r2:.4f}")
    print(f"  BIAS  (平均偏差): {bias:.2f}")
    print(f"  BIAS% (平均偏差百分比): {bias_pct:.2f}%")

    result = {
        'mape': mape,
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'bias': bias,
        'bias_pct': bias_pct,
        'points_compared': len(common_times)
    }

    print(f"\n评估完成")
    return result


def evaluate_day_type_by_comparison(forecast_df, actual_df):
    """
    通过对比预测数据和实际数据评估日期类型

    参数:
        forecast_df: 当日预测数据DataFrame，包含 'time' 和 'value' 列
        actual_df: 当日实际数据DataFrame，包含 'time' 和 'value' 列

    返回:
        日期类型: 'weekday' / 'holiday' / 'anomaly'
    """
    print("=" * 60)
    print("日期类型评估 - 通过预测与实际对比")
    print("=" * 60)

    if len(actual_df) == 0:
        raise ValueError("实际数据为空")

    print(f"\n预测数据点数: {len(forecast_df)}")
    print(f"实际数据点数: {len(actual_df)}")

    # 计算预测与实际的误差
    # 需要对齐时间点
    actual_df = actual_df.set_index('time')
    forecast_df = forecast_df.set_index('time')

    # 找到共同的时间点
    common_times = actual_df.index.intersection(forecast_df.index)

    if len(common_times) == 0:
        print("警告: 预测与实际数据没有时间重叠")
        # 使用全部实际数据
        actual_values = actual_df['value'].values
    else:
        # 对齐后的数据
        actual_values = actual_df.loc[common_times, 'value'].values
        forecast_values = forecast_df.loc[common_times, 'value'].values

        # 计算误差指标
        mae = np.mean(np.abs(actual_values - forecast_values))
        mape = np.mean(np.abs((actual_values - forecast_values) / (actual_values + 1e-8))) * 100

        print(f"\n预测误差:")
        print(f"  MAE: {mae:.2f}")
        print(f"  MAPE: {mape:.2f}%")

    # 基于实际数据评估日期类型
    print("\n--- 评估日期类型 ---")
    evaluated_day_type = evaluate_day_type(actual_df.reset_index(), historical_stats=None,
                                            anomaly_dates=None, interval_minutes=15)

    day_type_display = {'weekday': '工作日', 'holiday': '节假日', 'anomaly': '异常日期'}
    print(f"评估结果: {evaluated_day_type} ({day_type_display.get(evaluated_day_type, evaluated_day_type)})")

    return evaluated_day_type


def preprocess_for_prediction(input_df, interval_minutes=15, max_history_days=None):
    """
    预测前的数据预处理（内部函数）

    参数:
        input_df: 原始数据DataFrame
        interval_minutes: 时间间隔
        max_history_days: 最大历史天数

    返回:
        包含预处理后数据和相关统计的字典
    """
    config = get_interval_config(interval_minutes)

    if max_history_days is None:
        max_history_days = config['train_window_days']
    stats_window_days = config['stats_window_days']

    # 加载并预处理数据
    df = load_and_preprocess_data(input_df, max_history_days=max_history_days, interval_minutes=interval_minutes)

    # 检测异常日期
    anomaly_dates = detect_anomaly_days(df, z_threshold=3.0)

    # 计算历史统计
    historical_stats = calculate_day_type_stats(df, anomaly_dates=anomaly_dates)

    if historical_stats:
        print(f"\n历史统计 (基于{stats_window_days}天数据):")
        if 'weekday' in historical_stats:
            print(f"  工作日 - 均值: {historical_stats['weekday']['mean']:.1f}, 标准差: {historical_stats['weekday']['std']:.1f}")
        if 'holiday' in historical_stats:
            print(f"  假期 - 均值: {historical_stats['holiday']['mean']:.1f}, 标准差: {historical_stats['holiday']['std']:.1f}")

    return {
        'df': df,
        'historical_stats': historical_stats,
        'anomaly_dates': anomaly_dates,
        'config': config
    }


def evaluate_day_type(full_day_data, historical_stats=None, anomaly_dates=None, interval_minutes=15, dynamic_patterns=None):
    """
    24小时后评估当天的日期类型（完整数据评估）

    基于全天完整数据进行三种类型判断：
    - weekday: 工作日
    - holiday: 节假日
    - anomaly: 异常日期

    参数:
        full_day_data: 当天完整数据（DataFrame，包含time和value列）
        historical_stats: 历史统计数据
        anomaly_dates: 已检测到的异常日期集合
        interval_minutes: 时间间隔分钟数（15或60）
        dynamic_patterns: 动态生成的模板 dict{'weekday': array, 'holiday': array}

    返回:
        'weekday', 'holiday' 或 'anomaly'
    """
    if len(full_day_data) == 0:
        raise ValueError("当天没有数据")

    # 获取配置
    config = get_interval_config(interval_minutes)
    intervals_per_day = config['intervals_per_day']

    # 获取日期
    date_obj = full_day_data['time'].iloc[0].date()

    # 检查是否已标记为异常日期
    is_anomaly = anomaly_dates is not None and date_obj in anomaly_dates

    # 提取全天数据特征
    values = full_day_data['value'].values
    mean_value = np.mean(values)
    std_value = np.std(values)
    min_value = np.min(values)
    max_value = np.max(values)
    range_value = max_value - min_value

    # 计算白天(6-22点)和夜间(0-6, 22-24点)的均值差异
    full_day_data_copy = full_day_data.copy()
    full_day_data_copy['hour'] = full_day_data_copy['time'].dt.hour
    day_hours = full_day_data_copy[full_day_data_copy['hour'].between(6, 21)]
    night_hours = full_day_data_copy[~full_day_data_copy['hour'].between(6, 21)]

    day_mean = day_hours['value'].mean() if len(day_hours) > 0 else 0
    night_mean = night_hours['value'].mean() if len(night_hours) > 0 else 0
    day_night_diff = day_mean - night_mean

    print(f"\n全天数据特征:")
    print(f"  日期: {date_obj}")
    print(f"  时间间隔: {interval_minutes}分钟")
    print(f"  数据点数: {len(values)} (预期{intervals_per_day}点/天)")
    print(f"  平均值: {mean_value:.2f}")
    print(f"  标准差: {std_value:.2f}")
    print(f"  最小值: {min_value:.2f}")
    print(f"  最大值: {max_value:.2f}")
    print(f"  极差: {range_value:.2f}")
    print(f"  白天均值: {day_mean:.2f}, 夜间均值: {night_mean:.2f}, 差值: {day_night_diff:.2f}")

    # 计算与模板的相似度（使用完整的全天数据）
    if len(values) >= intervals_per_day:
        full_values = values[:intervals_per_day]
    else:
        # 数据不足，用边缘值填充
        full_values = np.pad(values, (0, intervals_per_day - len(values)), mode='edge')

    # 必须使用动态模板
    if dynamic_patterns is None or not isinstance(dynamic_patterns, dict):
        raise ValueError("evaluate_day_type requires dynamic_patterns parameter from generate_pattern_from_history()")

    weekday_pattern = dynamic_patterns.get('weekday')
    holiday_pattern = dynamic_patterns.get('holiday')

    weekday_similarity = calculate_pattern_similarity(full_values, weekday_pattern)
    holiday_similarity = calculate_pattern_similarity(full_values, holiday_pattern)

    print(f"  与工作日模板相似度: {weekday_similarity:.4f}")
    print(f"  与假期模板相似度: {holiday_similarity:.4f}")

    # 异常检测（基于全天数据）
    anomaly_reasons = []

    # 1. 与两种模板都不相似
    max_similarity = max(weekday_similarity, holiday_similarity)
    if max_similarity < 0.4:
        anomaly_reasons.append(f"与模板相似度过低({max_similarity:.3f})")

    # 2. 偏离历史统计
    if historical_stats:
        if 'weekday' in historical_stats:
            wd_mean = historical_stats['weekday']['mean']
            wd_std = historical_stats['weekday']['std']
            if abs(mean_value - wd_mean) > 3 * wd_std:
                anomaly_reasons.append(f"均值偏离工作日({abs(mean_value - wd_mean):.1f} > 3σ)")

        if 'holiday' in historical_stats:
            hl_mean = historical_stats['holiday']['mean']
            hl_std = historical_stats['holiday']['std']
            if abs(mean_value - hl_mean) > 3 * hl_std:
                anomaly_reasons.append(f"均值偏离假期({abs(mean_value - hl_mean):.1f} > 3σ)")

    # 3. 极差异常
    if range_value < 10:
        anomaly_reasons.append("极差过小(几乎无波动)")
    elif range_value > 500:
        anomaly_reasons.append("极差过大(变化过于剧烈)")

    # 4. 白天夜间差异异常（工作日应该有明显差异）
    if is_anomaly:
        # 如果是预标记的异常日期，直接返回anomaly
        print(f"\n评估结果: ANOMALY (历史预标记异常日期)")
        return 'anomaly'

    if len(anomaly_reasons) > 0:
        print(f"\n评估结果: ANOMALY")
        print(f"  异常原因: {', '.join(anomaly_reasons)}")
        return 'anomaly'

    # 非异常日期，判断为工作日或假期
    if weekday_similarity >= holiday_similarity:
        print(f"\n评估结果: WEEKDAY")
        return 'weekday'
    else:
        print(f"\n评估结果: HOLIDAY")
        return 'holiday'


def batch_evaluate_days(df, historical_stats=None, anomaly_dates=None, interval_minutes=15):
    """
    批量评估多个日期的类型（用于24小时后评估）

    参数:
        df: 包含多天数据的DataFrame
        historical_stats: 历史统计数据
        anomaly_dates: 异常日期集合
        interval_minutes: 时间间隔分钟数（15或60）

    返回:
        dict: {日期: 日期类型}
    """
    df = df.copy()
    df['date'] = df['time'].dt.date

    results = {}
    for date_obj in df['date'].unique():
        day_data = df[df['date'] == date_obj]
        day_type = evaluate_day_type(day_data, historical_stats, anomaly_dates, interval_minutes=interval_minutes)
        results[date_obj] = day_type

    return results


def generate_sample_data(days=800, pattern_type='mixed', interval_minutes=15):
    """
    生成示例数据用于测试 - 支持生成2年以上数据

    参数:
        days: 生成天数（默认800天≈2.2年）
        pattern_type: 'weekday', 'weekend', 'mixed'
        interval_minutes: 时间间隔分钟数（15或60）
    """
    np.random.seed(42)

    # 根据时间间隔确定频率
    freq = '15min' if interval_minutes == 15 else '60min'
    points_per_hour = 4 if interval_minutes == 15 else 1

    # 生成时间序列
    end_time = datetime.now().replace(hour=18, minute=0, second=0, microsecond=0)
    start_time = end_time - timedelta(days=days)

    time_range = pd.date_range(start=start_time, end=end_time, freq=freq)

    values = []
    day_types = []
    for t in time_range:
        hour = t.hour
        is_weekend = t.weekday() >= 5

        # 设置day_type: 1=工作日, 0=休息日
        day_type = 0 if is_weekend else 1

        # 季节性因子（年周期）
        day_of_year = t.timetuple().tm_yday
        seasonal_factor = 1 + 0.1 * np.sin(2 * np.pi * day_of_year / 365)

        if is_weekend:
            # 假期模式：较低的活动量，平缓
            base = (50 + 30 * np.sin(2 * np.pi * (hour - 6) / 18)) * seasonal_factor if 6 <= hour <= 24 else 40 * seasonal_factor
            noise = np.random.normal(0, 5)
        else:
            # 工作日模式：早晚高峰
            if 7 <= hour <= 9:  # 早高峰
                base = (150 + 50 * np.sin(2 * np.pi * (hour - 7) / 3)) * seasonal_factor
            elif 17 <= hour <= 19:  # 晚高峰
                base = (160 + 40 * np.sin(2 * np.pi * (hour - 17) / 3)) * seasonal_factor
            elif 9 <= hour <= 17:  # 工作时间
                base = (130 + 20 * np.sin(2 * np.pi * hour / 24)) * seasonal_factor
            else:  # 夜间
                base = (40 + 10 * np.sin(2 * np.pi * hour / 24)) * seasonal_factor
            noise = np.random.normal(0, 8)

        values.append(max(0, base + noise))
        day_types.append(day_type)

    df = pd.DataFrame({
        'time': time_range,
        'value': values,
        'day_type': day_types
    })

    return df


if __name__ == '__main__':
    import time

    # 记录开始时间
    start_time = time.time()

    # 生成示例数据（模拟2年历史数据）
    print("\n生成示例数据（模拟2年历史数据，约70000+数据点）...")
    sample_df = generate_sample_data(days=800, pattern_type='mixed')
    print(f"示例数据形状: {sample_df.shape}")
    print(f"数据时间跨度: {sample_df['time'].min().date()} 至 {sample_df['time'].max().date()}")

    # 执行预测
    try:
        # 使用默认参数（自动截取最近90天）
        value_predict_df = predict_daily_remaining(sample_df)

        # 显示结果
        print("\n" + "=" * 60)
        print("预测结果：")
        print("=" * 60)
        print(value_predict_df.head(10))
        print("...")
        print(value_predict_df.tail(10))

        print("\n" + "=" * 60)
        print(f"预测结果统计：")
        print("=" * 60)
        print(value_predict_df['value_predicted'].describe())

        # 保存结果（可选）
        # value_predict_df.to_csv('forecast_result.csv', index=False)
        # print("\n结果已保存到 forecast_result.csv")

    except Exception as e:
        print(f"预测失败: {e}")
        import traceback
        traceback.print_exc()

    # 记录结束时间
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("\n" + "=" * 60)
    print(f"总运行时间: {elapsed_time:.2f} 秒 ({elapsed_time/60:.2f} 分钟)")
    print("=" * 60)

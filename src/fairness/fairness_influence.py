import pandas as pd
from itertools import combinations
import numpy as np


def generate_multi_patterns(df, max_predicates=2, bin_numerical=True, bins=3):
    """生成最多 max_predicates 条件的组合 pattern"""
    simple_patterns = generate_simple_patterns(df, bin_numerical, bins)
    patterns = []

    for r in range(1, max_predicates + 1):
        for combo in combinations(simple_patterns, r):
            merged = {}
            conflict = False
            for p in combo:
                for k, v in p.items():
                    if k in merged and merged[k] != v:
                        conflict = True  # 避免冲突条件（如 age ∈ A 和 age ∈ B）
                        break
                    merged[k] = v
                if conflict:
                    break
            if not conflict:
                patterns.append(merged)
    return patterns


def generate_simple_patterns(df, bin_numerical=True, bins=3):
    """生成所有一阶谓词（单特征谓词）"""
    patterns = []
    for col in df.columns[:-1]:  # exclude 'influence'
        if pd.api.types.is_numeric_dtype(df[col]) and bin_numerical:
            binned = pd.qcut(df[col], bins, duplicates='drop')
            for interval in binned.unique():
                patterns.append({col: interval})
        else:
            for val in df[col].unique():
                patterns.append({col: val})
    return patterns

def pattern_support(df, pattern):
    """计算 pattern 覆盖的样本子集及其支持度"""
    mask = np.ones(len(df), dtype=bool)
    for col, val in pattern.items():
        if isinstance(val, pd.Interval):
            mask &= df[col].between(val.left, val.right, inclusive='left')
        else:
            mask &= df[col] == val
    return df[mask], mask.sum() / len(df)

def evaluate_patterns(df, min_support=0.01, top_k=5):
    """主函数：从 influence data 中找出 top-k 高 interestingness 的群体 pattern"""
    #patterns = generate_simple_patterns(df)
    patterns = generate_multi_patterns(df)

    results = []

    total_influence = df["influence"].sum()

    for pattern in patterns:
        subset, support = pattern_support(df, pattern)
        if support < min_support:
            continue

        responsibility = subset["influence"].sum() / total_influence
        interestingness = responsibility / support if support > 0 else 0

        results.append({
            "pattern": pattern,
            "support": support,
            "responsibility": responsibility,
            "interestingness": interestingness
        })

    # 排序选出 top-k
    results = sorted(results, key=lambda x: -x["interestingness"])
    return results[:top_k]
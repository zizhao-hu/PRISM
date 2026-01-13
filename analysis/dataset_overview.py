#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
import math
import json
import statistics as stats
from collections import Counter, defaultdict
from dataclasses import dataclass

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
DATA_DIR = os.path.join(BASE_DIR, 'benchmark_synthetic_dataset')
FIG_DIR = os.path.join(BASE_DIR, 'analysis', 'figures')
OUT_MD = os.path.join(BASE_DIR, 'analysis', 'overview.md')
OUT_JSON = os.path.join(BASE_DIR, 'analysis', 'overview.json')

os.makedirs(FIG_DIR, exist_ok=True)

# Paths
PC_PATH = os.path.join(DATA_DIR, 'personal_chat', 'personal_chat.csv')
SP_PATH = os.path.join(DATA_DIR, 'safety_protocol', 'safety_protocol.csv')
CD_PATH = os.path.join(DATA_DIR, 'coding', 'coding.csv')

code_fence_pattern = re.compile(r"```([a-zA-Z0-9_+-]*)\n([\s\S]*?)```", re.MULTILINE)
only_code_fence_pattern = re.compile(r"```[\s\S]*?```", re.MULTILINE)

LANG_MAP = {
    'ts': 'TypeScript/JS', 'tsx': 'TypeScript/JS', 'js': 'TypeScript/JS', 'jsx': 'TypeScript/JS',
    'yaml': 'YAML', 'yml': 'YAML',
    'sql': 'SQL', 'psql': 'SQL',
    'py': 'Python', 'python': 'Python',
    'go': 'Go', 'golang': 'Go',
    'java': 'Java', 'kotlin': 'Kotlin', 'kt': 'Kotlin',
    'rs': 'Rust', 'rust': 'Rust',
    'sh': 'Bash', 'bash': 'Bash', 'shell': 'Bash',
    'rego': 'Rego', 'hcl': 'HCL',
    'graphql': 'GraphQL', 'gql': 'GraphQL',
    'proto': 'Protobuf', 'protobuf': 'Protobuf',
    'nginx': 'Nginx', 'conf': 'Config',
    'json': 'JSON', 'md': 'Markdown', 'latex': 'LaTeX', 'tex': 'LaTeX',
}

@dataclass
class Summary:
    name: str
    samples: int
    structure: str
    content_type: str
    median_turns: float | None
    min_turns: int | None
    max_turns: int | None
    code_presence_pct: float | None
    median_words: float
    p95_words: float


def count_words(text: str) -> int:
    if not isinstance(text, str):
        return 0
    # Split on whitespace; count tokens
    return len(re.findall(r"\b\w+\b", text))


def analyze_personal_chat(path: str) -> tuple[Summary, pd.DataFrame]:
    df = pd.read_csv(path)
    # Assumed columns: conversation_id, turn_index, role, message
    df['message'] = df['message'].fillna('')
    df['words'] = df['message'].apply(count_words)
    g = df.groupby('conversation_id')
    turns = g['turn_index'].max().astype(int)
    words_per_conv = g['words'].sum()
    summary = Summary(
        name='Personal Chat',
        samples=g.ngroups,
        structure='Multi-turn dialogs',
        content_type='Text',
        median_turns=float(turns.median()),
        min_turns=int(turns.min()),
        max_turns=int(turns.max()),
        code_presence_pct=0.0,
        median_words=float(words_per_conv.median()),
        p95_words=float(words_per_conv.quantile(0.95)),
    )
    # Figures: turns distribution, words distribution
    plt.figure(figsize=(4,2.5), dpi=200)
    sns.boxplot(x=turns, color='#4c72b0')
    plt.xlabel('Turns per conversation')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'pc_turns_box.png'))
    plt.close()

    plt.figure(figsize=(4,2.5), dpi=200)
    sns.histplot(words_per_conv, bins=20, color='#55a868')
    plt.xlabel('Words per conversation')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'pc_words_hist.png'))
    plt.close()

    return summary, g[['turn_index', 'words']].sum().reset_index(names='conversation_id')


def analyze_safety_protocol(path: str) -> tuple[Summary, pd.DataFrame]:
    df = pd.read_csv(path, quotechar='"', skipinitialspace=True)
    # Columns: document_id,title,content
    df['content'] = df['content'].fillna('')
    df['words'] = df['content'].apply(count_words)
    words = df['words']
    summary = Summary(
        name='Safety Protocol',
        samples=len(df),
        structure='Long-form documents',
        content_type='Text (policy)',
        median_turns=None,
        min_turns=None,
        max_turns=None,
        code_presence_pct=0.0,
        median_words=float(words.median()),
        p95_words=float(words.quantile(0.95)),
    )
    # Figure: words distribution
    plt.figure(figsize=(4,2.5), dpi=200)
    sns.histplot(words, bins=20, color='#c44e52')
    plt.xlabel('Words per document')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'sp_words_hist.png'))
    plt.close()
    return summary, df[['document_id', 'words']]


def detect_languages_from_code(message: str) -> list[str]:
    langs = []
    for m in code_fence_pattern.finditer(message or ''):
        lang = (m.group(1) or '').strip().lower()
        if not lang:
            continue
        mapped = LANG_MAP.get(lang, lang.upper())
        langs.append(mapped)
    return langs


def analyze_coding(path: str) -> tuple[Summary, pd.DataFrame, Counter]:
    df = pd.read_csv(path)
    df['message'] = df['message'].fillna('')
    # Count code blocks per message
    df['code_blocks'] = df['message'].apply(lambda s: len(list(code_fence_pattern.finditer(s))) or len(list(only_code_fence_pattern.finditer(s))))
    df['has_code'] = df['code_blocks'] > 0
    # Words including code
    df['words_all'] = df['message'].apply(count_words)
    # Words excluding fenced code
    def strip_code(s: str) -> str:
        return only_code_fence_pattern.sub(' ', s or '')
    df['message_text'] = df['message'].apply(strip_code)
    df['words_text'] = df['message_text'].apply(count_words)
    # Languages
    df['langs'] = df['message'].apply(detect_languages_from_code)

    g = df.groupby('conversation_id')
    turns = g['turn_index'].max().astype(int)
    words_all = g['words_all'].sum()
    words_text = g['words_text'].sum()
    code_blocks_per_conv = g['code_blocks'].sum()
    has_any_code = (code_blocks_per_conv > 0).mean() * 100.0

    # Aggregate language counts at conversation level (unique langs per scenario) and message level
    lang_counter = Counter()
    for langs in df['langs']:
        lang_counter.update(langs)

    summary = Summary(
        name='Coding',
        samples=g.ngroups,
        structure='Multi-turn dialogs',
        content_type='Text + Code',
        median_turns=float(turns.median()),
        min_turns=int(turns.min()),
        max_turns=int(turns.max()),
        code_presence_pct=float(has_any_code),
        median_words=float(words_all.median()),
        p95_words=float(words_all.quantile(0.95)),
    )

    # Figures
    plt.figure(figsize=(4,2.5), dpi=200)
    sns.boxplot(x=turns, color='#8172b3')
    plt.xlabel('Turns per scenario')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'cd_turns_box.png'))
    plt.close()

    plt.figure(figsize=(4,2.5), dpi=200)
    sns.histplot(words_all, bins=20, color='#937860')
    plt.xlabel('Words per scenario (incl. code)')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'cd_words_hist.png'))
    plt.close()

    # Top languages
    top_langs = lang_counter.most_common(12)
    if top_langs:
        labels, counts = zip(*top_langs)
        plt.figure(figsize=(5,2.8), dpi=200)
        sns.barplot(x=list(counts), y=list(labels), orient='h', color='#4c72b0')
        plt.xlabel('Code blocks (by fence tag)')
        plt.ylabel('Language')
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, 'cd_langs_top.png'))
        plt.close()

    return summary, g[['turn_index', 'words_all', 'words_text', 'code_blocks']].sum().reset_index(names='conversation_id'), lang_counter


def save_overview_md(pc_sum: Summary, sp_sum: Summary, cd_sum: Summary, lang_counter: Counter):
    def fmt(x):
        if x is None:
            return '—'
        if isinstance(x, float):
            return f"{x:.1f}"
        return str(x)

    lines = []
    lines.append('# Dataset Overview\n')
    lines.append('| Dataset | Samples | Structure | Content type | Median turns (min–max) | Code presence | Median words | P95 words |')
    lines.append('|---|---:|---|---|---:|---:|---:|---:|')
    for s in [pc_sum, sp_sum, cd_sum]:
        mt = '—' if s.median_turns is None else f"{s.median_turns:.1f} ({s.min_turns}-{s.max_turns})"
        code_pct = '—' if s.code_presence_pct is None else f"{s.code_presence_pct:.1f}%"
        lines.append(f"| {s.name} | {s.samples} | {s.structure} | {s.content_type} | {mt} | {code_pct} | {s.median_words:.1f} | {s.p95_words:.1f} |")

    lines.append('\n## Coding dataset: top languages (by fenced blocks)\n')
    top_langs = lang_counter.most_common(15)
    if top_langs:
        lines.append('| Language | Code blocks |')
        lines.append('|---|---:|')
        for k, v in top_langs:
            lines.append(f"| {k} | {v} |")
    with open(OUT_MD, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


def main():
    print('Analyzing datasets...')
    pc_sum, _pc_df = analyze_personal_chat(PC_PATH)
    sp_sum, _sp_df = analyze_safety_protocol(SP_PATH)
    cd_sum, _cd_df, lang_counter = analyze_coding(CD_PATH)

    overview = {
        'personal_chat': pc_sum.__dict__,
        'safety_protocol': sp_sum.__dict__,
        'coding': cd_sum.__dict__,
        'coding_top_languages': dict(lang_counter.most_common(30)),
        'figures_dir': os.path.relpath(FIG_DIR, BASE_DIR),
    }
    with open(OUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(overview, f, indent=2)

    save_overview_md(pc_sum, sp_sum, cd_sum, lang_counter)
    # Combined figure: samples per dataset
    plt.figure(figsize=(3.6,2.4), dpi=200)
    names = ['Personal Chat', 'Safety Protocol', 'Coding']
    counts = [pc_sum.samples, sp_sum.samples, cd_sum.samples]
    sns.barplot(x=names, y=counts, color='#4c72b0')
    plt.ylabel('Samples')
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'samples_per_dataset.png'))
    plt.close()

    print('Done. Outputs:')
    print(' -', os.path.relpath(OUT_MD, BASE_DIR))
    print(' -', os.path.relpath(OUT_JSON, BASE_DIR))
    for fn in sorted(os.listdir(FIG_DIR)):
        print(' - figures/', fn)


if __name__ == '__main__':
    main()

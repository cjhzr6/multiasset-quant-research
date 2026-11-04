"""Rebuild publication figures from public derived results, without vendor data."""
from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import PercentFormatter

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'figure_data'
OUT = ROOT / 'figures'
OUT.mkdir(exist_ok=True)
plt.rcParams.update({
    'font.family': ['Microsoft YaHei', 'DejaVu Sans'],
    'font.size': 9, 'axes.titlesize': 10, 'axes.labelsize': 9,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.unicode_minus': False, 'pdf.fonttype': 42, 'svg.fonttype': 'none',
    'figure.facecolor': 'white', 'savefig.facecolor': 'white',
})
COLORS = {'RISK-B1': '#0072B2', '固定25%': '#D55E00'}
checks = {}

def finish(fig, name):
    for ext in ('pdf', 'svg', 'png'):
        fig.savefig(OUT / f'{name}.{ext}', dpi=300, bbox_inches='tight')
    plt.close(fig)

def nav_chart(file, name, locator):
    frame = pd.read_csv(DATA / file, parse_dates=['date'])
    assert not frame.duplicated(['date', 'strategy']).any()
    fig, axes = plt.subplots(2, 1, figsize=(6.7, 4.35), sharex=True,
                             gridspec_kw={'height_ratios': [2.2, 1], 'hspace': .09})
    stats = {}
    for strategy in ('RISK-B1', '固定25%'):
        d = frame[frame.strategy.eq(strategy)].sort_values('date')
        nav = d.nav.to_numpy(float)
        assert np.isfinite(nav).all() and (nav > 0).all()
        ret = nav / np.r_[1., nav[:-1]] - 1
        peak = np.maximum.accumulate(np.r_[1., nav])[1:]
        dd = nav / peak - 1
        style = '-' if strategy == 'RISK-B1' else '--'
        axes[0].plot(d.date, nav, color=COLORS[strategy], ls=style, lw=1.6, label=strategy)
        axes[1].plot(d.date, dd, color=COLORS[strategy], ls=style, lw=1.15)
        stats[strategy] = {'sessions': len(d), 'return': float(nav[-1] - 1),
            'sharpe_zero_cash': float(np.sqrt(252)*ret.mean()/ret.std(ddof=1)),
            'max_drawdown': float(dd.min()), 'first_date': str(d.date.min().date()),
            'last_date': str(d.date.max().date())}
    axes[0].axhline(1, color='#999999', lw=.6)
    axes[0].set_ylabel('净值（初始 = 1）')
    axes[0].legend(loc='upper left', frameon=False, ncol=2)
    axes[1].set_ylabel('回撤')
    axes[1].yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    axes[1].xaxis.set_major_locator(mdates.MonthLocator(interval=locator))
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    for ax in axes:
        ax.grid(axis='y', color='#dddddd', lw=.55)
        ax.margins(x=.01)
    fig.align_ylabels(axes)
    finish(fig, name)
    checks[name] = stats

nav_chart('nav_2024_2025.csv', 'historical_nav_drawdown', 4)
nav_chart('nav_2026.csv', 'extension_nav_drawdown', 2)

d = pd.read_csv(DATA / 'weights_2026.csv', parse_dates=['date'])
components = ['股票', '标普500', '黄金', '国债', '现金']
values = d[components].to_numpy(float)
assert np.isfinite(values).all() and (values >= -1e-10).all()
checks['weights'] = {'rows': len(d), 'minimum_sum': float(values.sum(axis=1).min()),
                     'maximum_sum': float(values.sum(axis=1).max()),
                     'means': d[components].mean().to_dict()}
fig, ax = plt.subplots(figsize=(6.7, 3.15))
ax.stackplot(d.date, *[d[c] for c in components], labels=components,
             colors=['#0072B2', '#56B4E9', '#E69F00', '#009E73', '#d6d6d6'], linewidth=0)
ax.set_ylim(0, 1.01)
ax.yaxis.set_major_formatter(PercentFormatter(1))
ax.set_ylabel('实际市值 / 净资产')
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax.legend(loc='upper center', bbox_to_anchor=(.5, 1.17), ncol=5, frameon=False)
ax.margins(x=0)
finish(fig, 'actual_weights_2026')

d = pd.read_csv(DATA / 'contributions.csv')
labels = {'physical_gold': '黄金ETF', 'sp500_qdii': '标普500 ETF',
          'rmb_gov_bond': '国债ETF', 'stock_and_unallocated': '股票及未分拆项*'}
fig, axes = plt.subplots(1, 2, figsize=(6.7, 3.15))
checks['contributions'] = {}
for ax, period, title in zip(axes, ['2024_2025', '2026'], ['2024—2025', '2026补充期（175日）']):
    x = np.arange(4)
    for j, (prefix, strategy) in enumerate([('B1', 'RISK-B1'), ('FIXED25', '固定25%')]):
        row = d[d.account.eq(prefix+'_'+period)].set_index('component')
        vals = row.loc[list(labels), 'contribution_pp_on_initial'].to_numpy()
        ax.bar(x+(j-.5)*.36, vals, width=.34, color=COLORS[strategy],
               hatch=None if j==0 else '//', label=strategy)
        checks['contributions'][prefix+'_'+period] = float(vals.sum())
    ax.axhline(0, color='#888888', lw=.6)
    ax.set_xticks(x, list(labels.values()), rotation=20, ha='right', fontsize=8)
    ax.set_title(title)
    ax.grid(axis='y', color='#dddddd', lw=.5)
    ax.set_axisbelow(True)
axes[0].set_ylabel('对初始资金的收益贡献（百分点）')
axes[0].legend(frameon=False, fontsize=8)
fig.subplots_adjust(wspace=.32, bottom=.2)
finish(fig, 'component_contributions')

# Monthly returns are derived directly from each standalone account, not stitched across periods.
monthly = []
for file, period in [('nav_2024_2025.csv', '2024_2025'), ('nav_2026.csv', '2026_restart')]:
    f = pd.read_csv(DATA/file, parse_dates=['date'])
    for strategy, group in f.groupby('strategy'):
        end = group.set_index('date').nav.resample('ME').last().dropna()
        r = end.to_numpy()/np.r_[1., end.to_numpy()[:-1]]-1
        monthly += [{'period': period, 'month': str(t.to_period('M')), 'strategy': strategy, 'return': float(v)} for t,v in zip(end.index, r)]
monthly_frame = pd.DataFrame(monthly)
monthly_frame.to_csv(DATA/'monthly_returns.csv', index=False, encoding='utf-8')
lookup = monthly_frame.set_index(['month','strategy'])['return'].to_dict()
table = [r'\begin{table}[htbp]\centering\small',
 r'\caption{标准成本月度收益（\%）}\label{tab:monthly}',
 r'\begin{tabular}{@{}lrrlrrlrr@{}}\toprule',
 r'月份 & B1 & 固定 & 月份 & B1 & 固定 & 月份 & B1 & 固定 \\\midrule']
for month in range(1,13):
    cells = []
    for year in (2024,2025,2026):
        key = f'{year}-{month:02d}'
        if (key,'RISK-B1') in lookup:
            cells += [key, f"{lookup[key,'RISK-B1']*100:.2f}", f"{lookup[key,'固定25%']*100:.2f}"]
        else:
            cells += ['--','--','--']
    table.append(' & '.join(cells)+r' \\')
table += [r'\bottomrule\end{tabular}',
 r'\note{B1与固定分别指RISK-B1及固定25\%。2024—2025连续；2026重新以现金起步，一月从100万元计算，九月仅截至21日。表由日净值生成；未出现的月份为无评价，不是零收益。}',
 r'\end{table}']
(ROOT/'monthly_table.tex').write_text('\n'.join(table)+'\n',encoding='utf-8')
# Reconcile independently derived monthly returns and components to the account endpoint.
for period, chart in [('2024_2025','historical_nav_drawdown'),('2026','extension_nav_drawdown')]:
    monthly_period = '2026_restart' if period == '2026' else period
    for prefix, strategy in [('B1','RISK-B1'),('FIXED25','固定25%')]:
        expected = checks[chart][strategy]['return']
        contribution = checks['contributions'][prefix+'_'+period]/100
        month_values = monthly_frame.loc[monthly_frame.period.eq(monthly_period)&monthly_frame.strategy.eq(strategy),'return'].to_numpy()
        compounded = float(np.prod(1+month_values)-1)
        assert abs(contribution-expected)<1e-10, (period,strategy,'component mismatch')
        assert abs(compounded-expected)<1e-10, (period,strategy,'monthly mismatch')
checks['reconciliation'] = 'Component sums and compounded monthly returns agree with each standalone account endpoint, tolerance 1e-10.'
checks['files'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(DATA.glob('*.csv'))}
(ROOT/'figure_checks.json').write_text(json.dumps(checks, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(checks, ensure_ascii=False, indent=2))

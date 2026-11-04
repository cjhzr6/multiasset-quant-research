# 正式策略研报（2026-11-05公开修订）

[阅读PDF](report.pdf) · [LaTeX正文](main.tex) · [图表脚本](build_figures.py)

本文围绕最终候选“E00量价选股＋RISK-B1跨资产风险预算”，先讲投资问题、规则和结果，再用归因及旧实验解释取舍。旧提交报告保留在上一级，没有覆盖。报告不含队伍、作者或学校信息。

## 不依赖供应商数据的图表再现

在仓库根目录执行：

```powershell
python -m pip install -r docs/strategy-report/requirements.txt
python docs/strategy-report/build_figures.py
```

会生成四幅PDF/PNG图、月度收益表及`figure_checks.json`。这些是**已有策略结果的再分析**，不重新训练，不运行交易账户。

`figure_data/`仅有归一化账户净值、实际组合占比和初始资金贡献；无原始股票/ETF报价或新闻。`figure_sources.json`记录本地原结果路径和SHA，外部读者不必拥有这些本地文件即可重画图，但无法据此重新生成上游策略预测。

已用Windows、Python 3.12.7验证。制图中文字体优先Microsoft YaHei；其他系统请在`build_figures.py`改为已有的CJK字体，避免方框字。

## 编译PDF

预装XeLaTeX及常见宏包：fontspec、geometry、amsmath、amssymb、booktabs、array、longtable、graphicx、xcolor、caption、hyperref、fancyhdr。当前正文使用Windows宋体和微软雅黑。Linux/macOS需自行将`main.tex`字体改为已安装的CJK字体；未验证其他系统。

```powershell
python docs/strategy-report/build_report.py
```

脚本重画图、编译两遍并检查缺字、未定义引用及内容溢出，输出`report.pdf`，中间文件在被忽略的`build/`。无需shell-escape、外部文献数据库、GPU或数据终端。若MiKTeX用户级/系统级版本不一致，避免直接全量更新；当前版本已在现有系统成功编译，并使用原生XeTeX中文断行以规避xeCJK版本冲突。

## 指标及证据边界

- 历史：2024-01-02—2025-12-31，485日连续账户。
- 补充：2026-01-05—09-21，175日现金重启，不能拼成跨年净值或称全年/盲测。
- 夏普参考现金0、日收益样本标准差、252日约定。
- 股票及未分拆贡献为差额，不能称选股alpha。
- 压力情景平均现金跳升的机制仍待进一步审计，报告未将其解释为已证实的摩擦稳健性。
- 本次只修订报告与展示；未重新训练、下载数据或修改策略参数。

## 2026-11-05发布修订

更正附录N01/N02/N03的恢复状态，与10月14日复现证据保持一致；更新日期。主策略参数、净值和结果不变。原10月31日报告保留于本地档案，原正式提交PDF/ZIP保持原SHA。新增账户核查见[发布核查](../发布核查_20261105.md)。

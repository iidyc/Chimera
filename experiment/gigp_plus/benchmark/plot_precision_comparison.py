"""Keep paper-default and matched-precision series explicit in one comparison."""
import argparse
import csv
from pathlib import Path

from collect_results import pareto


def comparison_points(matched, paper):
    preferred = {}
    for rows, bits in [(matched, 4), (paper, 2)]:
        for row in rows:
            if row['three_runs_verified'].lower() != 'true':
                continue
            if int(row['bits']) != bits or int(row['concurrent_queries']) != 3:
                raise ValueError('Unexpected precision or concurrency in comparison input')
            key = tuple(row[k] for k in ['system', 'bits', 'gpu', 'dataset', 'config', 'query_sha256'])
            if key not in preferred or (row['mode'] == 'full' and preferred[key]['mode'] != 'full'):
                preferred[key] = dict(row, recall_at_100=float(row['recall_at_100']), qps=float(row['qps']))
    rows = list(preferred.values())
    for gpu in ['a100', 'v100']:
        for dataset in ['lotte', 'hotpot', 'msmarco']:
            hashes = {r['query_sha256'] for r in rows if r['gpu'] == gpu and r['dataset'] == dataset}
            if len(hashes) > 1:
                raise ValueError(f'Query cohort mismatch: {gpu}/{dataset}')
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--matched', type=Path, required=True)
    p.add_argument('--paper', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    with a.matched.open() as f:
        matched = list(csv.DictReader(f))
    with a.paper.open() as f:
        paper = list(csv.DictReader(f))
    rows = comparison_points(matched, paper)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 9, 'pdf.fonttype': 42})
    fig, axes = plt.subplots(2, 3, figsize=(11.4, 6.3), squeeze=False)
    styles = [('Chimera', '4', '#c23b3b', 'o', '-', 'Chimera (4-bit)'),
              ('GIGP+', '2', '#208a82', 's', '-', 'GIGP+ (2-bit, paper precision)'),
              ('GIGP+', '4', '#60748e', '^', '--', 'GIGP+ (4-bit, matched precision)')]
    labels = {'lotte': 'LoTTE', 'hotpot': 'HotpotQA', 'msmarco': 'MS MARCO'}
    plotted = []
    for i, gpu in enumerate(['a100', 'v100']):
        for j, dataset in enumerate(labels):
            ax = axes[i, j]
            subset = [r for r in rows if r['gpu'] == gpu and r['dataset'] == dataset]
            for system, bits, color, marker, line, label in styles:
                points = sorted(pareto([r for r in subset if r['system'] == system and r['bits'] == bits]),
                                key=lambda r: r['recall_at_100'])
                plotted.extend(points)
                if points:
                    ax.errorbar([r['recall_at_100'] for r in points], [r['qps'] for r in points],
                                yerr=[float(r['qps_std'] or 0) for r in points],
                                color=color, marker=marker, linestyle=line, markersize=3.5,
                                linewidth=1.1, capsize=2, label=label)
            if not subset:
                ax.text(.5, .5, 'Awaiting measurements', ha='center', va='center', transform=ax.transAxes)
            elif not any(r['bits'] == '2' for r in subset):
                ax.text(.03, .05, '2-bit results pending', color='#555555', va='bottom', transform=ax.transAxes)
            elif all(r['mode'] == 'pilot' for r in subset if r['bits'] == '2'):
                ax.text(.03, .05, '2-bit pilot', color='#555555', va='bottom', transform=ax.transAxes)
            ax.set_title(f'{labels[dataset]} / {gpu.upper()}')
            ax.set_xlim(0, 1)
            ax.set_ylim(bottom=0)
            ax.set_xlabel('Recall@100')
            ax.set_ylabel('Throughput (QPS)')
            ax.grid(alpha=.2)
    handles, legend_labels = [], []
    for ax in axes.flat:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            if label not in legend_labels:
                handles.append(handle)
                legend_labels.append(label)
    fig.legend(handles, legend_labels, loc='upper center', ncol=3, frameon=False)
    coverage = {(r['system'], r['bits'], r['gpu'], r['dataset']) for r in plotted}
    partial = len(coverage) < 18 or any(r['mode'] != 'full' for r in plotted)
    fig.text(.5, .035, ('Partial coverage. ' if partial else '') +
             'Three concurrent queries; only three-repeat points shown; QPS error bars are standard deviations.',
             ha='center', fontsize=8)
    fig.text(.5, .012, 'GIGP+ preloads queries outside its timer; Chimera includes query transfers. Precisions differ as labeled.',
             ha='center', fontsize=8)
    fig.tight_layout(rect=(0, .065, 1, .925))
    a.output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ['.png', '.pdf']:
        fig.savefig(a.output.with_suffix(suffix), dpi=200)
    plt.close(fig)


if __name__ == '__main__':
    main()

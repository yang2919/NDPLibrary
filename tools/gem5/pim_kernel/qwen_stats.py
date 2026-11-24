"""Summarize the two explicit SE statistics dumps and exclude process teardown."""
import argparse
import json
from pathlib import Path
import re


def summarize(run, model, context):
    sim = run / 'sim'
    marker = f'QWEN_ROI_DONE {model} context={context} checksum='
    if not any(line.startswith(marker) for line in (sim / 'program.out').read_text().splitlines()):
        raise ValueError('missing completed layer marker')
    blocks = re.findall(r'Begin Simulation Statistics -+\n(.*?)\n-+ End Simulation Statistics',
                        (sim / 'stats.txt').read_text(), re.S)
    if len(blocks) != 3:
        raise ValueError(f'expected attention, FFN, and exit dumps; got {len(blocks)}')
    report = {'model': model, 'context_including_current_token': context, 'batch': 1, 'layers': 1,
              'operand': 'fp16', 'accumulator': 'fp32', 'cpu': 'timing', 'phases': {}}
    for name, block in zip(('attention', 'ffn'), blocks):
        stats = dict(re.findall(r'^([^\s#]+)\s+([\d.eE+-]+)\s', block, re.M))
        phase = {k: int(stats[k]) for k in ('simTicks', 'simInsts', 'simFreq', 'finalTick')}
        phase['cpuCycles'] = int(stats['board.processor.cores.core.numCycles'])
        phase['pim_requests'] = {k.split('::')[-1]: int(v) for k, v in stats.items()
                                 if '.pim.reqs::' in k}
        if phase['simTicks'] <= 0 or phase['pim_requests'].get('AbMac', 0) <= 0:
            raise ValueError(f'empty {name} ROI')
        report['phases'][name] = phase
    attention, ffn = report['phases'].values()
    if ffn['finalTick'] - attention['finalTick'] != ffn['simTicks']:
        raise ValueError('FFN stats do not begin at the attention dump')
    report['layer_ticks'] = attention['simTicks'] + ffn['simTicks']
    report['layer_simulated_seconds'] = report['layer_ticks'] / attention['simFreq']
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('model', choices=('moe', 'dense'))
    parser.add_argument('context', type=int)
    args = parser.parse_args()
    report = summarize(args.run, args.model, args.context)
    output = json.dumps(report, indent=2)
    (args.run / 'roi.json').write_text(output + '\n')
    print(output)

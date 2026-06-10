from pathlib import Path
import re

base = Path('/mnt/shared-storage-gpfs2/plm-gpfs/ylliu/scale_up/train_conceptlm_V22_olmo_7B/align16_0609172152')
prev = next(base.glob('prev/run/*/results/launch_manifest_*.txt'))
moved = next(base.glob('moved/run/*/results/launch_manifest_*.txt'))
out = base / 'manifest_alignment_report.txt'

def parse(p):
    d = {}
    for line in p.read_text().splitlines():
        if '=' in line:
            k, v = line.split('=', 1)
            d[k] = v
    return d

pa, ma = parse(prev), parse(moved)
ignore = {'date', 'repo_root', 'run_dir', 'wandb_exp_name', 'wandb_save_dir', 'checkpoint_dir', 'hidden_rank_log_path'}

def norm(s):
    s = s.replace(str(base / 'prev'), '<RUN_BASE>')
    s = s.replace(str(base / 'moved'), '<RUN_BASE>')
    s = s.replace('/mnt/shared-storage-gpfs2/plm-gpfs/jyhuang/works/megatron_ConceptlmV2', '<REPO>')
    s = s.replace('/mnt/shared-storage-gpfs2/plm-gpfs/ylliu/scale_up/train_conceptlm_V22_olmo_7B/repo', '<REPO>')
    s = re.sub(r'align16-prev-c22vq-muon-s8-lr6e5-0609172152-26879', '<META>', s)
    s = re.sub(r'align16-moved-c22vq-muon-s8-lr6e5-0609172152-4445e', '<META>', s)
    s = re.sub(r'align16-prev-c22vq-muon-s8-lr6e5-0609172152', '<JOB>', s)
    s = re.sub(r'align16-moved-c22vq-muon-s8-lr6e5-0609172152', '<JOB>', s)
    s = re.sub(r'10\.102\.\d+\.\d+', '<MASTER_ADDR>', s)
    return s

raw_diffs = []
norm_diffs = []
for k in sorted(set(pa) | set(ma)):
    pv, mv = pa.get(k, ''), ma.get(k, '')
    if pv != mv:
        raw_diffs.append(k)
    if k not in ignore and norm(pv) != norm(mv):
        norm_diffs.append((k, norm(pv), norm(mv)))

key_fields = [
    'world_size', 'gpus_per_node', 'node_count', 'precision', 'transformer_impl',
    'layer_spec', 'olmo3_qk_norm_impl', 'attention_backend', 'init_method_variant',
    'init_method_std', 'optimizer', 'olmo3_z_loss_multiplier',
    'olmo3_embedding_weight_decay_zero', 'grad_reduce_in_bf16', 'save_full_state',
    'tp pp cp mbs gbs seq train_iters', 'train_entrypoint', 'save_interval',
    'train_args', 'parallel_args', 'model_args', 'data_args', 'checkpoint_args'
]

lines = []
lines.append(f'prev_manifest={prev}')
lines.append(f'moved_manifest={moved}')
lines.append('')
lines.append('raw_diff_keys=' + ','.join(raw_diffs))
lines.append('')
lines.append('normalized_diffs=')
if not norm_diffs:
    lines.append('NONE')
else:
    for k, pv, mv in norm_diffs:
        lines.append(f'[{k}]')
        lines.append(f'prev={pv}')
        lines.append(f'moved={mv}')
lines.append('')
lines.append('key_fields=')
for k in key_fields:
    lines.append(f'{k}: prev={pa.get(k)} | moved={ma.get(k)}')
out.write_text('\n'.join(lines) + '\n')
print(out)

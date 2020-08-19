import subprocess
import shutil
from tqdm import tqdm
import torch
import os
import argparse
import sys
from pathlib import Path
from cosypose.config import PROJECT_DIR, RESULTS_DIR


TOOLKIT_DIR = Path(PROJECT_DIR / 'deps' / 'bop_toolkit_challenge')
EVAL_SCRIPT_PATH = TOOLKIT_DIR / 'scripts/eval_bop19.py'
DUMMY_EVAL_SCRIPT_PATH = TOOLKIT_DIR / 'scripts/eval_bop19_dummy.py'

sys.path.append(TOOLKIT_DIR.as_posix())
from bop_toolkit_lib import inout  # noqa
# from bop_toolkit_lib.config import results_path as BOP_RESULTS_PATH  # noqa


def main():
    parser = argparse.ArgumentParser('Bop evaluation')
    parser.add_argument('--result_id', default='', type=str)
    parser.add_argument('--method', default='', type=str)
    parser.add_argument('--dataset', default='', type=str)
    parser.add_argument('--split', default='test', type=str)
    parser.add_argument('--csv_path', default='', type=str)
    parser.add_argument('--dummy', action='store_true')
    parser.add_argument('--convert_only', action='store_true')
    args = parser.parse_args()
    run_evaluation(args)


def run_evaluation(args):
    results_path = RESULTS_DIR / args.result_id / f'dataset={args.dataset}' / 'results.pth.tar'
    csv_path = args.csv_path
    convert_results(results_path, csv_path, method=args.method)

    if not args.dummy:
        shutil.copy(csv_path, RESULTS_DIR / args.result_id / f'dataset={args.dataset}' / csv_path.name)

    if not args.convert_only:
        run_bop_evaluation(csv_path, dummy=args.dummy)
    return csv_path


def convert_results(results_path, out_csv_path, method):
    predictions = torch.load(results_path)['predictions']
    predictions = predictions[method]
    print("Predictions from:", results_path)
    print("Method:", method)
    print("Number of predictions: ", len(predictions))

    preds = []
    for n in tqdm(range(len(predictions))):
        TCO_n = predictions.poses[n]
        t = TCO_n[:3, -1] * 1e3  # m -> mm conversion
        R = TCO_n[:3, :3]
        row = predictions.infos.iloc[n]
        obj_id = int(row.label.split('_')[-1])
        score = row.score
        time = row.time
        pred = dict(scene_id=row.scene_id,
                    im_id=row.view_id,
                    obj_id=obj_id,
                    score=score,
                    t=t, R=R, time=time)
        preds.append(pred)
    print("Wrote:", out_csv_path)
    inout.save_bop_results(out_csv_path, preds)
    return out_csv_path


def run_bop_evaluation(filename, dummy=False):
    myenv = os.environ.copy()
    myenv['PYTHONPATH'] = TOOLKIT_DIR.as_posix()
    myenv['COSYPOSE_DIR'] = PROJECT_DIR.as_posix()
    if dummy:
        script_path = DUMMY_EVAL_SCRIPT_PATH
    else:
        script_path = EVAL_SCRIPT_PATH
    subprocess.call(['python', script_path.as_posix(),
                     '--renderer_type', 'python',
                     '--result_filenames', filename],
                    env=myenv, cwd=TOOLKIT_DIR.as_posix())


if __name__ == '__main__':
    main()

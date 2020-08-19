import subprocess
from tqdm import tqdm
import torch
import numpy as np
import os
import argparse
import sys
from cosypose.config import PROJECT_DIR, LOCAL_DATA_DIR, RESULTS_DIR, MEMORY, BOP_TOOLKIT_DIR

SISO_SCRIPT_PATH = BOP_TOOLKIT_DIR / 'scripts/eval_siso.py'
VIVO_SCRIPT_PATH = BOP_TOOLKIT_DIR / 'scripts/eval_vivo.py'

sys.path.append(BOP_TOOLKIT_DIR.as_posix())
from bop_toolkit_lib import inout  # noqa


def main():
    parser = argparse.ArgumentParser('Bop evaluation')
    parser.add_argument('--result_id', default='', type=str)
    parser.add_argument('--method', default='', type=str)
    parser.add_argument('--vivo', action='store_true')
    args = parser.parse_args()
    n_rand = np.random.randint(1e6)
    csv_path = LOCAL_DATA_DIR / 'bop_predictions_csv' / f'cosypose{n_rand}-eccv2020_tless-test-primesense.csv'
    csv_path.parent.mkdir(exist_ok=True)
    results_path = RESULTS_DIR / args.result_id / 'results.pth.tar'
    convert_results(results_path, csv_path, method=args.method)
    run_evaluation(csv_path, args.vivo)


@MEMORY.cache
def convert_results(results_path, out_csv_path, method):
    predictions = torch.load(results_path)['predictions'][method]
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
        time = -1.0
        pred = dict(scene_id=row.scene_id,
                    im_id=row.view_id,
                    obj_id=obj_id,
                    score=score,
                    t=t, R=R, time=time)
        preds.append(pred)
    print("Wrote:", out_csv_path)
    inout.save_bop_results(out_csv_path, preds)
    return out_csv_path


def run_evaluation(filename, is_vivo):
    if is_vivo:
        script_path = VIVO_SCRIPT_PATH
    else:
        script_path = SISO_SCRIPT_PATH
    myenv = os.environ.copy()
    myenv['PYTHONPATH'] = BOP_TOOLKIT_DIR.as_posix()
    myenv['COSYPOSE_DIR'] = PROJECT_DIR.as_posix()
    print(script_path)
    subprocess.call(['python', script_path.as_posix(),
                     '--renderer_type', 'python',
                     '--result_filename', filename],
                    env=myenv, cwd=BOP_TOOLKIT_DIR.as_posix())


if __name__ == '__main__':
    main()

import argparse
import multiprocessing
from copy import deepcopy
from cosypose.config import RESULTS_DIR
from cosypose.bop_config import BOP_CONFIG
from cosypose.scripts.run_bop20_eval import run_evaluation
from cosypose.config import LOCAL_DATA_DIR


def main():
    parser = argparse.ArgumentParser('Bop multi evaluation')
    parser.add_argument('--result_id', default='', type=str)
    parser.add_argument('--method', default='maskrcnn_detections/refiner/iteration=4', type=str)
    parser.add_argument('--dummy', action='store_true')
    parser.add_argument('--convert_only', action='store_true')
    args = parser.parse_args()

    result_dir = RESULTS_DIR / args.result_id
    result_ds_dirs = list(result_dir.iterdir())

    processes = dict()
    for result_ds_dir in result_ds_dirs:
        this_cfg = deepcopy(args)
        ds_name = str(result_ds_dir).split('=')[-1]
        has_test_set = len(BOP_CONFIG[ds_name]['test_ds_name']) > 0
        dummy = args.dummy or not has_test_set
        convert_only = args.convert_only or not has_test_set
        this_cfg.dummy = dummy
        this_cfg.convert_only = convert_only
        this_cfg.dataset = ds_name
        this_cfg.split = 'test'

        start_str = 'challenge2020dummy' if args.dummy else 'challenge2020'
        result_id_int = args.result_id.split('-')[-1]
        csv_path = LOCAL_DATA_DIR / 'bop_predictions_csv'
        split = 'test'
        csv_path = csv_path / f'{start_str}-{result_id_int}_{ds_name}-{split}.csv'
        csv_path.parent.mkdir(exist_ok=True)
        this_cfg.csv_path = csv_path

        proc = multiprocessing.Process(target=run_evaluation,
                                       kwargs=dict(args=this_cfg))
        proc.start()
        processes[ds_name] = (this_cfg, proc)

    for ds_name, (_, proc) in processes.items():
        proc.join()

    for _ in range(10):
        print(f"{'-'*80}")

    for ds_name, (cfg, _) in processes.items():
        results_dir = LOCAL_DATA_DIR / 'bop_eval_outputs' / cfg.csv_path.with_suffix('').name
        scores_path = results_dir / 'scores_bop19.json'
        print(f"{'-'*80}")
        print(f'{ds_name}: {scores_path}')
        print(scores_path.read_text())


if __name__ == '__main__':
    main()

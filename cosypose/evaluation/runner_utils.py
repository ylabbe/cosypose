from collections import OrderedDict
import pandas as pd
from collections import defaultdict

from cosypose.utils.distributed import get_tmp_dir, get_rank
from cosypose.utils.logging import get_logger

logger = get_logger(__name__)


def run_pred_eval(pred_runner, pred_kwargs, eval_runner, eval_preds=None):
    all_predictions = dict()
    for pred_prefix, pred_kwargs_n in pred_kwargs.items():
        print("Prediction :", pred_prefix)
        preds = pred_runner.get_predictions(**pred_kwargs_n)
        for preds_name, preds_n in preds.items():
            all_predictions[f'{pred_prefix}/{preds_name}'] = preds_n

    all_predictions = OrderedDict({k: v for k, v in sorted(all_predictions.items(), key=lambda item: item[0])})
    eval_metrics, eval_dfs = dict(), dict()

    for preds_k, preds in all_predictions.items():
        print("Evaluation :", preds_k)
        if eval_preds is None or preds_k in eval_preds:
            eval_metrics[preds_k], eval_dfs[preds_k] = eval_runner.evaluate(preds)

    all_predictions = gather_predictions(all_predictions)

    if get_rank() == 0:
        results = format_results(all_predictions,
                                 eval_metrics,
                                 eval_dfs)
    else:
        results = None
    return results


def gather_predictions(all_predictions):
    for k, v in all_predictions.items():
        all_predictions[k] = v.gather_distributed(tmp_dir=get_tmp_dir()).cpu()
    return all_predictions


def format_results(predictions,
                   eval_metrics,
                   eval_dfs,
                   print_metrics=True):
    summary = dict()
    df = defaultdict(list)
    summary_txt = ''
    for k, v in eval_metrics.items():
        summary_txt += f"\n{k}\n{'-'*80}\n"
        for k_, v_ in v.items():
            summary[f'{k}/{k_}'] = v_
            df['method'].append(k)
            df['metric'].append(k_)
            df['value'].append(v_)
            summary_txt += f'{k}/{k_}: {v_}\n'
        summary_txt += f"{'-'*80}"
    if print_metrics:
        logger.info(summary_txt)

    df = pd.DataFrame(df)
    results = dict(
        summary=summary,
        summary_txt=summary_txt,
        predictions=predictions,
        metrics=eval_metrics,
        summary_df=df,
        dfs=eval_dfs,
    )
    return results

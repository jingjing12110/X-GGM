import os
import argparse

from eval.gqa_ood.gqa_eval import GQAEval
from eval.gqa_ood.plot_tail import plot_tail

from utils import write_txt

# python evaluation.py --ood_test
# --predictions [prediction path (on ood_testdev_all or gqa_testdev)]
# python evaluation.py --eval_tail_size
# --predictions [prediction path (on ood_val_all or gqa_val)]
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--eval_tail_size',
        default=True,
        # action='store_true'
    )
    parser.add_argument(
        '--save_dir',
        default='snap/gqa_ood/0323_GGM_5')
    parser.add_argument(
        '--ood_test', default=True, type=bool)
    parser.add_argument(
        '--predictions', type=str,
        default='snap/gqa_ood/0323_GGM_5/val_all_predict.json')
    args = parser.parse_args()

    if args.eval_tail_size:
        result_eval_file = args.predictions
        # Retrieve scores
        alpha_list = [9.0, 7.0, 5.0, 3.6, 2.8, 2.2, 1.8, 1.4, 1.0, 0.8, 0.4, 0.3,
                      0.2, 0.1, 0.0, -0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7]
        acc_list = []
        for alpha in alpha_list:
            ques_file_path = \
                f'data/gqa_ood/alpha_tail/val_bal_tail_{alpha:.1f}.json'
            gqa_eval = GQAEval(result_eval_file,
                               ques_file_path,
                               choices_path=None,
                               EVAL_CONSISTENCY=False)
            acc = gqa_eval.get_acc_result()['accuracy']
            acc_list.append(acc)
        
        print("Alpha:", alpha_list)
        print("Accuracy:", acc_list)
        # Plot: save to "tail_plot_[model_name].pdf"
        plot_tail(alpha=list(map(lambda x: x + 1, alpha_list)), accuracy=acc_list,
                  model_name='default')  # We plot 1+alpha vs. accuracy
    if args.ood_test:
        result_eval_file = args.predictions
        file_list = {'Tail': 'ood_testdev_tail.json',
                     'Head': 'ood_testdev_head.json',
                     'All': 'ood_testdev_all.json'}
        result = {}
        for setup, ques_file_path in file_list.items():
            gqa_eval = GQAEval(result_eval_file,
                               'data/gqa_ood/org/' + ques_file_path,
                               choices_path=None,
                               EVAL_CONSISTENCY=False)
            result[setup] = gqa_eval.get_acc_result()['accuracy']
            
            result_string, detail_result_string = gqa_eval.get_str_result()
            print('\n___%s___' % setup)
            for result_string_ in result_string:
                print(result_string_)
        
        print('\nRESULTS:\n')
        delta = (result['Head'] - result['Tail']) / result['Tail'] * 100.
        msg = f"Accuracy (all, tail, head, delta):" \
              f" {result['All']:.2f}, {result['Tail']:.2f}, " \
              f"{result['Head']:.2f}, {delta:.2f}\n"
        print(msg)
        write_txt(os.path.join(args.save_dir, f'result.txt'), msg)

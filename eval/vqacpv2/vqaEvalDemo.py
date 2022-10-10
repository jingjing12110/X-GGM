import os
import argparse
import json
import random

from eval.vqacpv2.vqaEval import VQAEval
from eval.vqacpv2.vqa import VQA


def main(args):
    # set up file names and paths
    versionType = 'cpv2_'  # this should be '' when using VQA v2.0 dataset
    # 'OpenEnded' only for v2.0. 'OpenEnded' or 'MultipleChoice' for v1.0
    taskType = 'OpenEnded'
    # 'mscoco' only for v1.0. 'mscoco' for real and 'abstract_v002'
    # for abstract for v1.0.
    dataType = 'mscoco'
    
    if args.tmode == 'OOD':
        annFile = 'data/vqacpv2/raw_anns/vqacp_v2_testsplit_annotations.json'
        quesFile = 'data/vqacpv2/raw_anns/vqacp_v2_testsplit_questions.json'
        dataSubType = 'TestSplit'
    elif args.tmode == 'ID':
        annFile = 'data/vqacpv2/raw_anns/vqacp_v2_valsplit_annotations.json'
        quesFile = 'data/vqacpv2/raw_anns/vqacp_v2_valsplit_questions.json'
        dataSubType = 'ValSplit'
    else:
        annFile = '../data/v2_mscoco_val2014_annotations.json'
        quesFile = '../data/v2_OpenEnded_mscoco_val2014_questions.json'
    resultType = 'fake'
    fileTypes = ['accuracy', 'evalQA', 'evalQuesType', 'evalAnsType']
    
    # An example result json file has been provided in './Results' folder.
    [accuracyFile, evalQAFile, evalQuesTypeFile, evalAnsTypeFile] = [
        '%s%s_%s_%s_%s_%s.json' % (versionType, taskType, dataType, dataSubType,
                                   resultType, fileType) for fileType in
        fileTypes]

    # create vqa object and vqaRes object
    vqa = VQA(annFile, quesFile)
    vqaRes = vqa.loadRes(args.resfile, quesFile)
    
    # create vqaEval object by taking vqa and vqaRes
    # n is precision of accuracy (number of places after decimal), default is 2
    vqaEval = VQAEval(vqa, vqaRes, n=2)
    
    # evaluate results
    """
    If you have a list of question ids on which you would like to evaluate your
    results, pass it as a list to below function
    By default it uses all the question ids in annotation file
    """
    vqaEval.evaluate()
    
    # print accuracies
    print("Overall Accuracy is: %.02f\n" % (vqaEval.accuracy['overall']))
    print("Per Answer Type Accuracy is the following:")
    for ansType in vqaEval.accuracy['perAnswerType']:
        print(
            "%s : %.02f" % (ansType, vqaEval.accuracy['perAnswerType'][ansType]))
    
    # print("\nPer Question Type Accuracy is the following:")
    # for quesType in vqaEval.accuracy['perQuestionType']:
    #     print("%s : %.02f" % (
    #         quesType, vqaEval.accuracy['perQuestionType'][quesType]))
    
    # demo how to use evalQA to retrieve low score result
    # 35 is per question percentage accuracy
    evals = [quesId for quesId in vqaEval.evalQA if vqaEval.evalQA[quesId] < 35]
    if len(evals) > 0:
        print('\nground truth answers')
        randomEval = random.choice(evals)
        randomAnn = vqa.loadQA(randomEval)
        vqa.showQA(randomAnn)
        
        print(
            '\n generated answer (accuracy %.02f)' % (vqaEval.evalQA[randomEval]))
        ann = vqaRes.loadQA(randomEval)[0]
        print("Answer:   %s\n" % (ann['answer']))
    
    # save evaluation results to ./Results folder
    results = vqaEval.accuracy['perAnswerType']
    results['Overall'] = vqaEval.accuracy['overall']
    if args.save_json:
        with open(os.path.join(args.output_dir, accuracyFile), 'a+') as f:
            json.dump(
                results, f, sort_keys=True, indent=4)
            # json.dump(
            #     vqaEval.accuracy['perAnswerType'],
            #     f, sort_keys=True, indent=4)
            # json.dump(
            #     vqaEval.accuracy['perQuestionType'],
            #     f, sort_keys=True, indent=4)
            
        # json.dump(vqaEval.evalQA, open(os.path.join(
        #     args.output_dir, evalQAFile), 'w'))
        # json.dump(vqaEval.evalQuesType, open(os.path.join(
        #     args.output_dir, evalQuesTypeFile), 'w'))
        # json.dump(vqaEval.evalAnsType, open(os.path.join(
        #     args.output_dir, evalAnsTypeFile), 'w'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        "Save a model's predictions for the VQA-CP test set")
    parser.add_argument(
        '--cpv2', default=True, type=bool)
    parser.add_argument(
        '--output_dir', type=str,
        # default='snap/vqacpv2_2/0313_N-GGM-LD',
        # default='snap/ablation/0128_vqacpv2',
        # default='snap/UpDn/0330_GGM',
        default='/media/kaka/SX500/code/lvc-vqa/snap/lxmert-CIB-vqacpv2/0713_lxmert_pretrained_Epoch20LXRT_CIB_beta1e3_epoch10_lr4e5_bs96',
    )
    parser.add_argument(
        '--save_json', default=True, type=bool)
    parser.add_argument(
        "--tmode", default='OOD', type=str, help="['OOD', 'ID']")
    parser.add_argument(
        '--resfile', type=str,
        default=None)
    
    args = parser.parse_args()
    
    # testing OOD
    args.resfile = os.path.join(args.output_dir, f'{args.tmode}_predict.json')
    main(args)
    
    # testing ID
    # args.tmode = 'ID'
    # args.resfile = os.path.join(args.output_dir, f'{args.tmode}_predict.json')
    # main(args)

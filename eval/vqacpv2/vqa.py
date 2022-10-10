__author__ = 'aagrawal'
__version__ = '0.9'

# The following functions are defined:
#  VQA   - VQA class that loads VQA annotation file and prepares data structures.
#  getQuesIds - Get question ids that satisfy given filter conditions.
#  getImgIds  - Get image ids that satisfy given filter conditions.
#  loadQA     - Load questions and answers with the specified question ids.
#  showQA     - Display the specified questions and answers.
#  loadRes    - Load result file and create result object.

# Help on each function can be accessed by: "help(COCO.function)"

import json
import datetime


class VQA:
    def __init__(self, annotation_file=None, question_file=None):
        """
         Constructor of VQA helper class for reading and visualizing
         questions and answers.
        :param annotation_file (str): location of VQA annotation file
        :return:
        """
        # load dataset
        self.dataset = {}
        self.questions = {}
        self.qa = {}
        self.qqa = {}
        self.imgToQA = {}
        if not annotation_file == None and not question_file == None:
            print('loading VQA annotations and questions into memory...')
            time_t = datetime.datetime.utcnow()
            print(annotation_file)
            dataset = json.load(open(annotation_file, 'r'))
            questions = json.load(open(question_file, 'r'))
            print(datetime.datetime.utcnow() - time_t)
            
            self.dataset = dataset
            self.questions = questions
            self.createIndex()
    
    def createIndex(self):
        # create index
        print('creating index...')
        imgToQA = {ann['image_id']: [] for ann in self.dataset}
        qa = {ann['question_id']: [] for ann in self.dataset}
        qqa = {ann['question_id']: [] for ann in self.dataset}
        for ann in self.dataset:
            imgToQA[ann['image_id']] += [ann]
            qa[ann['question_id']] = ann
        for ques in self.questions:
            qqa[ques['question_id']] = ques
        print('index created!\n')
        
        # create class members
        self.qa = qa
        self.qqa = qqa
        self.imgToQA = imgToQA
    
    def getQuesIds(self, imgIds=[], quesTypes=[], ansTypes=[]):
        """
        Get question ids that satisfy given filter conditions. default skips that filter
        :param 	imgIds    (int array)   : get question ids for given imgs
                quesTypes (str array)   : get question ids for given question types
                ansTypes  (str array)   : get question ids for given answer types
        :return:    ids   (int array)   : integer array of question ids
        """
        imgIds = imgIds if type(imgIds) == list else [imgIds]
        quesTypes = quesTypes if type(quesTypes) == list else [quesTypes]
        ansTypes = ansTypes if type(ansTypes) == list else [ansTypes]
        
        if len(imgIds) == len(quesTypes) == len(ansTypes) == 0:
            anns = self.dataset
        else:
            if not len(imgIds) == 0:
                anns = sum([self.imgToQA[imgId] for imgId in imgIds if
                            imgId in self.imgToQA], [])
            else:
                anns = self.dataset
            anns = anns if len(quesTypes) == 0 else [ann for ann in anns if ann[
                'question_type'] in quesTypes]
            anns = anns if len(ansTypes) == 0 else [ann for ann in anns if ann[
                'answer_type'] in ansTypes]
        ids = [ann['question_id'] for ann in anns]
        return ids
    
    def getImgIds(self, quesIds=[], quesTypes=[], ansTypes=[]):
        """
        Get image ids that satisfy given filter conditions. default skips that filter
        :param quesIds   (int array)   : get image ids for given question ids
               quesTypes (str array)   : get image ids for given question types
               ansTypes  (str array)   : get image ids for given answer types
        :return: ids     (int array)   : integer array of image ids
        """
        quesIds = quesIds if type(quesIds) == list else [quesIds]
        quesTypes = quesTypes if type(quesTypes) == list else [quesTypes]
        ansTypes = ansTypes if type(ansTypes) == list else [ansTypes]
        
        if len(quesIds) == len(quesTypes) == len(ansTypes) == 0:
            anns = self.dataset
        else:
            if not len(quesIds) == 0:
                anns = sum(
                    [self.qa[quesId] for quesId in quesIds if quesId in self.qa],
                    [])
            else:
                anns = self.dataset
            anns = anns if len(quesTypes) == 0 else [ann for ann in anns if ann[
                'question_type'] in quesTypes]
            anns = anns if len(ansTypes) == 0 else [ann for ann in anns if ann[
                'answer_type'] in ansTypes]
        ids = [ann['image_id'] for ann in anns]
        return ids
    
    def loadQA(self, ids=[]):
        """
        Load questions and answers with the specified question ids.
        :param ids (int array)       : integer ids specifying question ids
        :return: qa (object array)   : loaded qa objects
        """
        if type(ids) == list:
            return [self.qa[id] for id in ids]
        elif type(ids) == int:
            return [self.qa[ids]]
    
    def showQA(self, anns):
        """
        Display the specified annotations.
        :param anns (array of object): annotations to display
        :return: None
        """
        if len(anns) == 0:
            return 0
        for ann in anns:
            quesId = ann['question_id']
            print("Question: %s" % (self.qqa[quesId]['question']))
            
            for ans in ann['answers']:
                print("Answer %d: %s" % (ans['answer_id'], ans['answer']))
    
    def loadRes(self, resFile, quesFile):
        """
        Load result file and return a result object.
        :param   resFile (str)     : file name of result file
        :return: res (obj)         : result api object
        """
        res = VQA()
        res.questions = json.load(open(quesFile))
        
        print('Loading and preparing results...     ')
        time_t = datetime.datetime.utcnow()
        anns = json.load(open(resFile))
        assert type(anns) == list, 'results is not an array of objects'
        annsQuesIds = [ann['question_id'] for ann in anns]
        assert set(annsQuesIds) == set(self.getQuesIds()), \
            'Results do not correspond to current VQA set. ' \
            'Either the results do not have predictions for all ' \
            'question ids in annotation file or there is atleast one' \
            ' question id that does not belong to the question ids ' \
            'in the annotation file.'
        for ann in anns:
            quesId = ann['question_id']
            qaAnn = self.qa[quesId]
            ann['image_id'] = qaAnn['image_id']
            ann['question_type'] = qaAnn['question_type']
            ann['answer_type'] = qaAnn['answer_type']
        print('DONE (t=%0.2fs)\n' % (
            (datetime.datetime.utcnow() - time_t).total_seconds()))
        
        res.dataset = anns
        res.createIndex()
        return res

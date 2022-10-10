# @File  :data_loader.py
# @Time  :2020/12/31
# @Desc  :
import torch
from prefetch_generator import BackgroundGenerator


class DataLoaderX(torch.utils.data.DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())
    

class DataPrefetcher(object):
    def __init__(self, loader):
        self.loader = iter(loader)
        
        self.preload()
    
    def preload(self):
        try:
            self.batch = next(self.loader)
        except StopIteration:
            self.batch = None
            return
    
    def next(self):
        batch = self.batch
        self.preload()
        return batch


class _RepeatSampler(object):
    """ 一直repeat的sampler """
    
    def __init__(self, sampler):
        self.sampler = sampler
    
    def __iter__(self):
        while True:
            yield from iter(self.sampler)


class MultiEpochsDataLoader(torch.utils.data.DataLoader):
    """ 多epoch训练时，DataLoader对象不用重新建立线程和batch_sampler对象，
    以节约每个epoch的初始化时间 """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, 'batch_sampler',
                           _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()
    
    def __len__(self):
        return len(self.batch_sampler.sampler)
    
    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)

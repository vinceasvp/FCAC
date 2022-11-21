import numpy as np
from sklearn.utils import shuffle
import torch
from .sampler import CategoriesSampler, SupportsetSampler, TrueIncreTrainCategoriesSampler

def get_dataloader(args, session):
    if session == 0:
        if args.project == 'STDU':
            trainset, valset, trainloader, valloader = get_base_dataloader_STDU(args)
    else:
        trainset, valset, trainloader, valloader = get_new_dataloader(args, session)
    return trainset, valset, trainloader, valloader

def get_testloader(args, session):
    if args.tmp_train:
        num_base_class = args.STDU.num_tmpb
        num_incre_class = args.STDU.num_tmpi
    else:
        num_base_class = args.num_base
        num_incre_class = 0

    # test on all encountered classes
    class_new = get_session_classes(args, session)

    if 'librispeech' in args.dataset:
        testset = args.Dataset.LBRS(root=args.dataroot, phase="test",
                                      index=class_new, k=None, args=args)
    testloader = torch.utils.data.DataLoader(dataset=testset, batch_size=args.dataloader.test_batch_size, shuffle=False,
                                             num_workers=args.dataloader.num_workers, pin_memory=True)

    return testset, testloader

def get_pretrain_dataloader(args):
    num_base = args.STDU.num_tmpb if args.tmp_train else args.num_base
    class_index = np.arange(num_base)
    if 'librispeech' in args.dataset:
        trainset = args.Dataset.LBRS(root=args.dataroot, phase="train",
                                             index=class_index, base_sess=True, args=args)
        valset = args.Dataset.LBRS(root=args.dataroot, phase="val", index=class_index, base_sess=True, args=args)
    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=args.dataloader.train_batch_size, shuffle=True,
                                              num_workers=8, pin_memory=True)
    valloader = torch.utils.data.DataLoader(
        dataset=valset, batch_size=args.dataloader.test_batch_size, shuffle=False, num_workers=8, pin_memory=True)

    return trainset, valset, trainloader, valloader

def get_base_dataloader_STDU(args):
    if args.tmp_train:
        num_base_class = args.STDU.num_tmpb
        num_incre_class = args.STDU.num_tmpi
    else:
        num_base_class = args.num_base
        num_incre_class = 0

    class_index = np.arange(num_base_class + num_incre_class)

    if 'librispeech' in args.dataset:
        trainset = args.Dataset.LBRS(root=args.dataroot, phase='train', index=class_index, k=None, args=args)
        valset = args.Dataset.LBRS(root=args.dataroot, phase='val', index=class_index, k=None, args=args) # k is same as new_loader's testset k
    # DataLoader(test_set, batch_sampler=sampler, num_workers=8, pin_memory=True)
    train_sampler = TrueIncreTrainCategoriesSampler(label=trainset.targets, n_batch=args.episode.train_episode, 
                                    na_base_cls=num_base_class, na_inc_cls=num_incre_class, 
                                    np_base_cls=args.episode.low_way, np_inc_cls=args.episode.episode_way,
                                    nb_shot=args.episode.low_shot,nn_shot=args.episode.episode_shot, n_query=args.episode.episode_query)
    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_sampler=train_sampler, num_workers=8,
                                                pin_memory=True)

    valloader = torch.utils.data.DataLoader(
        dataset=valset, batch_size=args.dataloader.test_batch_size, shuffle=False, num_workers=8, pin_memory=True)

    return trainset, valset, trainloader, valloader


def get_dataset_for_data_init(args):
    if args.tmp_train:
        num_base_class = args.STDU.num_tmpb
        num_incre_class = args.STDU.num_tmpi
    else:
        num_base_class = args.num_base
        num_incre_class = 0

    class_index = np.arange(num_base_class)
    if 'librispeech' in args.dataset:
        trainset = args.Dataset.LBRS(root=args.dataroot, phase='train', index=class_index, k=None, args=args)
    return trainset
    

def get_new_dataloader(args, session):
    if args.tmp_train:
        num_base_class = args.STDU.num_tmpb
        num_incre_class = args.STDU.num_tmpi
    else:
        num_base_class = args.num_base
        num_incre_class = 0

    assert session > 0
    if 'librispeech' in args.dataset:
        session_classes = np.arange(num_base_class + (session -1) * args.way, num_base_class + session * args.way)
        trainset = args.Dataset.LBRS(root=args.dataroot, phase='train', index=session_classes, k=None, args=args)
    train_sampler = SupportsetSampler(label=trainset.targets, n_cls=args.episode.episode_way, 
                                n_per=args.episode.episode_shot,n_batch=1, seq_sample=args.seq_sample)

    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_sampler=train_sampler, num_workers=8,
                                                pin_memory=True)
                                                
    class_new = get_session_classes(args, session)

    if 'librispeech' in args.dataset:
        valset = args.Dataset.LBRS(root=args.dataroot, phase='val', index=class_new, k=None, args=args)
    valloader = torch.utils.data.DataLoader(dataset=valset, batch_size=args.dataloader.test_batch_size, shuffle=False,
                                                num_workers=8, pin_memory=True)
    return trainset, valset, trainloader, valloader

def get_session_classes(args,  session):
    if args.tmp_train:
        num_base_class = args.sis.num_tmpb
        num_incre_class = args.sis.num_tmpi
    else:
        num_base_class = args.num_base
        num_incre_class = 0

    class_list = np.arange(num_base_class + session * args.way)
    return class_list
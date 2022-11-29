import numpy as np
from sklearn.utils import shuffle
import torch
from .sampler import CategoriesSampler, SupportsetSampler, TrueIncreTrainCategoriesSampler

def get_dataloader(args, session):
    if session == 0:
        if args.project == 'base':
            trainset, valset, trainloader, valloader = get_pretrain_dataloader(args)
        elif args.project == 'stdu':
            trainset, valset, trainloader, valloader = get_base_dataloader_stdu(args)
    else:
        trainset, valset, trainloader, valloader = get_new_dataloader(args, session)
    return trainset, valset, trainloader, valloader

def get_testloader(args, session):
    if args.tmp_train:
        num_base_class = args.stdu.num_tmpb
        num_incre_class = args.stdu.num_tmpi
    else:
        num_base_class = args.num_base
        num_incre_class = 0

    # test on all encountered classes
    class_new = get_session_classes(args, session)

    if args.dataset == 'FMC':
        testset = args.Dataset.FSDCLIPS(root=args.dataroot, phase="test",
                                      index=class_new, k=None)
    elif 'nsynth' in args.dataset:
        testset = args.Dataset.NDS(root=args.dataroot, phase="test",
                                      index=class_new, k=None, args=args)
    elif 'librispeech' in args.dataset:
        testset = args.Dataset.LBRS(root=args.dataroot, phase="test",
                                      index=class_new, k=None, args=args)
    elif args.dataset in ['f2n', 'f2l', 'n2f', 'n2l', 'l2f', 'l2n']:
        testset = args.Dataset.S2S(dataset=args.dataset, root=args.dataroot, phase="test",
                                index=class_new, k=None, args=args)
    testloader = torch.utils.data.DataLoader(dataset=testset, batch_size=args.dataloader.test_batch_size, shuffle=False,
                                             num_workers=args.dataloader.num_workers, pin_memory=True)

    return testset, testloader

def get_pretrain_dataloader(args):
    num_base = args.stdu.num_tmpb if args.tmp_train else args.num_base
    class_index = np.arange(num_base)

    if args.dataset == 'FMC':
        trainset = args.Dataset.FSDCLIPS(root=args.dataroot, phase="train",
                                             index=class_index, base_sess=True)
        valset = args.Dataset.FSDCLIPS(root=args.dataroot, phase="val", index=class_index, base_sess=True)
    elif 'nsynth' in args.dataset:
        trainset = args.Dataset.NDS(root=args.dataroot, phase="train",
                                             index=class_index, base_sess=True, args=args)
        valset = args.Dataset.NDS(root=args.dataroot, phase="val", index=class_index, base_sess=True, args=args)
    elif 'librispeech' in args.dataset:
        trainset = args.Dataset.LBRS(root=args.dataroot, phase="train",
                                             index=class_index, base_sess=True, args=args)
        valset = args.Dataset.LBRS(root=args.dataroot, phase="val", index=class_index, base_sess=True, args=args)
    elif args.dataset in ['f2n', 'f2l', 'n2f', 'n2l', 'l2f', 'l2n']:
        trainset = args.Dataset.S2S(dataset=args.dataset, root=args.dataroot, phase="train",
                                             index=class_index, base_sess=True, args=args)
        valset = args.Dataset.S2S(dataset=args.dataset, root=args.dataroot, phase="val", index=class_index, base_sess=True, args=args)
    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=args.dataloader.train_batch_size, shuffle=True,
                                              num_workers=8, pin_memory=True)
    valloader = torch.utils.data.DataLoader(
        dataset=valset, batch_size=args.dataloader.test_batch_size, shuffle=False, num_workers=8, pin_memory=True)

    return trainset, valset, trainloader, valloader

def get_base_dataloader_stdu(args):
    if args.tmp_train:
        num_base_class = args.stdu.num_tmpb
        num_incre_class = args.stdu.num_tmpi
    else:
        num_base_class = args.num_base
        num_incre_class = 0

    class_index = np.arange(num_base_class + num_incre_class)

    if args.dataset == 'FMC':
        trainset = args.Dataset.FSDCLIPS(root=args.dataroot, phase='train', index=class_index, k=None)
        valset = args.Dataset.FSDCLIPS(root=args.dataroot, phase='val', index=class_index, k=100) # k is same as new_loader's testset k
    # DataLoader(test_set, batch_sampler=sampler, num_workers=8, pin_memory=True)
    elif 'nsynth' in args.dataset:
        trainset = args.Dataset.NDS(root=args.dataroot, phase='train', index=class_index, k=None, args=args)
        valset = args.Dataset.NDS(root=args.dataroot, phase='val', index=class_index, k=None, args=args) # k is same as new_loader's testset k
    elif 'librispeech' in args.dataset:
        trainset = args.Dataset.LBRS(root=args.dataroot, phase='train', index=class_index, k=None, args=args)
        valset = args.Dataset.LBRS(root=args.dataroot, phase='val', index=class_index, k=None, args=args) # k is same as new_loader's testset k
    elif args.dataset in ['f2n', 'f2l', 'n2f', 'n2l', 'l2f', 'l2n']:
        trainset = args.Dataset.S2S(dataset=args.dataset, root=args.dataroot, phase='train', index=class_index, k=None, args=args)
        valset = args.Dataset.S2S(dataset=args.dataset, root=args.dataroot, phase='val', index=class_index, k=None, args=args) # k is same as new_loader's testset k
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
        num_base_class = args.stdu.num_tmpb
        num_incre_class = args.stdu.num_tmpi
    else:
        num_base_class = args.num_base
        num_incre_class = 0

    class_index = np.arange(num_base_class)

    if args.dataset == 'FMC':
        trainset = args.Dataset.FSDCLIPS(root=args.dataroot, phase='train', index=class_index, k=None)
    elif 'nsynth' in args.dataset:
        trainset = args.Dataset.NDS(root=args.dataroot, phase='train', index=class_index, k=None, args=args)
    elif 'librispeech' in args.dataset:
        trainset = args.Dataset.LBRS(root=args.dataroot, phase='train', index=class_index, k=None, args=args)
    elif args.dataset in ['f2n', 'f2l', 'n2f', 'n2l', 'l2f', 'l2n']:   
        trainset = args.Dataset.S2S(dataset=args.dataset, root=args.dataroot, phase='train', index=class_index, k=None, args=args)
    return trainset

def get_new_dataloader(args, session):
    if args.tmp_train:
        num_base_class = args.stdu.num_tmpb
        num_incre_class = args.stdu.num_tmpi
    else:
        num_base_class = args.num_base
        num_incre_class = 0

    assert session > 0
    if args.dataset == 'FMC':
        session_classes = np.arange(num_base_class + (session -1) * args.way, num_base_class + session * args.way)
        trainset = args.Dataset.FSDCLIPS(root=args.dataroot, phase='train', index=session_classes, k=None)
    elif 'nsynth' in args.dataset:
        session_classes = np.arange(num_base_class + (session -1) * args.way, num_base_class + session * args.way)
        trainset = args.Dataset.NDS(root=args.dataroot, phase='train', index=session_classes, k=None, args=args)
    elif 'librispeech' in args.dataset:
        session_classes = np.arange(num_base_class + (session -1) * args.way, num_base_class + session * args.way)
        trainset = args.Dataset.LBRS(root=args.dataroot, phase='train', index=session_classes, k=None, args=args)
    elif args.dataset in ['f2n', 'f2l', 'n2f', 'n2l', 'l2f', 'l2n']:
        session_classes = np.arange(num_base_class + (session -1) * args.way, num_base_class + session * args.way)
        trainset = args.Dataset.S2S(dataset=args.dataset, root=args.dataroot, phase='train', index=session_classes, k=None, args=args)
    train_sampler = SupportsetSampler(label=trainset.targets, n_cls=args.episode.episode_way, 
                                n_per=args.episode.episode_shot,n_batch=1, seq_sample=args.seq_sample)

    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_sampler=train_sampler, num_workers=8,
                                                pin_memory=True)
                                                
    class_new = get_session_classes(args, session)

    if args.dataset == 'FMC':
        valset = args.Dataset.FSDCLIPS(root=args.dataroot, phase='val', index=class_new, k=None)
    if 'nsynth' in args.dataset:
        valset = args.Dataset.NDS(root=args.dataroot, phase='val', index=class_new, k=None, args=args)
    if 'librispeech' in args.dataset:
        valset = args.Dataset.LBRS(root=args.dataroot, phase='val', index=class_new, k=None, args=args)
    elif args.dataset in ['f2n', 'f2l', 'n2f', 'n2l', 'l2f', 'l2n']:
        valset = args.Dataset.S2S(dataset=args.dataset, root=args.dataroot, phase='val', index=class_new, k=None, args=args)
    valloader = torch.utils.data.DataLoader(dataset=valset, batch_size=args.dataloader.test_batch_size, shuffle=False,
                                                num_workers=8, pin_memory=True)
    return trainset, valset, trainloader, valloader

def get_session_classes(args,  session):
    if args.tmp_train:
        num_base_class = args.stdu.num_tmpb
        num_incre_class = args.stdu.num_tmpi
    else:
        num_base_class = args.num_base
        num_incre_class = 0

    class_list = np.arange(num_base_class + session * args.way)
    return class_list
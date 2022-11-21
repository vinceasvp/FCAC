import abc
from copy import deepcopy
import imp
import os.path as osp
import os
import time

import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
import pandas
import numpy as np
from sklearn.manifold import TSNE

from dataloader.dataloader import get_dataloader, get_dataset_for_data_init, get_testloader
from utils.utils import (
    DAverageMeter,
    acc_utils,
    cal_auxIndex,
    count_per_cls_acc,
    ensure_path,
    Averager, Timer, count_acc,
    plot_embedding,
    plot_tsne,
)


class Trainer(object, metaclass=abc.ABCMeta):
    def __init__(self, args):
        print("************base init************")
        self.args = args
        self.set_up_datasets()
        self.set_save_path()
        self.dt, self.ft = Averager(), Averager()
        self.bt, self.ot = Averager(), Averager()
        self.timer = Timer()

        # train statistics
        self.trlog = {}
        self.trlog['train_loss'] = []
        self.trlog['val_loss'] = []
        self.trlog['test_loss'] = []
        self.trlog['train_acc'] = []
        self.trlog['val_acc'] = []
        self.trlog['test_acc'] = []
        self.trlog['max_acc_epoch'] = 0
        self.trlog['max_acc'] = [0.0] * args.num_session

        self.result_list = [args]
        self.sess_acc_dict = {}

    def set_up_datasets(self):
        if self.args.dataset == 'librispeech':
            import dataloader.librispeech.librispeech as Dataset
        self.args.Dataset=Dataset

    def set_save_path(self):
        if self.args.debug:
            self.args.save_path = 'debug/%s/' % self.args.dataset
        else:
            self.args.save_path = '%s/' % self.args.dataset
        # make a dir for every project
        self.args.save_path = self.args.save_path + '%s/' % self.args.project
        # make a dir for every method
        self.args.save_path = self.args.save_path + self.args.config.split('.')[0].split('/')[-1] + "/"
        # make a dir for different hyper params
        self.args.save_path = self.args.save_path  +self.ns2str(self.args.epochs) + self.ns2str(self.args.lr) \
                   + self.ns2str(self.args.episode)

        self.args.save_path = os.path.join('checkpoints', self.args.save_path) # , time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
        ensure_path(self.args.save_path)
        return None

    def ns2str(self, ns):
        d = vars(ns)
        s = str(d)
        s = s.strip('{').strip('}').replace(",", "").replace(":", "").replace("'", "").replace(" ", "")
        return s

    def get_optimizer(self):
        # for base only
        optimizer = torch.optim.SGD(self.model.parameters(), self.args.lr.lr_base, momentum=0.9, nesterov=True,
                                    weight_decay=self.args.optimizer.decay)
        if self.args.scheduler.schedule == 'Step':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.args.scheduler.step, gamma=self.args.scheduler.gamma)
        elif self.args.scheduler.schedule == 'Milestone':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.args.scheduler.milestones,
                                                             gamma=self.args.scheduler.gamma)
        return optimizer, scheduler
        
    def update_param(self, model, pretrained_dict):
        # for cec or some other model but not for base model
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items()}
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        return model

    def data_init(self, data_dict, session):
        # for base model
        self.model.load_state_dict(self.best_model_dict)
        self.model = self.replace_base_fc(get_dataset_for_data_init(self.args), self.model)
        best_model_dir = os.path.join(self.args.save_path, 'session' + str(session) + '_max_acc.pth')
        print('Replace the fc with average embedding, and save it to :%s' % best_model_dir)
        self.best_model_dict = deepcopy(self.model.state_dict())
        torch.save(dict(params=self.model.state_dict()), best_model_dir)
        
    def replace_base_fc(self, trainset, model):
        num_base_class = self.args.sis.num_tmpb if self.args.tmp_train else self.args.num_base
        model = model.eval()
        assert len(set(trainset.targets)) == num_base_class
        trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=128,
                                                num_workers=8, pin_memory=True, shuffle=False)
        # trainloader.dataset.transform = transform
        embedding_list = []
        label_list = []
        # data_list=[]
        with torch.no_grad():
            for i, batch in enumerate(trainloader):
                data, label = [_.cuda() for _ in batch]
                model.module.mode = 'encoder'
                embedding = model(data)

                embedding_list.append(embedding.cpu())
                label_list.append(label.cpu())
        embedding_list = torch.cat(embedding_list, dim=0)
        label_list = torch.cat(label_list, dim=0)

        proto_list = []
        for class_index in range(num_base_class):
            data_index = (label_list == class_index).nonzero()
            embedding_this = embedding_list[data_index.squeeze(-1)]
            embedding_this = embedding_this.mean(0)
            proto_list.append(embedding_this)
        proto_list = torch.stack(proto_list, dim=0)

        model.module.fc.weight.data[:num_base_class] = proto_list

        return model

    def save_better_model(self, va, net_dict, session, save_model_dir=None):
        # save better model
        if (va * 100) >= self.trlog['max_acc'][session]:
            self.trlog['max_acc'][session] = float('%.3f' % (va * 100))
            self.trlog['max_acc_epoch'] = net_dict['epoch']
            if save_model_dir is None:
                save_model_dir = os.path.join(self.args.save_path, 'session' + str(session) + '_max_acc.pth')
            torch.save(dict(params=self.model.state_dict()), save_model_dir)
            torch.save(net_dict['optimizer'].state_dict(), os.path.join(self.args.save_path, 'optimizer_best.pth'))
            self.best_model_dict = deepcopy(self.model.state_dict())
            print('********A better model is found!!**********')
            print('Saving model to :%s' % save_model_dir)
        print('best epoch {}, best val acc={:.3f}'.format(self.trlog['max_acc_epoch'],
                                                            self.trlog['max_acc'][session]))

    def save_model(self, tsa, session):
        # save model
        self.trlog['max_acc'][session] = float('%.3f' % (tsa * 100))
        save_model_dir = os.path.join(self.args.save_path, 'session' + str(session) + '_max_acc.pth')
        torch.save(dict(params=self.model.state_dict()), save_model_dir)
        self.best_model_dict = deepcopy(self.model.state_dict())
        print('Saving model to :%s' % save_model_dir)
        print('test acc={:.3f}'.format(self.trlog['max_acc'][session]))

    def record_info(self, va, vl, net_dict, res_dict, start_time, epochs):
        self.trlog['val_loss'].append(vl)
        self.trlog['val_acc'].append(va)
        lrc = net_dict['scheduler'].get_last_lr()[0]
        self.result_list.append(
            'epoch:%03d,lr:%.5f,training_loss:%.5f,training_acc:%.5f,val_loss:%.5f,val_acc:%.5f' % (
                net_dict['epoch'], lrc, res_dict['tl'], res_dict['ta'], vl, va))
        self.trlog['train_loss'].append(res_dict['tl'])
        self.trlog['train_acc'].append(res_dict['ta'])
        print('epoch:%03d,lr:%.4f,training_loss:%.5f,training_acc:%.5f,val_loss:%.5f,val_acc:%.5f' % (
            net_dict['epoch'], lrc, res_dict['tl'], res_dict['ta'], vl, va))
        print('This epoch takes %d seconds' % (time.time() - start_time),
                '\nstill need around %.2f mins to finish' % (
                        (time.time() - start_time) * (epochs - net_dict['epoch']) / 60))

    def pretty_output(self):
        final_out_dict = {}
        final_out_dict['cur_acc'] = []
        final_out_dict['base_Acc'] = []
        final_out_dict['novel_Acc'] = []
        final_out_dict['Both_ACC'] = []
        for k, v in self.sess_acc_dict.items():
            final_out_dict['cur_acc'].append(v['cur_acc'])
            final_out_dict['base_Acc'].append(v['base_acc'])
            final_out_dict['novel_Acc'].append(v['novel_acc'])
            final_out_dict['Both_ACC'].append(v['all_acc'])
        cpi, msr_overall, acc_aver_df, ar_over = cal_auxIndex(final_out_dict)
        pd = final_out_dict['Both_ACC'][0] - final_out_dict['Both_ACC'][-1]
        indexes = {'PD':pd, 'CPI':cpi, 'AR':ar_over, 'MSR':msr_overall}
        indexes_df = pandas.DataFrame.from_dict(indexes, orient='index')
        final_df = pandas.DataFrame(final_out_dict)
        final_df = final_df.T
        # pretty output
        pandas.set_option('display.max_rows', None)
        pandas.set_option('display.max_columns', None)
        pandas.set_option('display.width', None)
        pandas.set_option('display.max_colwidth', None)

        excel_fn = os.path.join(self.args.save_path, "output.xlsx")
        print("save output at ", excel_fn)
        writer = pandas.ExcelWriter(excel_fn)
        final_df.to_excel(writer, sheet_name='final_df')
        acc_aver_df.to_excel(writer, sheet_name='final_df', startrow=7)
        indexes_df.to_excel(writer, sheet_name='final_df', startrow=13)
        indexes_df.T.to_excel(writer, sheet_name='final_df', startrow=20)
        writer.save()

        output = f"\nreslut on {self.args.dataset}, method {self.args.project}\
                    \n{self.args.save_path}\
                    \n****************************************Pretty Output********************************************\
                    \n{final_df}\
                    \n===> Comprehensive Performance Index(CPI) v2: {cpi}\n===> PD: {pd}\
                    \n===> Memory Strock Ratio(MSR) Overall: {msr_overall}\n===> Amnesia Rate(AR): {ar_over}\
                    \n===> Acc Average: \n{acc_aver_df}\
                    \n***********************************************************************************************"
        print(output)
        return output


    def load_model(self, model_dir):
        if model_dir != None:  #
            print('Loading init parameters from: %s' % model_dir)
            self.best_model_dict = torch.load(model_dir)['params']
        else:
            print('*********WARNINGl: NO INIT MODEL**********')
            # raise ValueError('You must initialize a pre-trained model')
            pass
        self.model = self.update_param(self.model, self.best_model_dict)

    @abc.abstractmethod
    def train(self):
        pass

    def base_train(self, model, trainloader, optimizer, scheduler, epoch):
        pass

    def test(self, data_dict, model, session):
        pass
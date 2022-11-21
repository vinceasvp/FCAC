from copy import deepcopy
import math
import os
import time
from requests import session
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataloader.dataloader import get_dataloader, get_dataset_for_data_init, get_pretrain_dataloader, get_testloader

from models.base.fscil_trainer import FSCILTrainer as Trainer
from utils.utils import AverageMeter, Averager, DAverageMeter, acc_utils, count_acc, count_per_cls_acc, get_base_novel_ids, save_list_to_txt
from .Network import MYNET



class FSCILTrainer(Trainer):
    def __init__(self, args):

        super().__init__(args)
        print("*********STDU init***************")
        self.set_up_model()

    def set_up_model(self):
        self.model = MYNET(self.args, mode=self.args.network.base_mode)
        # print(MYNET)
        # print(self.model)
        self.model = nn.DataParallel(self.model, list(range(self.args.num_gpu)))
        self.model = self.model.cuda()

        if self.args.model_dir.sis_model_dir != None:  #
            print('Loading init parameters from: %s' % self.args.model_dir.sis_model_dir)
            self.best_model_dict = torch.load(self.args.model_dir.sis_model_dir)['params']
        else:
            print('*********WARNINGl: NO INIT MODEL**********')
            # raise ValueError('You must initialize a pre-trained model')
            pass

    def get_optimizer_incremental(self):

        optimizer = torch.optim.SGD([{'params': self.model.module.encoder.parameters(), 'lr': self.args.lr.lr_sis_base},
                                     {'params': self.model.module.slf_attn.parameters(), 'lr': self.args.lr.lrg}, 
                                     {'params': self.model.module.transatt_proto.parameters(), 'lr': self.args.lr.lrg}],
                                    momentum=0.9, nesterov=True, weight_decay=self.args.optimizer.decay)

        if self.args.scheduler.schedule == 'Step':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.args.scheduler.step, gamma=self.args.scheduler.gamma)
        elif self.args.scheduler.schedule == 'Milestone':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.args.scheduler.milestones,
                                                             gamma=self.args.scheduler.gamma)

        return optimizer, scheduler

    def get_optimizer_standard(self):

        optimizer = torch.optim.SGD([{'params': self.model.module.encoder.parameters(), 'lr': self.args.lr.lr_std}],
                                    momentum=0.9, nesterov=True, weight_decay=self.args.optimizer.decay)

        if self.args.scheduler.schedule == 'Step':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.args.scheduler.step, gamma=self.args.scheduler.gamma)
        elif self.args.scheduler.schedule == 'Milestone':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.args.scheduler.milestones,
                                                             gamma=self.args.scheduler.gamma)

        return optimizer, scheduler
    

    def train(self):
        # args = self.args
        if self.args.model_dir.sis_model_dir != None:
            self.set_up_datasets(self.args)
            self.best_model_dict = torch.load(self.args.model_dir.sis_model_dir)['params']
        else:
            self.STDU_train()

        for session in range(self.args.start_session, self.args.num_session):
            acc_dict_aver = DAverageMeter()
            tsl_aver = AverageMeter()
            tsa_aver = AverageMeter()
            time_aver = AverageMeter()
            data_dict = {}
            data_dict['train_set'], data_dict['valset'], data_dict['trainloader'], data_dict['valloader'] \
                = get_dataloader(self.args, session)
            data_dict['testset'], data_dict['testloader'] = get_testloader(self.args, session)

            self.model = self.update_param(self.model, self.best_model_dict)

            if session == 0:  # load base class train img label           
                # always replace fc with avg mean and save the replaced model
                self.data_init(data_dict, session)

                self.model.module.mode = 'avg_cos'
                tsl, tsa, acc_dict, cls_sample_count = self.test(self.model, data_dict['testloader'], session, data_dict['trainloader'])

                self.sess_acc_dict[f'sess {session}'] = acc_dict
                self.trlog['max_acc'][session] = float('%.3f' % (tsa * 100))
                self.result_list.append('Session {}, Test Best Epoch {},\nbest test Acc {:.4f}\n'.format(
                    session, self.trlog['max_acc_epoch'], self.trlog['max_acc'][session], ))

                # print(acc_dict)
                print(cls_sample_count)
                print('The test acc of base session={:.3f}'.format(self.trlog['max_acc'][session]))


            else:  # incremental learning sessions
                print("Inference session: [%d]" % session)
                self.model.load_state_dict(self.best_model_dict)
                self.model.module.mode = self.args.network.new_mode
                self.model.eval()
                
                for i in range(self.args.test_times):
                    get_dataloader(self.args, session)
                    self.model.module.update_fc(data_dict['trainloader'], np.unique(data_dict['train_set'].targets), session)
                    prototypes = self.model.module.fc.weight.data[:self.args.num_base+self.args.way*session]
                    prototypes = prototypes.detach().cpu().numpy()
                    np.save(f"/data/caowc/FSCIL/checkpoints/save_data/session{session}_mean_prototypes.npy", prototypes)
                    tmp_tsl, tmp_tsa, tmp_acc_dict, cls_sample_count = self.test(self.model, data_dict['testloader'], session, data_dict['trainloader'])
                    tsl_aver.update(tmp_tsl)
                    tsa_aver.update(tmp_tsa)
                    acc_dict_aver.update(tmp_acc_dict)
                tsl = tsl_aver.average()
                tsa = tsa_aver.average()
                acc_dict = acc_dict_aver.average()

                self.sess_acc_dict[f'sess {session}'] = acc_dict

                # save model
                self.save_model(tsa, session)
                self.result_list.append('Session {}, Test Best Epoch {},\nbest test Acc {:.4f}\n'.format(
                    session, self.trlog['max_acc_epoch'], self.trlog['max_acc'][session]))

        print(cls_sample_count)
        output = self.pretty_output()

        self.result_list.append(f"Base Session Best epoch:{self.trlog['max_acc_epoch']}, \
            Base Session Best acc:{self.trlog['max_acc']}")
        self.result_list.append(self.sess_acc_dict)
        self.result_list.append(output)
        save_list_to_txt(os.path.join(self.args.save_path, 'results.txt'), self.result_list)

        print(f"Base Session Best epoch:{self.trlog['max_acc_epoch']}, \
            Base Session Best acc:{self.trlog['max_acc']}")

    def STDU_train(self):
        self.args.tmp_train = True
        if self.args.model_dir.tmp_model_dir is not None:
            self.load_model(self.args.model_dir.tmp_model_dir)
            self.standard_train(temp=True, pretrained=True)
        else:
            self.standard_train(temp=True)

        self.reset_trlog(self.args.sis.num_tmps)
        self.incremental_train()

        self.args.tmp_train = False
        self.reset_trlog(self.args.num_session)
        if self.args.model_dir.s0_model_dir is not None:
            self.load_model(self.args.model_dir.s0_model_dir)
            self.standard_train(temp=False, pretrained=True)
        else:
            self.standard_train(temp=False)


    def base_train(self, model, trainloader, optimizer, scheduler, epoch, args):
        tl = Averager()
        ta = Averager()
        tqdm_gen = tqdm(trainloader)

        label = torch.arange(args.episode.episode_way + args.episode.low_way).repeat(args.episode.episode_query)
        label = label.type(torch.cuda.LongTensor)

        for i, batch in enumerate(tqdm_gen, 1):
            data, true_label = [_.cuda() for _ in batch]

            bse_idx = args.episode.low_way * args.episode.low_shot # base support end
            bqe_idx = args.episode.low_way * (args.episode.low_shot + args.episode.episode_query) # base query end
            nse_idx = args.episode.episode_way * args.episode.episode_shot + args.episode.low_way * (args.episode.low_shot + args.episode.episode_query) # novel support end
            nqe_idx = args.episode.episode_way * (args.episode.episode_shot + args.episode.episode_query) + args.episode.low_way * (args.episode.low_shot + args.episode.episode_query) # novel query end
            base_labels = true_label[ : bqe_idx]
            bs_labels = true_label[ : bse_idx]
            bq_labels = true_label[bse_idx : bqe_idx]
            novel_labels = true_label[bqe_idx : ]
            ns_labels = true_label[bqe_idx : nse_idx]
            nq_labels = true_label[nse_idx : nqe_idx]

            base_ids, novel_ids =  base_labels.unique(), novel_labels.unique()
            print(f'\nBase classes:{base_ids}, novel classes:{novel_ids}')
            
            # sample low_way data
            # k = args.episode_way * args.episode_shot
            proto, query = data[:bse_idx], data[bse_idx:bqe_idx]
            proto_novel = data[bqe_idx:nse_idx]
            query_novel = data[nse_idx: nqe_idx]

            # encoder data to get embeddings
            model.module.mode = 'encoder'
            data = model(data[:bqe_idx])
            proto_novel = model(proto_novel)
            query_novel = model(query_novel)

            # split embeddins
            proto, query = data[:bse_idx], data[bse_idx:bqe_idx]
            if self.args.sis.ap.use_ap:
                sup_emb = torch.cat([proto, proto_novel], dim=0)
                novel_ids = label.unique()
            else:
                sup_emb = None

            # reshape embeddings to shape(shot, way, embedding_dim)
            proto = proto.view(args.episode.low_shot, args.episode.low_way, proto.shape[-1])
            query = query.view(args.episode.episode_query, args.episode.low_way, query.shape[-1])
            proto_novel = proto_novel.view(args.episode.episode_shot, args.episode.episode_way, proto_novel.shape[-1])
            query_novel = query_novel.view(args.episode.episode_query, args.episode.episode_way, query_novel.shape[-1])

            # calculate the mean of shot axis to get prototype
            proto = proto.mean(0, keepdim=True)
            proto_novel = proto_novel.mean(0, keepdim=True)
            proto = torch.cat([proto, proto_novel], dim=1)
            query = torch.cat([query, query_novel], dim=1)

            proto = proto.unsqueeze(0)
            query = query.unsqueeze(0)

            logits, anchor_loss, pqa_query, pqa_proto  = model.module._forward(proto, query, args.STDU.pqa, sup_emb=sup_emb, novel_ids=novel_ids)

            total_loss = F.cross_entropy(logits, label)

            total_loss = total_loss + anchor_loss
            acc = count_acc(logits, label)
            ta.add(acc)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            tl.add(total_loss.item())
            lrc = scheduler.get_last_lr()[0]
            tqdm_gen.set_description(
                'Session 0, epo {}, lrc={:.4f},total loss={:.4f} acc={:.4f}'.format(epoch, lrc, total_loss.item(), ta.item()))

        tl = tl.item()
        ta = ta.item()
        return tl, ta


    def standard_train(self, temp=False, pretrained=False):
        session = 0
        # prepare data
        data_dict = {}
        data_dict['train_set'], data_dict['valset'], data_dict['trainloader'], data_dict['valloader'] \
            = get_pretrain_dataloader(self.args)
        data_dict['testset'], data_dict['testloader'] = get_testloader(self.args, session)

        # prepare model
        # self.model.load_state_dict(self.best_model_dict)

        # prepare net_dict
        net_dict = {}
        if temp:
            print('==> Classes for this TEMPORARY standard train stage:\n', np.unique(data_dict['train_set'].targets))
        else:
            print('==> Classes for this standard train stage:\n', np.unique(data_dict['train_set'].targets))

        net_dict['optimizer'], net_dict['scheduler'] = self.get_optimizer_standard()

        """****************train and val*************************"""
        if not pretrained:
            for epoch in range(self.args.epochs.epochs_std):
                start_time = time.time()
                # train base sess, here is a normal mode
                tl, ta = self.standard_base_train(self.model, data_dict['trainloader'], net_dict['optimizer'], net_dict['scheduler'], epoch, temp)
                # test model with all seen class
                net_dict['epoch'] = epoch
                res_dict = {'tl': tl, 'ta': ta}
                tsl, tsa, acc_dict, cls_sample_count = self.standard_test(self.model, data_dict['testloader'], epoch, session, temp)
                if temp:
                    save_model_dir = os.path.join(self.args.save_path, f'temp_std_train{self.args.epochs.epochs_std}_max_acc.pth')
                else:
                    save_model_dir = os.path.join(self.args.save_path, f'std_train{self.args.epochs.epochs_std}_max_acc.pth')
                self.save_better_model(tsa, net_dict, session, save_model_dir)
                self.record_info(tsa, tsl, net_dict, res_dict, start_time, self.args.epochs.epochs_std)
                net_dict['scheduler'].step()
            """****************record on best model*************************"""
            if temp:
                self.result_list.append('TEMPORARY standard train stage, Test Best Epoch {},\nbest test Acc {:.4f}\n'.format(
                            self.trlog['max_acc_epoch'], self.trlog['max_acc'][session], ))
            else:
                self.result_list.append('Standard train stage, Test Best Epoch {},\nbest test Acc {:.4f}\n'.format(
                            self.trlog['max_acc_epoch'], self.trlog['max_acc'][session], ))
            # self.sess_acc_dict[f'sess {session}'] = acc_dict
            print('test loss={:.3f}, test acc={:.3f}'.format(tsl, tsa))

        if not self.args.strategy.not_data_init:
            """****************data init and test again*************************"""
            #data init and replace the model
            self.data_init(data_dict, session)
            self.model.module.mode = 'avg_cos'
            tsl, tsa, acc_dict, cls_sample_count = self.standard_test(self.model,data_dict['testloader'], 0, session, temp)
            print(cls_sample_count)
            # self.sess_acc_dict[f'sess {session}'] = acc_dict
            if (tsa * 100) >= self.trlog['max_acc'][session]:
                self.trlog['max_acc'][session] = float('%.3f' % (tsa * 100))
                if temp:
                    print('The NEW(after data init) best test acc of TEMPORARY standard train stage={:.3f}'.format(self.trlog['max_acc'][session]))
                else:
                    print('The NEW(after data init) best test acc of standard train stage={:.3f}'.format(self.trlog['max_acc'][session]))

        # output = self.pretty_output()

        if temp:
            self.result_list.append(f"==> TEMPORARY standard train stage: Best epoch:{self.trlog['max_acc_epoch']}, \
                Best acc:{self.trlog['max_acc']}")
        else:
            self.result_list.append(f"==> Standard train stage: Best epoch:{self.trlog['max_acc_epoch']}, \
                Best acc:{self.trlog['max_acc']}")
        # self.result_list.append(self.sess_acc_dict)
        # self.result_list.append(output)
        save_list_to_txt(os.path.join(self.args.save_path, 'results.txt'), self.result_list)

        if temp:
            print(f"==> TEMPORARY standard train stage: Best epoch:{self.trlog['max_acc_epoch']}, \
                Best acc:{self.trlog['max_acc']}")
        else:
            print(f"==> Standard train stage: Best epoch:{self.trlog['max_acc_epoch']}, \
                Best acc:{self.trlog['max_acc']}")
        print('Total time used %.2f mins' % total_time)

    def standard_base_train(self, model, trainloader, optimizer, scheduler, epoch, temp):
        num_base = self.args.sis.num_tmpb if temp else self.args.num_base
        tl = Averager()
        ta = Averager()
        model = model.train()
        model.module.mode = 'encoder'
        # standard classification for pretrain
        tqdm_gen = tqdm(trainloader)
        for i, batch in enumerate(tqdm_gen, 1):
            data, train_label = [_.cuda() for _ in batch]

            logits = model(data)
            logits = logits[:, :num_base]
            loss = F.cross_entropy(logits, train_label)
            acc = count_acc(logits, train_label)

            total_loss = loss

            lrc = scheduler.get_last_lr()[0]
            if temp:
                tqdm_gen.set_description(
                    'TEMPORARY standard train, epo {}, lrc={:.4f},total loss={:.4f} acc={:.4f}'.format(epoch, lrc, total_loss.item(), acc))
            else:
                tqdm_gen.set_description(
                    'Standard train, epo {}, lrc={:.4f},total loss={:.4f} acc={:.4f}'.format(epoch, lrc, total_loss.item(), acc))
            tl.add(total_loss.item())
            ta.add(acc)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        tl = tl.item()
        ta = ta.item()
        return tl, ta

    def standard_test(self, model, testloader, epoch, session, temp):
        num_base = self.args.sis.num_tmpb if temp else self.args.num_base
        num_session = self.args.sis.num_tmps if temp else self.args.num_session
        test_class = num_base + session * self.args.way
        model = model.eval()
        model.module.mode = 'encoder'
        vl = Averager()
        va = Averager()
        da = DAverageMeter()
        ca = DAverageMeter()
        with torch.no_grad():
            tqdm_gen = tqdm(testloader)
            for i, batch in enumerate(tqdm_gen, 1):
                data, test_label = [_.cuda() for _ in batch]
                logits = model(data)
                logits = logits[:, :test_class]
                loss = F.cross_entropy(logits, test_label)
                acc = count_acc(logits, test_label)
                acc = count_acc(logits, test_label)
                per_cls_acc, cls_sample_count = count_per_cls_acc(logits, test_label)
                vl.add(loss.item())
                va.add(acc)
                da.update(per_cls_acc)
                ca.update(cls_sample_count)
            vl = vl.item()
            va = va.item()
            da = da.average()
            ca = ca.average()
            acc_dict = acc_utils(da, num_base,num_session, self.args.way, session)
        print(acc_dict)
        # print(ca)
        if temp:
            print('epo {}, TEMPORARY standard test, loss={:.4f} acc={:.4f}'.format(epoch, vl, va))
        else:
            print('epo {}, standard test, loss={:.4f} acc={:.4f}'.format(epoch, vl, va))
        return vl, va, acc_dict, ca

    def incremental_train(self):
        session = 0
        data_dict = {}
        data_dict['train_set'], data_dict['valset'], data_dict['trainloader'], data_dict['valloader'] \
            = get_dataloader(self.args, session)
        data_dict['testset'], data_dict['testloader'] = get_testloader(self.args, session)

        self.model = self.update_param(self.model, self.best_model_dict)

        print('new classes for this session:\n', np.unique(data_dict['train_set'].targets))
        optimizer, scheduler = self.get_optimizer_incremental()

        for epoch in range(self.args.epochs.epochs_sis_base):
            start_time = time.time()
            # train base sess
            self.model.eval()
            tl, ta = self.base_train(self.model, data_dict['trainloader'], optimizer, scheduler, epoch, self.args)

            self.model = self.replace_base_fc(get_dataset_for_data_init(self.args), self.model)

            self.model.module.mode = 'avg_cos'

            # prepare to validate
            net_dict = {'optimizer': optimizer, 'scheduler': scheduler, 'epoch': epoch}
            res_dict = {'result_list': self.result_list, 'tl': tl, 'ta': ta}
            vl, va, acc_dict, cls_sample_count = self.validate(session, net_dict,  res_dict, start_time)
            save_model_dir = os.path.join(self.args.save_path, f'incre_train{self.args.epochs.epochs_sis_base}_max_acc.pth')
            self.save_better_model(va, net_dict, session, save_model_dir)
            self.record_info(va, vl, net_dict, res_dict, start_time, self.args.epochs.epochs_sis_base)  
            net_dict['scheduler'].step()               
        # always replace fc with avg mean and save the replaced model
        self.data_init(data_dict, session)

        self.model.module.mode = 'avg_cos'
        tsl, tsa, acc_dict, cls_sample_count = self.test(self.model, data_dict['testloader'], session, data_dict['trainloader'])
        # self.sess_acc_dict[f'sess {session}'] = acc_dict
        self.trlog['max_acc'][session] = float('%.3f' % (tsa * 100))
        self.result_list.append('Incremental train, Test Best Epoch {},\nbest test Acc {:.4f}\n'.format(
            self.trlog['max_acc_epoch'], self.trlog['max_acc'][session]))

        print(acc_dict)
        print(cls_sample_count)
        print('The test acc of incremental train={:.3f}'.format(self.trlog['max_acc'][session]))

    def reset_trlog(self, sessions):
        self.trlog = {}
        self.trlog['train_loss'] = []
        self.trlog['val_loss'] = []
        self.trlog['test_loss'] = []
        self.trlog['train_acc'] = []
        self.trlog['val_acc'] = []
        self.trlog['test_acc'] = []
        self.trlog['max_acc_epoch'] = 0
        self.trlog['max_acc'] = [0.0] * sessions

    def validate(self, session, net_dict,  res_dict, start_time):
        # take the last session's testloader for validation
        print('>>>= validation stage')
        with torch.no_grad():
            model = self.model

            for session in range(1, self.args.num_session):
                train_set, valset, trainloader, valloader = get_dataloader(self.args, session)

                # trainloader.dataset.transform = testloader.dataset.transform
                model.module.mode = 'avg_cos'
                model.eval()
                model.module.update_fc(trainloader, np.unique(train_set.targets), session)

                vl, va, acc_dict, cls_sample_count = self.test(model,valloader,  session, trainloader)
                print('Validation: Session {}, total loss={:.4f}, acc={:.4f}'.format(session, vl, va))
        return vl, va, acc_dict, cls_sample_count

    def test(self, model, testloader,  session, trainloader):
        test_class = self.args.num_base + session * self.args.way

        model = model.eval()
        if session>0:
            if self.args.sis.ap.ap_on_test:
                sup_emb, novel_ids = self.get_sup_emb(trainloader)
            else:
                sup_emb = None
                novel_ids = None
        else:
            sup_emb = None
            novel_ids = None
        vl = Averager()
        va = Averager()
        da = DAverageMeter()
        ca = DAverageMeter()
        fc_proto = model.module.fc.weight[:test_class, :].detach()
        # plot_tsne(fc_proto, test_class, "figs/session{}.png".format(session), session)
        pred_list = []
        label_list = []
        raw_query_list = []
        pqa_query_list = []
        pqa_proto_list = []
        with torch.no_grad():
            for i, batch in enumerate(testloader, 1):
                data, test_label = [_.cuda() for _ in batch]
                model.module.mode = 'encoder'
                query = model(data)
                label_list.append(test_label)
                raw_query_list.append(query)
                query = query.unsqueeze(0).unsqueeze(0)
                proto = model.module.fc.weight[:test_class, :].detach()
                proto = proto.unsqueeze(0).unsqueeze(0)

                logits, anchor_loss, pqa_query, pqa_proto = model.module._forward(proto, query, self.args.STDU.pqa, sup_emb, novel_ids)
                pqa_query_list.append(pqa_query)
                pqa_proto_list.append(pqa_proto)
                pred = torch.argmax(logits, dim=1)
                if session == self.args.num_session - 1:
                    pred_list.append(pred)
                    label_list.append(test_label)
                loss = F.cross_entropy(logits, test_label)
                loss = loss + anchor_loss
                acc = count_acc(logits, test_label)
                per_cls_acc, cls_sample_count = count_per_cls_acc(logits, test_label)
                vl.add(loss.item())
                va.add(acc)
                da.update(per_cls_acc)
                ca.update(cls_sample_count)
            vl = vl.item()
            va = va.item()
            da = da.average()
            ca = ca.average()
            acc_dict = acc_utils(da, self.args.num_base, self.args.num_session, self.args.way, session)
        print(acc_dict)
        return vl, va, acc_dict, ca
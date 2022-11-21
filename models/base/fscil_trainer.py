from sklearn import svm
from tqdm import tqdm
from .base import Trainer
from .Network import MYNET
import os.path as osp
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy

from utils.utils import *
from dataloader.dataloader import get_dataloader, get_testloader


class FSCILTrainer(Trainer):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.set_up_model()

    def set_up_model(self):
        self.model = MYNET(self.args, mode=self.args.network.base_mode)
        print(MYNET)
        self.model = nn.DataParallel(self.model, list(range(self.args.num_gpu)))
        self.model = self.model.cuda()

        if self.args.model_dir.base_model_dir is not None:
            print('Loading init parameters from: %s' % self.args.model_dir.base_model_dir)
            self.best_model_dict = torch.load(self.args.model_dir.base_model_dir)['params']
            # self.best_model_dict = torch.load(self.args.model_dir)['state_dict']
        else:
            print('random init params')
            if self.args.start_session > 0:
                print('WARING: Random init weights for new sessions!')
            self.best_model_dict = deepcopy(self.model.state_dict())

    def train(self):

        for session in range(self.args.start_session, self.args.num_session):
            # prepare data
            data_dict = {}
            data_dict['train_set'], data_dict['valset'], data_dict['trainloader'], data_dict['valloader'] \
                = get_dataloader(self.args, session)
            data_dict['testset'], data_dict['testloader'] = get_testloader(self.args, session)

            # prepare model
            self.model.load_state_dict(self.best_model_dict)

            # prepare net_dict
            net_dict = {}
            if session == 0:  # load base class train img label
                print('new classes for this session:\n', np.unique(data_dict['train_set'].targets))
                net_dict['optimizer'], net_dict['scheduler'] = self.get_optimizer()

                """****************train and val*************************"""
                if not self.args.model_dir.base_model_dir:
                    for epoch in range(self.args.epochs.epochs_base):
                        start_time = time.time()
                        # train base sess, here is a normal mode
                        tl, ta = self.base_train(self.model, data_dict['trainloader'], net_dict['optimizer'], net_dict['scheduler'], epoch)
                        # test model with all seen class
                        net_dict['epoch'] = epoch
                        res_dict = {'tl': tl, 'ta': ta}
                        tsl, tsa, acc_dict, cls_sample_count = self.test(self.model, data_dict['testloader'], epoch, session)
                        self.save_better_model(tsa, net_dict, session)
                        self.record_info(tsa, tsl, net_dict, res_dict, start_time, self.args.epochs.epochs_base)
                        net_dict['scheduler'].step()
                """****************record on best model*************************"""
                self.result_list.append('Session {}, Test Best Epoch {},\nbest test Acc {:.4f}\n'.format(
                    session, self.trlog['max_acc_epoch'], self.trlog['max_acc'][session], ))
                self.sess_acc_dict[f'sess {session}'] = acc_dict
                print('test loss={:.3f}, test acc={:.3f}'.format(tsl, tsa))

                if not self.args.strategy.not_data_init:
                    """****************data init and test again*************************"""
                    #data init and replace the model
                    self.data_init(data_dict, session)
                    self.model.module.mode = 'avg_cos'
                    tsl, tsa, acc_dict, cls_sample_count = self.test(self.model, data_dict['testloader'], epoch, session)
                    print(cls_sample_count)
                    self.sess_acc_dict[f'sess {session}'] = acc_dict
                    if (tsa * 100) >= self.trlog['max_acc'][session]:
                        self.trlog['max_acc'][session] = float('%.3f' % (tsa * 100))
                        print('The new best test acc of base session={:.3f}'.format(self.trlog['max_acc'][session]))

            else:  # incremental learning sessions
                print("training session: [%d]" % session)

                self.model.module.mode = self.args.network.new_mode
                self.model.eval()
                # trainloader.dataset.transform = testloader.dataset.transform
                self.model.module.update_fc(data_dict['trainloader'], np.unique(data_dict['train_set'].targets), session)
                
                tsl, tsa, acc_dict, cls_sample_count = self.test(self.model, data_dict['testloader'], 0, session) 
                self.sess_acc_dict[f'sess {session}'] = acc_dict 
                self.save_model(tsa, session) 
                self.result_list.append('Session {}, test Acc {:.3f}\n'.format(session, self.trlog['max_acc'][session]))

        output = self.pretty_output()

        self.result_list.append(f"Base Session Best epoch:{self.trlog['max_acc_epoch']}, \
            Base Session Best acc:{self.trlog['max_acc']}")
        self.result_list.append(self.sess_acc_dict)
        self.result_list.append(output)
        save_list_to_txt(os.path.join(self.args.save_path, 'results.txt'), self.result_list)

        print(f"Base Session Best epoch:{self.trlog['max_acc_epoch']}, \
            Base Session Best acc:{self.trlog['max_acc']}")
        print('Total time used %.2f mins' % total_time)

    def base_train(self, model, trainloader, optimizer, scheduler, epoch):
        tl = Averager()
        ta = Averager()
        model = model.train()
        # standard classification for pretrain
        tqdm_gen = tqdm(trainloader)
        for i, batch in enumerate(tqdm_gen, 1):
            data, train_label = [_.cuda() for _ in batch]

            logits = model(data)
            logits = logits[:, :self.args.num_base]
            loss = F.cross_entropy(logits, train_label)
            acc = count_acc(logits, train_label)

            total_loss = loss

            lrc = scheduler.get_last_lr()[0]
            tqdm_gen.set_description(
                'Session 0, epo {}, lrc={:.4f},total loss={:.4f} acc={:.4f}'.format(epoch, lrc, total_loss.item(), acc))
            tl.add(total_loss.item())
            ta.add(acc)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        tl = tl.item()
        ta = ta.item()
        return tl, ta

    def test(self, model, testloader, epoch, session):
        test_class = self.args.num_base + session * self.args.way
        model = model.eval()
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
            acc_dict = acc_utils(da, self.args.num_base, self.args.num_session, self.args.way, session)
        print(acc_dict)
        print(ca)
        print('epo {}, test, loss={:.4f} acc={:.4f}'.format(epoch, vl, va))
        return vl, va, acc_dict, ca



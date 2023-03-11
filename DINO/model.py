import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

import numpy, tqdm, sys, time, soundfile

from loss import *
from models.ECAPATDNN512 import *
from tools import *
import dino_utils as utils

class model(nn.Module):
    def __init__(self, niter_per_ep=1092009, **kwargs):
        super(model, self).__init__()
        self._S_Net = utils.Wrapper(
            ECAPA_TDNN(**kwargs), 
            LossFunction(
                nOut = kwargs['nOut'],
                out_dim = kwargs['out_dim'], 
                use_bn_in_head = kwargs['use_bn_in_head'],  
                norm_last_layer = kwargs['norm_last_layer'],
                nlayers = kwargs['nlayers'],
                hidden_dim = kwargs['hidden_dim'],
                )
            ).cuda()
        self._T_Net = utils.Wrapper(
            ECAPA_TDNN(**kwargs), 
            LossFunction(
                nOut = kwargs['nOut'],
                out_dim = kwargs['out_dim'], 
                use_bn_in_head = kwargs['use_bn_in_head'],  
                norm_last_layer = True,
                nlayers = kwargs['nlayers'],
                hidden_dim = kwargs['hidden_dim'],
                )
            ).cuda()

        self._T_Net_without_ddp = self._T_Net
        self._T_Net_without_ddp.load_state_dict(self._S_Net.state_dict())
        # there is no backpropagation through the teacher, so no need for gradients
        for p in self._T_Net.parameters():
            p.requires_grad = False
        print(f"Student and Teacher are built.")
            
        self.dino_loss = DINOLoss(
            out_dim = kwargs['out_dim'], 
            local_view_num = kwargs['local_view_num'],
            warmup_teacher_temp = kwargs['warmup_teacher_temp'],
            teacher_temp = kwargs['teacher_temp'],
            warmup_teacher_temp_epochs = kwargs['warmup_teacher_temp_epochs'],
            max_epoch = kwargs['max_epoch'],
            student_temp = kwargs['student_temp'],
            center_momentum = kwargs['center_momentum']
            ).cuda()

        params_groups = utils.get_params_groups(self._S_Net)
        if kwargs['optimizer'] == "adamw":
            self.optimizer = torch.optim.AdamW(params_groups)  # to use with ViTs
        elif kwargs['optimizer'] == "adam":
            self.optimizer = torch.optim.Adam(params_groups, betas = (0.9, 0.95), amsgrad=True)    
        elif kwargs['optimizer'] == "sgd":
            self.optimizer = torch.optim.SGD(params_groups, lr=0, momentum=0.9)  # lr is set by scheduler
        elif kwargs['optimizer'] == "lars":
            self.optimizer = utils.LARS(params_groups)

        print("Model para number = %.2f"%(sum(param.numel() for param in self._S_Net.parameters()) / 1024 / 1024))

        # niter_per_ep = 148642 ##length of VoxCeleb1 niter_per_ep = 1092009 ##length of VoxCeleb2
        self.lr_schedule = utils.cosine_scheduler(
            kwargs['lr'],
            kwargs['lr_min'],
            kwargs['max_epoch'], niter_per_ep,
            warmup_epochs=kwargs['warmup_epochs'],
            start_warmup_value=1e-8,
        )
        self.wd_schedule = utils.cosine_scheduler(
            kwargs['weight_decay'],
            kwargs['weight_decay_end'],
            kwargs['max_epoch'], niter_per_ep,
        )
        self.momentum_schedule = utils.cosine_scheduler(
            kwargs['momentum_teacher'],
            1,
            kwargs['max_epoch'], niter_per_ep
        )

        self.clip_grad = kwargs['clip_grad']
        self.freeze_last_layer = kwargs['freeze_last_layer']

    def train_network(self, loader, epoch):
        stepsize = loader.batch_size;
        epoch = epoch-1
        self.train()                
        loss_total = 0
        
        tstart = time.time() # Used to monitor the training speed
        for counter, data in enumerate(loader, start = 1):

            iters = (len(loader) * epoch + counter)*stepsize  # global training iteration
            for i, param_group in enumerate(self.optimizer.param_groups):
                lr = self.lr_schedule[iters]
                param_group["lr"] = lr
                if i == 0:  # only the first group is regularized
                    param_group["weight_decay"] = self.wd_schedule[iters]

            global_data, local_data = data

            global_data = global_data.transpose(0,1)
            local_data = local_data.transpose(0,1)

            global_shape = global_data.size()
            local_shape = local_data.size()

            global_data = global_data.reshape(global_shape[0]*global_shape[1], global_shape[2])
            local_data = local_data.reshape(local_shape[0]*local_shape[1], local_shape[2])

            # move images to gpu
            global_data = torch.FloatTensor(global_data).cuda(non_blocking=True)
            local_data = torch.FloatTensor(local_data).cuda(non_blocking=True)
            all_data = [global_data, local_data]

            # teacher and student forward passes + compute dino loss
            teacher_output = self._T_Net(global_data)  # only the 2 global views pass through the teacher
            student_output = self._S_Net(all_data)

            loss = self.dino_loss(student_output, teacher_output, epoch)   

            if not math.isfinite(loss.item()):
                print("Loss is {}, stopping training".format(loss.item()), force=True)
                sys.exit(1)

            # student update
            self.optimizer.zero_grad()
            loss_it = loss.detach().cpu()
            loss_total    += loss_it
            
            param_norms = None
            loss.backward()
            if self.clip_grad:
                param_norms = utils.clip_gradients(self._S_Net, self.clip_grad)
            utils.cancel_gradients_last_layer(epoch, self._S_Net,
                                              self.freeze_last_layer)
            self.optimizer.step()


            # EMA update for the teacher
            with torch.no_grad():
                m = self.momentum_schedule[iters]  # momentum parameter
                for param_q, param_k in zip(self._S_Net.parameters(), self._T_Net_without_ddp.parameters()):
                    param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)
            
            # logging
            torch.cuda.synchronize()

            time_used = time.time() - tstart # Time for this epoch
            sys.stdout.write("[%2d] Lr: %6f, %.2f%% (est %.1f mins) Mean Loss %.4f (Loss %.4f)\r"%(epoch, lr, 100 * (counter / loader.__len__()), time_used * loader.__len__() / counter / 60, loss_total/counter, loss_it))
            sys.stdout.flush()

        sys.stdout.write("\n")
        return loss_total/counter, lr

    def evaluate_network(self, val_list, val_path, **kwargs):
        self.eval()
        files, feats = [], {}
        for line in open(val_list).read().splitlines():
            data = line.split()
            files.append(data[1])
            files.append(data[2])
        setfiles = list(set(files))
        setfiles.sort()  # Read the list of wav files
        for idx, file in tqdm.tqdm(enumerate(setfiles), total = len(setfiles)):
            audio, _ = soundfile.read(os.path.join(val_path, file))
            feat = torch.FloatTensor(numpy.stack([audio], axis=0)).cuda()
            with torch.no_grad():
                ref_feat = self._T_Net_without_ddp.backbone.forward(feat).detach().cpu()
            feats[file]     = ref_feat # Extract features for each data, get the feature dict
        scores, labels  = [], []
        for line in open(val_list).read().splitlines():
            data = line.split()
            ref_feat = F.normalize(feats[data[1]].cuda(), p=2, dim=1) # feature 1
            com_feat = F.normalize(feats[data[2]].cuda(), p=2, dim=1) # feature 2
            score = numpy.mean(torch.matmul(ref_feat, com_feat.T).detach().cpu().numpy()) # Get the score
            scores.append(score)
            labels.append(int(data[0]))
        EER = tuneThresholdfromScore(scores, labels, [1, 0.1])[1]
        fnrs, fprs, thresholds = ComputeErrorRates(scores, labels)
        minDCF, _ = ComputeMinDcf(fnrs, fprs, thresholds, 0.05, 1, 1)
        return EER, minDCF

    def save_network(self, path): # Save the model
        save_dict = {
            'network': self.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        utils.save_on_master(save_dict, path)

    def load_network(self, path): # Load the parameters of the pretrain model
        self_state = self.state_dict()
        checkpoint = torch.load(path, map_location="cpu")
        print("Ckpt file %s loaded!"%(path))

        if 'network' not in checkpoint.keys():
            loaded_state = checkpoint
        else:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("Optimizer loaded!")
            loaded_state = checkpoint['network']

        for name, param in loaded_state.items():
            origname = name
            if name not in self_state:
                name = name.replace("module.", "")
                if name not in self_state:
                    print("%s is not in the model."%origname)
                    continue
            if self_state[name].size() != loaded_state[origname].size():
                print("Wrong parameter length: %s, model: %s, loaded: %s"%(origname, self_state[name].size(), loaded_state[origname].size()))
                continue
            self_state[name].copy_(param)
        print("Model loaded!")

class DINOLoss(nn.Module):
    def __init__(self, out_dim, local_view_num, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, max_epoch, student_temp=0.1,
                 center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = local_view_num+2
        self.register_buffer("center", torch.zeros(1, out_dim))
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(max_epoch - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def forward(self, student_output, teacher_output, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        self.update_center(teacher_output)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True) / len(teacher_output)

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)
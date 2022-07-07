import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from AFSD.common.thumos_dataset import THUMOS_Dataset, get_video_info, \
    load_video_data, detection_collate, get_video_anno
from torch.utils.data import DataLoader
from AFSD.thumos14.multisegment_loss import MultiSegmentLoss
from AFSD.common.config import config
import numpy as np
import tqdm
import json
from AFSD.common import videotransforms
from AFSD.common.thumos_dataset import get_class_index_map
from AFSD.thumos14.BDNet import BDNet
from AFSD.common.segment_utils import softnms_v2
import argparse
from AFSD.evaluation.eval_detection import ANETdetection
import os
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
batch_size = config['training']['batch_size']
learning_rate = config['training']['learning_rate']
weight_decay = config['training']['weight_decay']
max_epoch = config['training']['max_epoch']
num_classes = config['dataset']['num_classes']
checkpoint_path = config['training']['checkpoint_path']
focal_loss = config['training']['focal_loss']
random_seed = config['training']['random_seed']
ngpu = config['ngpu']
os.environ["CUDA_VISIBLE_DEVICES"] = "6, 7"
writer = SummaryWriter("runs/scalar_example")
train_state_path = os.path.join(checkpoint_path, 'training')
if not os.path.exists(train_state_path):
    os.makedirs(train_state_path)

resume = config['training']['resume']

# getting path for fusion
rgb_data_path = config['testing'].get('rgb_data_path',
                                      '../../datasets/thumos14/test_npy/')
flow_data_path = config['testing'].get('flow_data_path',
                                       '../../datasets/thumos14/test_flow_npy/')
rgb_checkpoint_path = config['testing'].get('rgb_checkpoint_path',
                                            '../../models/thumos14/checkpoint-15.ckpt')
flow_checkpoint_path = config['testing'].get('flow_checkpoint_path',
                                             '../../models/thumos14_flow/checkpoint-16.ckpt')

def print_training_info():
    print('batch size: ', batch_size)
    print('learning rate: ', learning_rate)
    print('weight decay: ', weight_decay)
    print('max epoch: ', max_epoch)
    print('checkpoint path: ', checkpoint_path)
    print('loc weight: ', config['training']['lw'])
    print('cls weight: ', config['training']['cw'])
    print('ssl weight: ', config['training']['ssl'])
    print('piou:', config['training']['piou'])
    print('resume: ', resume)
    print('gpu num: ', ngpu)


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


GLOBAL_SEED = 1


def worker_init_fn(worker_id):
    set_seed(GLOBAL_SEED + worker_id)


def get_rng_states():
    states = []
    states.append(random.getstate())
    states.append(np.random.get_state())
    states.append(torch.get_rng_state())
    if torch.cuda.is_available():
        states.append(torch.cuda.get_rng_state())
    return states


def set_rng_state(states):
    random.setstate(states[0])
    np.random.set_state(states[1])
    torch.set_rng_state(states[2])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state(states[3])


def save_model(epoch, model, optimizer):
    # torch.save(model.module.state_dict(),
    #            os.path.join(checkpoint_path, 'checkpoint-{}.ckpt'.format(epoch)))
    torch.save(model.state_dict(),
               os.path.join(checkpoint_path, 'checkpoint-{}.ckpt'.format(epoch)))
    torch.save(model.state_dict(),
               os.path.join(checkpoint_path, 'checkpoint-{}.ckpt'.format(epoch)))
    torch.save({'optimizer': optimizer.state_dict(),
                'state': get_rng_states()},
               os.path.join(train_state_path, 'checkpoint_{}.ckpt'.format(epoch)))


def resume_training(resume, model, optimizer):
    start_epoch = 1
    if resume > 0:
        start_epoch += resume
        model_path = os.path.join(checkpoint_path, 'checkpoint-{}.ckpt'.format(resume))
        # model.module.load_state_dict(torch.load(model_path))
        model.load_state_dict(torch.load(model_path))
        train_path = os.path.join(train_state_path, 'checkpoint_{}.ckpt'.format(resume))
        state_dict = torch.load(train_path)
        optimizer.load_state_dict(state_dict['optimizer'])
        set_rng_state(state_dict['state'])
    return start_epoch


def calc_bce_loss(start, end, scores):
    start = torch.tanh(start).mean(-1)
    end = torch.tanh(end).mean(-1)
    loss_start = F.binary_cross_entropy(start.view(-1),
                                        scores[:, 0].contiguous().view(-1).cuda(),
                                        reduction='mean')
    loss_end = F.binary_cross_entropy(end.view(-1),
                                      scores[:, 1].contiguous().view(-1).cuda(),
                                      reduction='mean')
    return loss_start, loss_end


def forward_one_epoch(net, clips, targets, scores=None, training=True, ssl=False):
    clips = clips.cuda()
    targets = [t.cuda() for t in targets]
    if training:
        if ssl:
            # output_dict = net.module(clips, proposals=targets, ssl=ssl)
            output_dict = net(clips, proposals=targets, ssl=ssl)
        else:
            output_dict = net(clips, ssl=False)
    else:
        with torch.no_grad():
            output_dict = net(clips)

    if ssl:
        anchor, positive, negative = output_dict
        loss_ = []
        weights = [1, 0.1, 0.1]
        for i in range(3):
            loss_.append(nn.TripletMarginLoss()(anchor[i], positive[i], negative[i]) * weights[i])
        trip_loss = torch.stack(loss_).sum(0)
        return trip_loss
    else:
        loss_l, loss_c, loss_prop_l, loss_prop_c, loss_ct = CPD_Loss(
            [output_dict['loc'], output_dict['conf'],
             output_dict['prop_loc'], output_dict['prop_conf'],
             output_dict['center'], output_dict['priors'][0]],
            targets)
        loss_start, loss_end = calc_bce_loss(output_dict['start'], output_dict['end'], scores)
        scores_ = F.interpolate(scores, scale_factor=1.0 / 4)
        loss_start_loc_prop, loss_end_loc_prop = calc_bce_loss(output_dict['start_loc_prop'],
                                                               output_dict['end_loc_prop'],
                                                               scores_)
        loss_start_conf_prop, loss_end_conf_prop = calc_bce_loss(output_dict['start_conf_prop'],
                                                                 output_dict['end_conf_prop'],
                                                                 scores_)
        loss_start = loss_start + 0.1 * (loss_start_loc_prop + loss_start_conf_prop)
        loss_end = loss_end + 0.1 * (loss_end_loc_prop + loss_end_conf_prop)
        return loss_l, loss_c, loss_prop_l, loss_prop_c, loss_ct, loss_start, loss_end


def run_one_epoch(epoch, net, optimizer, data_loader, epoch_step_num, training=True):
    if training:
        net.train()
    else:
        net.eval()

    loss_loc_val = 0
    loss_conf_val = 0
    loss_prop_l_val = 0
    loss_prop_c_val = 0
    loss_ct_val = 0
    loss_start_val = 0
    loss_end_val = 0
    loss_trip_val = 0
    loss_contras_val = 0
    cost_val = 0
    with tqdm.tqdm(data_loader, total=epoch_step_num, ncols=0) as pbar:
        for n_iter, (clips, targets, scores, ssl_clips, ssl_targets, flags) in enumerate(pbar):
            loss_l, loss_c, loss_prop_l, loss_prop_c, \
            loss_ct, loss_start, loss_end = forward_one_epoch(
                net, clips, targets, scores, training=training, ssl=False)
            loss_l = loss_l * config['training']['lw']
            loss_c = loss_c * config['training']['cw']
            loss_prop_l = loss_prop_l * config['training']['lw']
            loss_prop_c = loss_prop_c * config['training']['cw']
            loss_ct = loss_ct * config['training']['cw']
            cost = loss_l + loss_c + loss_prop_l + loss_prop_c + loss_ct + loss_start + loss_end

            ssl_count = 0
            loss_trip = 0
            for i in range(len(flags)):
                if flags[i] and config['training']['ssl'] > 0:
                    loss_trip += forward_one_epoch(net, ssl_clips[i].unsqueeze(0), [ssl_targets[i]],
                                                   training=training, ssl=True) * config['training']['ssl']
                    loss_trip_val += loss_trip.cpu().detach().numpy()
                    ssl_count += 1
            if ssl_count:
                loss_trip_val /= ssl_count
                loss_trip /= ssl_count
            cost = cost + loss_trip
            if training:
                optimizer.zero_grad()
                cost.backward()
                optimizer.step()
            loss_loc_val += loss_l.cpu().detach().numpy()
            loss_conf_val += loss_c.cpu().detach().numpy()
            loss_prop_l_val += loss_prop_l.cpu().detach().numpy()
            loss_prop_c_val += loss_prop_c.cpu().detach().numpy()
            loss_ct_val += loss_ct.cpu().detach().numpy()
            loss_start_val += loss_start.cpu().detach().numpy()
            loss_end_val += loss_end.cpu().detach().numpy()
            cost_val += cost.cpu().detach().numpy()
            pbar.set_postfix(loss='{:.5f}'.format(float(cost.cpu().detach().numpy())))

    loss_loc_val /= (n_iter + 1)
    loss_conf_val /= (n_iter + 1)
    loss_prop_l_val /= (n_iter + 1)
    loss_prop_c_val /= (n_iter + 1)
    loss_ct_val /= (n_iter + 1)
    loss_start_val /= (n_iter + 1)
    loss_end_val /= (n_iter + 1)
    loss_trip_val /= (n_iter + 1)
    cost_val /= (n_iter + 1)

    if training:
        prefix = 'Train'
        save_model(epoch, net, optimizer)
    else:
        prefix = 'Val'

    writer.add_scalar('Total Loss', cost_val, epoch)
    writer.add_scalar('loc', loss_loc_val, epoch)
    writer.add_scalar('conf', loss_conf_val, epoch)



if __name__ == '__main__':
    print_training_info()
    set_seed(random_seed)
    ngpu = 3
    """
    Setup model
    """
    net = BDNet(in_channels=1,
                backbone_model=config['model']['backbone_model'],
                training=True)
    # for para in net.backbone.parameters():
    #     para.requires_grad = False

    net = nn.DataParallel(net, device_ids=[0, 1]).cuda()
    # net = net.cuda()

    # for k, v in net.named_parameters():
    #     print('{}: {}'.format(k, v.requires_grad))

    """
    Setup optimizer
    """
    optimizer = torch.optim.Adam(net.parameters(),
                                 lr=learning_rate,
                                 weight_decay=weight_decay)
    """
    Setup loss
    """
    piou = config['training']['piou']
    CPD_Loss = MultiSegmentLoss(num_classes, piou, 1.0, use_focal_loss=focal_loss)

    """
    Setup dataloader
    """
    train_video_infos = get_video_info(config['dataset']['training']['video_info_path'])
    train_video_annos = get_video_anno(train_video_infos,
                                       config['dataset']['training']['video_anno_path'])
    train_data_dict = load_video_data(train_video_infos,
                                      config['dataset']['training']['video_data_path'])
    train_dataset = THUMOS_Dataset(train_data_dict,
                                   train_video_infos,
                                   train_video_annos)
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                   num_workers=4, worker_init_fn=worker_init_fn,
                                   collate_fn=detection_collate, pin_memory=True, drop_last=True)
    epoch_step_num = len(train_dataset) // batch_size

    """
    Start training
    """
    start_epoch = resume_training(resume, net, optimizer)

    for i in range(start_epoch, max_epoch + 1):
        run_one_epoch(i, net, optimizer, train_data_loader, len(train_dataset) // batch_size)


    """
    Start testing
    """
    num_classes = config['dataset']['num_classes']
    conf_thresh = config['testing']['conf_thresh']
    top_k = config['testing']['top_k']
    nms_thresh = config['testing']['nms_thresh']
    nms_sigma = config['testing']['nms_sigma']
    clip_length = config['dataset']['testing']['clip_length']
    stride = config['dataset']['testing']['clip_stride']
    max_epoch = config['training']['max_epoch']
    checkpoint_path = config['testing']['checkpoint_path']
    json_name = config['testing']['output_json']
    output_path = config['testing']['output_path']
    softmax_func = True
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    fusion = config['testing']['fusion']

    for i in range(1, max_epoch+1):
        print(i)
        checkpoint_path = "./models/thumos14/checkpoint-"+str(i)+".ckpt"
        json_name = str(i)+".json"
        video_infos = get_video_info(config['dataset']['testing']['video_info_path'])
        originidx_to_idx, idx_to_class = get_class_index_map()
        npy_data_path = config['dataset']['testing']['video_data_path']

        net = BDNet(in_channels=config['model']['in_channels'],
                    backbone_model=config['model']['backbone_model'],
                    training=True)
        net = nn.DataParallel(net, device_ids=[0, 1]).cuda()
        # net = net.cuda()
        net.load_state_dict(torch.load(checkpoint_path))
        net.eval().cuda()

        if softmax_func:
            score_func = nn.Softmax(dim=-1)
        else:
            score_func = nn.Sigmoid()

        centor_crop = videotransforms.CenterCrop(config['dataset']['testing']['crop_size'])

        result_dict = {}
        for video_name in tqdm.tqdm(list(video_infos.keys()), ncols=0):
            sample_count = video_infos[video_name]['sample_count']
            sample_fps = video_infos[video_name]['sample_fps']
            if sample_count < clip_length:
                offsetlist = [0]
            else:
                offsetlist = list(range(0, sample_count - clip_length + 1, stride))
                if (sample_count - clip_length) % stride:
                    offsetlist += [sample_count - clip_length]

            data = np.load(os.path.join(npy_data_path, video_name + '.npy'))
            data = np.expand_dims(data, 0).repeat(1, axis=0)
            data = torch.from_numpy(data).float().unsqueeze(2)  # 1*8500*1*30
            data = data.view(1, 340, 25, 30)

            output = []
            for cl in range(num_classes):
                output.append([])
            res = torch.zeros(num_classes, top_k, 3)

            # print(video_name)
            for offset in offsetlist:
                clip = data[:, offset: offset + clip_length]
                clip = clip.float()
                clip = (clip / 255.0) * 2.0 - 1.0
                if clip.size(1) < clip_length:
                    tmp = torch.zeros([clip.size(0), clip_length - clip.size(1),
                                       96, 96]).float()
                    clip = torch.cat([clip, tmp], dim=1)
                clip = clip.unsqueeze(0).cuda()
                with torch.no_grad():
                    output_dict = net(clip)

                loc, conf, priors = output_dict['loc'], output_dict['conf'], output_dict['priors'][0]
                prop_loc, prop_conf = output_dict['prop_loc'], output_dict['prop_conf']
                center = output_dict['center']
                loc = loc[0]
                conf = conf[0]
                prop_loc = prop_loc[0]
                prop_conf = prop_conf[0]
                center = center[0]

                pre_loc_w = loc[:, :1] + loc[:, 1:]
                loc = 0.5 * pre_loc_w * prop_loc + loc
                decoded_segments = torch.cat(
                    [priors[:, :1] * clip_length - loc[:, :1],
                     priors[:, :1] * clip_length + loc[:, 1:]], dim=-1)
                decoded_segments.clamp_(min=0, max=clip_length)

                conf = score_func(conf)
                prop_conf = score_func(prop_conf)
                center = center.sigmoid()

                conf = (conf + prop_conf) / 2.0
                conf = conf * center
                conf = conf.view(-1, num_classes).transpose(1, 0)
                conf_scores = conf.clone()

                for cl in range(1, num_classes):
                    c_mask = conf_scores[cl] > conf_thresh
                    scores = conf_scores[cl][c_mask]
                    if scores.size(0) == 0:
                        continue
                    l_mask = c_mask.unsqueeze(1).expand_as(decoded_segments)
                    segments = decoded_segments[l_mask].view(-1, 2)
                    # decode to original time
                    # segments = (segments * clip_length + offset) / sample_fps
                    segments = (segments + offset) / sample_fps
                    segments = torch.cat([segments, scores.unsqueeze(1)], -1)

                    output[cl].append(segments)
                    # np.set_printoptions(precision=3, suppress=True)

            sum_count = 0
            for cl in range(1, num_classes):
                if len(output[cl]) == 0:
                    continue
                tmp = torch.cat(output[cl], 0)
                tmp, count = softnms_v2(tmp, sigma=nms_sigma, top_k=top_k)
                res[cl, :count] = tmp
                sum_count += count

            sum_count = min(sum_count, top_k)
            flt = res.contiguous().view(-1, 3)
            flt = flt.view(num_classes, -1, 3)
            proposal_list = []
            for cl in range(1, num_classes):
                class_name = idx_to_class[cl]
                tmp = flt[cl].contiguous()
                tmp = tmp[(tmp[:, 2] > 0).unsqueeze(-1).expand_as(tmp)].view(-1, 3)
                if tmp.size(0) == 0:
                    continue
                tmp = tmp.detach().cpu().numpy()
                for i in range(tmp.shape[0]):
                    tmp_proposal = {}
                    tmp_proposal['label'] = class_name
                    tmp_proposal['score'] = float(tmp[i, 2])
                    tmp_proposal['segment'] = [float(tmp[i, 0]),
                                               float(tmp[i, 1])]
                    proposal_list.append(tmp_proposal)

            result_dict[video_name] = proposal_list

        output_dict = {"version": "THUMOS14", "results": dict(result_dict), "external_data": {}}

        with open(os.path.join(output_path, json_name), "w") as out:
            json.dump(output_dict, out)
    writer.close()

    """
    Start evaluating
    """
    writer = SummaryWriter("runs/eval")
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    max_epoch = config['training']['max_epoch']
    max = 0
    max_i = 0
    x = []
    y = []
    for i in range(1, max_epoch + 1):
        gt_json = '../../thumos_annotations/thumos_gt.json'
        output_json = './output/' + str(i) + ".json"
        tious = [0.3, 0.4, 0.5, 0.6, 0.7]
        anet_detection = ANETdetection(
            ground_truth_filename=gt_json,
            prediction_filename=output_json,
            subset='test', tiou_thresholds=tious)
        mAPs, average_mAP, ap = anet_detection.evaluate()
        # print(mAPs, "\n", average_mAP, "\n", ap)
        print("epoch", i)
        for (tiou, mAP) in zip(tious, mAPs):
            print("mAP at tIoU {} is {}".format(tiou, mAP))
        print(average_mAP, "\n")

        if average_mAP > max:
            max = average_mAP
            max_i = i

        writer.add_scalar('average_mAP', round(average_mAP, 4), i)
        x.append(i)
        y.append(round(average_mAP, 4))

    plt.plot(x, y)
    plt.xlabel("epoch")
    plt.ylabel("average mAP")
    plt.show()


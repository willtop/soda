import argparse
import os
from functools import partial

import numpy as np
import torch
import torch.distributed as dist
import yaml
from metric import KNN, LinearProbe
from torchvision.utils import make_grid, save_image
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from ema_pytorch import EMA

from model.SODA import SODA
from model.encoder import Network
from model.decoder import UNet_decoder
from utils import Config, get_optimizer, init_seeds, reduce_tensor, DataLoaderDDP, print0
from datasets import get_dataset


def train(opt):
    yaml_path = opt.config
    local_rank = opt.local_rank
    use_amp = opt.use_amp
    whether_latent_traverse = opt.latent_traverse

    with open(yaml_path, 'r') as f:
        opt = yaml.full_load(f)
    print0(opt)
    opt = Config(opt)
    model_dir = os.path.join(opt.save_dir, "ckpts")
    vis_dir = os.path.join(opt.save_dir, "visual")
    tsbd_dir = os.path.join(opt.save_dir, "tensorboard")
    if local_rank == 0:
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(vis_dir, exist_ok=True)

    device = "cuda:%d" % local_rank
    soda = SODA(encoder=Network(**opt.encoder),
                decoder=UNet_decoder(**opt.decoder),
                **opt.diffusion,
                device=device)
    soda.to(device)
    if local_rank == 0:
        ema = EMA(soda, beta=opt.ema, update_after_step=0, update_every=1)
        ema.to(device)
        ema.eval()
        writer = SummaryWriter(log_dir=tsbd_dir)

    soda = torch.nn.SyncBatchNorm.convert_sync_batchnorm(soda)
    soda = torch.nn.parallel.DistributedDataParallel(
        soda, device_ids=[local_rank], output_device=local_rank)

    num_classes, train, down_train, down_test = get_dataset(name=opt.dataset, root="./data")

    # cluster configs
    if local_rank == 0:
        knn = KNN(down_train, down_test)
        lp = LinearProbe(down_train, down_test, num_classes)

    train_loader, sampler = DataLoaderDDP(train,
                                          batch_size=opt.batch_size,
                                          shuffle=True)

    lr = opt.lrate
    DDP_multiplier = dist.get_world_size()
    print0("Using DDP, lr = %f * %d" % (lr, DDP_multiplier))
    lr *= DDP_multiplier
    optim = get_optimizer([{'params': soda.module.encoder.parameters(), 'lr': lr * opt.lrate_ratio},
                           {'params': soda.module.decoder.parameters(), 'lr': lr}], opt, lr=0)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # add in latent traverse visualization
    if whether_latent_traverse:
        # if finished whole training process
        # latest_epoch = opt.n_epoch
        latest_epoch = 60
        target = os.path.join(model_dir, f"model_{latest_epoch}.pth")
        print0("loading model at", target)
        checkpoint = torch.load(target, map_location=device)
        soda.load_state_dict(checkpoint['MODEL'])
        if local_rank == 0:
            ema.load_state_dict(checkpoint['EMA'])
        ema_sample_method = ema.ema_model.ddim_sample
        ema.ema_model.eval()
        # randomly select two samples: source image and target image
        start_img_id, end_img_id = np.random.choice(len(train), size=2, replace=False)
        start_img, _ = train[start_img_id]
        end_img, _ =  train[end_img_id]
        with torch.no_grad():
            start_z_guide = ema.ema_model.encode(start_img.unsqueeze(0).to(device), norm=False, use_amp=use_amp)
            end_z_guide = ema.ema_model.encode(end_img.unsqueeze(0).to(device), norm=False, use_amp=use_amp)
        assert start_z_guide.shape == end_z_guide.shape == (1, opt.encoder['feature_dim'])
        start_z_guide, end_z_guide = start_z_guide.cpu().numpy().squeeze(), end_z_guide.cpu().numpy().squeeze()
        # making a batch of z values, with each component changing gradually from start latent to end latent
        latent_dims_to_traverse = np.random.choice(opt.encoder['feature_dim'], size=10, replace=False)
        all_z_guides = []
        for latent_dim in latent_dims_to_traverse:
            latent_val_list = np.linspace(start=start_z_guide[latent_dim], 
                                          stop=end_z_guide[latent_dim],
                                          num=20,
                                          endpoint=True)
            for latent_val in latent_val_list:
                z_guide = start_z_guide.copy()
                z_guide[latent_dim] = latent_val
                all_z_guides.append(z_guide)
        all_z_guides = np.stack(all_z_guides, axis=0)
        assert np.shape(all_z_guides) == (200, opt.encoder['feature_dim'])       

        with torch.no_grad():
            x_gen = ema_sample_method(200, start_img.shape, torch.tensor(all_z_guides).to(device))
        grid = make_grid(x_gen.cpu(), nrow=20)
        save_path = os.path.join(vis_dir, f"latent_traverse_ep{latest_epoch}_ema.png")
        save_image(grid, save_path)

        # save start and end image
        x_orig = torch.stack([start_img, end_img])
        grid = make_grid(x_orig, nrow=2)
        save_image(grid, os.path.join(vis_dir, "latent_traverse_start_finish.png"))
        print('latent_traverse_finished')
        return 


    if opt.load_epoch != -1:
        target = os.path.join(model_dir, f"model_{opt.load_epoch}.pth")
        print0("loading model at", target)
        checkpoint = torch.load(target, map_location=device)
        soda.load_state_dict(checkpoint['MODEL'])
        if local_rank == 0:
            ema.load_state_dict(checkpoint['EMA'])
        optim.load_state_dict(checkpoint['opt'])

    for ep in range(opt.load_epoch + 1, opt.n_epoch):
        optim.param_groups[1]['lr'] = lr * min((ep + 1.0) / opt.warm_epoch, 1.0) # warmup
        optim.param_groups[0]['lr'] = optim.param_groups[1]['lr'] * opt.lrate_ratio
        sampler.set_epoch(ep)
        dist.barrier()
        # training
        soda.train()
        if local_rank == 0:
            enc_lr = optim.param_groups[0]['lr']
            dec_lr = optim.param_groups[1]['lr']
            print(f'epoch {ep}, lr {enc_lr:f} & {dec_lr:f}')
            loss_ema = None
            pbar = tqdm(train_loader)
        else:
            pbar = train_loader
        for source, target in pbar:
            optim.zero_grad()
            source = source.to(device)
            target = target.to(device)
            loss = soda(source, target, use_amp=use_amp)
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(parameters=soda.parameters(), max_norm=opt.grad_clip_norm)
            scaler.step(optim)
            scaler.update()

            # logging
            dist.barrier()
            loss = reduce_tensor(loss)
            if local_rank == 0:
                ema.update()
                if loss_ema is None:
                    loss_ema = loss.item()
                else:
                    loss_ema = 0.95 * loss_ema + 0.05 * loss.item()
                pbar.set_description(f"loss: {loss_ema:.4f}")

        # testing
        if local_rank == 0:
            writer.add_scalar('lr/enc', enc_lr, ep)
            writer.add_scalar('lr/dec', dec_lr, ep)
            writer.add_scalar('loss', loss_ema, ep)
            if (opt.save_per != 0 and ep % opt.save_per == 0) or ep == opt.n_epoch - 1:
                pass
            else:
                continue

            if ep > 0 and ep % opt.eval_per == 0:
                print(f'epoch {ep}, evaluating:')
                soda.eval()
                feat_func = partial(ema.ema_model.encode, norm=True, use_amp=use_amp)
                test_knn = knn.evaluate(feat_func)
                test_lp = lp.evaluate(feat_func)
                writer.add_scalar('metrics/K Nearest Neighbors', test_knn, ep)
                writer.add_scalar('metrics/Linear Probe', test_lp, ep)

            ema_sample_method = ema.ema_model.ddim_sample
            ema.ema_model.eval()
            with torch.no_grad():
                z_guide = ema.ema_model.encode(source[:opt.n_sample], norm=False, use_amp=use_amp)
                x_gen = ema_sample_method(opt.n_sample, target.shape[1:], z_guide)
            # save an image of currently generated samples (top rows)
            # followed by real images (bottom rows)
            x_real = target[:opt.n_sample]
            x_all = torch.cat([x_gen.cpu(), x_real.cpu()])
            grid = make_grid(x_all, nrow=10)

            save_path = os.path.join(vis_dir, f"image_ep{ep}_ema.png")
            save_image(grid, save_path)
            print('saved image at', save_path)

            checkpoint = {
                'MODEL': soda.state_dict(),
                'EMA': ema.state_dict(),
                'opt': optim.state_dict(),
            }
            save_path = os.path.join(model_dir, f"model_{ep}.pth")
            torch.save(checkpoint, save_path)
            print('saved model at', save_path)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    # originally had default value as -1, however we should always have a GPU ready
    parser.add_argument('--local_rank', default=0, type=int,
                        help='node rank for distributed training')
    parser.add_argument("--use_amp", action='store_true', default=False)
    parser.add_argument("--latent_traverse", action='store_true', default=False)
    opt = parser.parse_args()
    print0(opt)

    init_seeds(no=opt.local_rank)
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(opt.local_rank)
    train(opt)

    print("Main script finished!")

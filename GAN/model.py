import torch.nn as nn
import torch
import itertools
import time
from tqdm.notebook import tqdm
import json
import random
from torchvision.utils import save_image

from blocks import Generator, Discriminator
from utils import sample_fake, save_checkpoint, AvgStats, init_weights, update_req_grad, unnorm
from visdom_server.visdom_server import Visualizer


class CycleGAN(object):
    def __init__(self, in_ch, out_ch, epochs, device, img_ds, start_lr=2e-4, lmbda=10, idt_coef=0.5, cycle_coef=1,
                 decay_epoch=0, num_res_blocks=6):
        self.epochs = epochs
        self.decay_epoch = decay_epoch if decay_epoch > 0 else int(self.epochs / 2)
        self.lmbda = lmbda
        self.idt_coef = idt_coef
        self.cycle_coef = cycle_coef
        self.device = device
        self.img_ds = img_ds
        self.gen_mtp = Generator(in_ch, out_ch, num_res_blocks=num_res_blocks)
        self.gen_ptm = Generator(in_ch, out_ch, num_res_blocks=num_res_blocks)
        self.disc_m = Discriminator(in_ch)
        self.disc_p = Discriminator(in_ch)
        self.init_models()
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        self.adam_gen = torch.optim.Adam(itertools.chain(self.gen_mtp.parameters(), self.gen_ptm.parameters()),
                                         lr=start_lr, betas=(0.5, 0.999))
        self.adam_disc = torch.optim.Adam(itertools.chain(self.disc_m.parameters(), self.disc_p.parameters()),
                                          lr=start_lr, betas=(0.5, 0.999))

        self.gen_lr_sched = torch.optim.lr_scheduler.CyclicLR(self.adam_gen, base_lr=start_lr, max_lr=5 * start_lr,
                                                              cycle_momentum=False)
        self.disc_lr_sched = torch.optim.lr_scheduler.CyclicLR(self.adam_disc, base_lr=start_lr, max_lr=5 * start_lr,
                                                               cycle_momentum=False)

        self.sample_monet = sample_fake()
        self.sample_photo = sample_fake()
        self.gen_stats = AvgStats()
        self.disc_stats = AvgStats()
        self.vis = Visualizer()

    def init_models(self):
        init_weights(self.gen_mtp)
        init_weights(self.gen_ptm)
        init_weights(self.disc_m)
        init_weights(self.disc_p)
        self.gen_mtp = self.gen_mtp.to(self.device)
        self.gen_ptm = self.gen_ptm.to(self.device)
        self.disc_m = self.disc_m.to(self.device)
        self.disc_p = self.disc_p.to(self.device)

    def train(self, photo_dl):
        for epoch in range(self.epochs):
            start_time = time.time()
            avg_gen_loss = 0.0
            avg_disc_loss = 0.0
            t = tqdm(photo_dl, leave=False, total=photo_dl.__len__())
            for i, (photo_real, monet_real) in enumerate(t):
                photo_img, monet_img = photo_real.to(self.device), monet_real.to(self.device)
                update_req_grad([self.disc_m, self.disc_p], False)
                self.adam_gen.zero_grad()

                # Forward pass through generator
                fake_photo = self.gen_mtp(monet_img)
                fake_monet = self.gen_ptm(photo_img)

                cycl_monet = self.gen_ptm(fake_photo)
                cycl_photo = self.gen_mtp(fake_monet)

                id_monet = self.gen_ptm(monet_img)
                id_photo = self.gen_mtp(photo_img)

                # generator losses - identity, Adversarial, cycle consistency
                idt_loss_monet = self.l1_loss(id_monet, monet_img) * self.lmbda * self.idt_coef
                idt_loss_photo = self.l1_loss(id_photo, photo_img) * self.lmbda * self.idt_coef

                cycle_loss_monet = self.l1_loss(cycl_monet, monet_img) * self.lmbda * self.cycle_coef
                cycle_loss_photo = self.l1_loss(cycl_photo, photo_img) * self.lmbda * self.cycle_coef

                monet_disc = self.disc_m(fake_monet)
                photo_disc = self.disc_p(fake_photo)

                real = torch.ones(monet_disc.size()).to(self.device)

                adv_loss_monet = self.mse_loss(monet_disc, real)
                adv_loss_photo = self.mse_loss(photo_disc, real)

                # total generator loss
                total_gen_loss = cycle_loss_monet + adv_loss_monet \
                                 + cycle_loss_photo + adv_loss_photo \
                                 + idt_loss_monet + idt_loss_photo

                avg_gen_loss += total_gen_loss.item()

                # backward pass
                total_gen_loss.backward()
                self.adam_gen.step()

                # Forward pass through Discriminator
                update_req_grad([self.disc_m, self.disc_p], True)
                self.adam_disc.zero_grad()

                fake_monet = self.sample_monet([fake_monet.cpu().data.numpy()])[0]
                fake_photo = self.sample_photo([fake_photo.cpu().data.numpy()])[0]
                fake_monet = torch.tensor(fake_monet).to(self.device)
                fake_photo = torch.tensor(fake_photo).to(self.device)

                monet_disc_real = self.disc_m(monet_img)
                monet_disc_fake = self.disc_m(fake_monet)
                photo_disc_real = self.disc_p(photo_img)
                photo_disc_fake = self.disc_p(fake_photo)

                real = torch.ones(monet_disc_real.size()).to(self.device)
                fake = torch.ones(monet_disc_fake.size()).to(self.device)

                # Discriminator losses
                # --------------------
                monet_disc_real_loss = self.mse_loss(monet_disc_real, real)
                monet_disc_fake_loss = self.mse_loss(monet_disc_fake, fake)
                photo_disc_real_loss = self.mse_loss(photo_disc_real, real)
                photo_disc_fake_loss = self.mse_loss(photo_disc_fake, fake)

                monet_disc_loss = (monet_disc_real_loss + monet_disc_fake_loss) / 2
                photo_disc_loss = (photo_disc_real_loss + photo_disc_fake_loss) / 2
                total_disc_loss = monet_disc_loss + photo_disc_loss
                avg_disc_loss += total_disc_loss.item()

                # Backward
                monet_disc_loss.backward()
                photo_disc_loss.backward()
                self.adam_disc.step()

                t.set_postfix(gen_loss=total_gen_loss.item(), disc_loss=total_disc_loss.item())

                # self.gen_lr_sched.step()
                # self.disc_lr_sched.step()

            # Save images
            idx = random.randrange(0, self.img_ds.__len__())
            photo_img, monet_img = self.img_ds.__getitem__(idx)
            photo_img, monet_img = photo_img.to(self.device), monet_img.to(self.device)
            photo_img = torch.unsqueeze(photo_img, 0)
            monet_img = torch.unsqueeze(monet_img, 0)

            fake_photo = self.gen_mtp(monet_img)
            fake_monet = self.gen_ptm(photo_img)

            cycl_monet = self.gen_ptm(fake_photo)
            cycl_photo = self.gen_mtp(fake_monet)

            id_monet = self.gen_ptm(monet_img)
            id_photo = self.gen_mtp(photo_img)

            fake_photo = unnorm(fake_photo.cpu().detach())[0]
            fake_monet = unnorm(fake_monet.cpu().detach())[0]

            cycl_photo = unnorm(cycl_photo.cpu().detach())[0]
            cycl_monet = unnorm(cycl_monet.cpu().detach())[0]

            id_photo = unnorm(id_photo.cpu().detach())[0]
            id_monet = unnorm(id_monet.cpu().detach())[0]

            photo_img = unnorm(photo_img.cpu().detach())[0]
            monet_img = unnorm(monet_img.cpu().detach())[0]

            save_image(fake_photo, "data/for_dash/fake_photo.jpg")
            save_image(fake_monet, "data/for_dash/fake_monet.jpg")

            save_image(cycl_photo, "data/for_dash/cycl_photo.jpg")
            save_image(cycl_monet, "data/for_dash/cycl_monet.jpg")

            save_image(id_photo, "data/for_dash/id_photo.jpg")
            save_image(id_monet, "data/for_dash/id_monet.jpg")

            save_image(photo_img, "data/for_dash/photo.jpg")
            save_image(monet_img, "data/for_dash/monet.jpg")

            avg_gen_loss /= photo_dl.__len__()
            avg_disc_loss /= photo_dl.__len__()
            time_req = time.time() - start_time

            self.gen_stats.append(avg_gen_loss, time_req)
            self.disc_stats.append(avg_disc_loss, time_req)

            with open('data/for_dash/gen_loss.json', 'r') as file:
                gen_loss = json.load(file)

            save_dict = {
                'epoch': epoch + 1,
                'gen_mtp': self.gen_mtp.state_dict(),
                'gen_ptm': self.gen_ptm.state_dict(),
                'disc_m': self.disc_m.state_dict(),
                'disc_p': self.disc_p.state_dict(),
                'optimizer_gen': self.adam_gen.state_dict(),
                'optimizer_disc': self.adam_disc.state_dict(),
                'gen_stats': self.gen_stats.state_dict(),
                'disc_stats': self.disc_stats.state_dict()
            }
            if gen_loss[-1] > avg_gen_loss:
                save_checkpoint(save_dict, 'best_checkpoint.ckpt')
            else:
                save_checkpoint(save_dict, 'current_checkpoint.ckpt')

            print("Epoch: (%d) | Generator Loss:%f | Discriminator Loss:%f" %
                  (epoch + 1, avg_gen_loss, avg_disc_loss))

            # Visualize
            self.vis.display_current_results(self.gen_stats.losses, self.disc_stats.losses)

    def load(self, cnt):
        self.gen_mtp.load_state_dict(cnt['gen_mtp'])
        self.gen_ptm.load_state_dict(cnt['gen_ptm'])
        self.disc_m.load_state_dict(cnt['disc_m'])
        self.disc_p.load_state_dict(cnt['disc_p'])
        self.adam_gen.load_state_dict(cnt['optimizer_gen'])
        self.adam_disc.load_state_dict(cnt['optimizer_disc'])
        self.gen_stats.load_state_dict(cnt['gen_stats'])
        self.disc_stats.load_state_dict(cnt['disc_stats'])
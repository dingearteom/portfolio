import visdom
from PIL import Image
import torchvision.transforms as transforms
import json
import torch
import numpy as np


class Visualizer:
    def __init__(self):
        self.vis = visdom.Visdom()

    def display_current_results(self, gen_loss, disc_loss):
        self.vis.close(env='main', win=None)
        self.plot_losses(gen_loss, disc_loss)

        images = []
        file_name_images = ['monet.jpg', 'id_monet.jpg', 'fake_photo.jpg', 'cycl_monet.jpg',
                            'photo.jpg', 'id_photo.jpg', 'fake_monet.jpg', 'cycl_photo.jpg']
        for file_name in file_name_images:
            images.append(transforms.ToTensor()(Image.open(f'data/for_dash/{file_name}')).numpy())

        images = np.array(images)
        self.vis.images(images, nrow=4, padding=2)

    def plot_losses(self, gen_loss, disc_loss):
        # with open('data/for_dash/gen_loss.json', 'r') as file:
        #     gen_loss = json.load(file)
        # with open('data/for_dash/desc_loss.json', 'r') as file:
        #     desc_loss = json.load(file)

        gen_loss = torch.unsqueeze(torch.tensor(gen_loss), 0)
        disc_loss = torch.unsqueeze(torch.tensor(disc_loss), 0)

        x = torch.unsqueeze(torch.arange(1, gen_loss.shape[1] + 1), 0)

        Y = torch.cat([gen_loss, disc_loss]).T
        X = torch.cat([x, x]).T
        self.vis.line(Y=Y, X=X, opts={'legend': ['gen loss', 'disc loss']})


if __name__ == '__main__':
    vis = Visualizer()
    vis.display_current_results()
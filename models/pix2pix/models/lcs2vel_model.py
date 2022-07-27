"""Model class template

This module provides a template for users to implement custom models.
You can specify '--model template' to use this model.
The class name should be consistent with both the filename and its model option.
The filename should be <model>_dataset.py
The class name should be <Model>Dataset.py
It implements a simple image-to-image translation baseline based on regression loss.
Given input-output pairs (data_A, data_B), it learns a network netG that can minimize the following L1 loss:
    min_<netG> ||netG(data_A) - data_B||_1
You need to implement the following functions:
    <modify_commandline_options>:　Add model-specific options and rewrite default values for existing options.
    <__init__>: Initialize this model class.
    <set_input>: Unpack input data and perform data pre-processing.
    <forward>: Run forward pass. This will be called by both <optimize_parameters> and <test>.
    <optimize_parameters>: Update network weights; it will be called in every training iteration.
"""
import itertools
import torch
from torch._C import default_generator
from .base_model import BaseModel
from . import networks


class Lcs2VelModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new model-specific options and rewrite default values for existing options.

        Parameters:
            parser -- the option parser
            is_train -- if it is training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        parser.set_defaults(dataroot='../dataset')
        parser.set_defaults(input_nc=1, output_nc=2)
        parser.set_defaults(netG='unet_256', direction='AtoB')  # You can rewrite default values for this model. For example, this model usually uses aligned dataset as its dataset.
        if is_train:
            parser.add_argument('--lambda_GAN', type=float, default=1.0, help='weight for GAN')
            parser.add_argument('--lambda_L1', type=float, default=10.0, help='weight for L1')
            parser.add_argument('--lambda_Scale', type=float, default=30.0, help='weight for Scale')
            parser.add_argument('--lambda_Cross', type=float, default=30.0, help='weight for Cross')
            parser.add_argument('--lambda_Diff', type=float, default=30.0, help='weight for Diff')
            parser.add_argument('--loss_on_Lcs', type=bool, default=False, help='Loss calculation area')

        return parser

    def __init__(self, opt):
        """Initialize this model class.

        Parameters:
            opt -- training/test options

        A few things can be done here.
        - (required) call the initialization function of BaseModel
        - define loss function, visualization images, model names, and optimizers
        """
        BaseModel.__init__(self, opt)  # call the initialization method of BaseModel
        # specify the training losses you want to print out. The program will call base_model.get_current_losses to plot the losses to the console and save them to the disk.
        self.loss_names = ['G_GAN', 'G_L1', 'G_Scale', 'G_Cross', 'G_Diff', 'D_real', 'D_fake']
        # specify the images you want to save and display. The program will call base_model.get_current_visuals to save and display these images.
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks to save and load networks.
        # you can use opt.isTrain to specify different behaviors for training and test. For example, some networks will not be used during test, and you don't need to load them.
        if self.isTrain:
            self.is_loss_on_lcs = opt.loss_on_Lcs

        if self.isTrain:
            self.model_names = ['G', 'D']
        else:
            self.model_names = ['G']
        # define networks; you can use opt.isTrain to specify different behaviors for training and test.
        self.netG = self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        
        if self.isTrain:  # only defined during training time
                        # define your loss functions. You can use losses provided by torch.nn such as torch.nn.L1Loss.
            # We also provide a GANLoss class "networks.GANLoss". self.criterionGAN = networks.GANLoss().to(self.device)
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            self.Lcs2VelLoss = networks.Lcs2Vel_Loss().to(self.device)
            # define and initialize optimizers. You can define one optimizer for each network.
            # If two networks are updated at the same time, you can use itertools.chain to group them. See cycle_gan_model.py for an example.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

        # Our program will automatically call <model.setup> to define schedulers, load networks, and print networks

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input: a dictionary that contains the data itself and its metadata information.
        """
        # tmp_A = input['A']  # get image data A
        self.real_A = input['A'].to(self.device)
        # self.real_B = input['B'].to(self.device) * self.real_A
        if self.isTrain:
            self.real_B = input['B'].to(self.device)
            if self.is_loss_on_lcs:
                self.real_B = self.real_B * self.real_A
        self.label_paths = input['A_paths']  # get image paths

    def forward(self):
        """Run forward pass. This will be called by both functions <optimize_parameters> and <test>."""
        # self.fake_B = self.netG(self.real_A) * self.real_A
        self.fake_B = self.netG(self.real_A)

    def backward_D(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # Cat image and label
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        real_AB = torch.cat((self.real_A, self.real_B), 1)

        pred_fake = self.netD(fake_AB.detach())
        pred_real = self.netD(real_AB)

        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        # Combined loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self):
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)

        self.loss_G_GAN = self.criterionGAN(self.netD(fake_AB), True) * self.opt.lambda_GAN
        # self.loss_G_L1 = self.criterionLoss(self.real_B, self.fake_B)
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1
        loss_Scale, loss_Cross, loss_Diff = self.Lcs2VelLoss(self.real_B, self.fake_B)
        self.loss_G_Scale = loss_Scale * self.opt.lambda_Scale
        self.loss_G_Cross = loss_Cross * self.opt.lambda_Cross
        self.loss_G_Diff = loss_Diff * self.opt.lambda_Diff
        self.loss_G = self.loss_G_GAN + self.loss_G_L1 + self.loss_G_Scale + self.loss_G_Cross + self.loss_G_Diff
        self.loss_G.backward()

    def optimize_parameters(self):
        """Update network weights; it will be called in every training iteration."""
        self.forward()               # first call forward to calculate intermediate results

        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad() # clear network G's existing gradients
        self.backward_G()              # calculate gradients for network G
        self.optimizer_G.step()        # update gradients for network G

        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

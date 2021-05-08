# -*- coding: utf-8 -*-
"""
Created on Sun May  2 08:27:12 2021

@author: Lord Tafaius
"""
import os
import uuid
import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from collections import namedtuple
from itertools import count
from PIL import Image
from time import time
from torch.autograd import Variable

import torch
import torch.nn as nn
import tenseal as ts
import torch.optim as optim
import torchvision.transforms as T

import model


class User:

    resize = T.Compose([T.Grayscale(num_output_channels=1),
                    T.ToPILImage(),
                    T.Resize(30, interpolation=Image.CUBIC),
                    T.ToTensor()])
    is_ipython = 'inline' in matplotlib.get_backend()
    if is_ipython:
            from IPython import display
    plt.ion()
    
    def __init__(self):
        self.cp=None
        self._prep_env()
        
    
    def _prep_env(self):
        self.env = gym.make('CartPole-v0').unwrapped
        bits_scale = 26
        context = ts.context(
        ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=8192,
            coeff_mod_bit_sizes=[31, bits_scale, bits_scale, bits_scale, bits_scale, bits_scale, bits_scale, 31]
        )
        context.global_scale = pow(2, bits_scale)
        context.generate_galois_keys()
        self.context = context
            
    def hook(self, platform):
        platform.client = self
        self.cp = platform
    
    def get_cart_location(self, screen_width):
        world_width = self.env.x_threshold * 2
        scale = screen_width / world_width
        return int(self.env.state[0] * scale + screen_width / 2.0)  # MIDDLE OF CART
    
    def get_screen(self):
        # Returned screen requested by gym is 400x600x3, but is sometimes larger
        # such as 800x1200x3. Transpose it into torch order (CHW).
        screen = self.env.render(mode='rgb_array').transpose((2, 0, 1))
        # Cart is in the lower half, so strip off the top and bottom of the screen
        _, screen_height, screen_width = screen.shape
        screen = screen[:, int(screen_height*0.45):int(screen_height * 0.75)]
        view_width = int(screen_width * 0.2)
        cart_location = self.get_cart_location(screen_width)
        if cart_location < view_width // 2:
            slice_range = slice(view_width)
        elif cart_location > (screen_width - view_width // 2):
            slice_range = slice(-view_width, None)
        else:
            slice_range = slice(cart_location - view_width // 2,
                    cart_location + view_width // 2)
        # Strip off the edges, so that we have a square image centered on a cart
        screen = screen[:, :, slice_range]
        # Convert to float, rescale, convert to torch tensor
        # (this doesn't require a copy)
        screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
        screen = torch.from_numpy(screen)
        # Resize, and add a batch dimension (BCHW)
        return User.resize(screen).unsqueeze(0)
    
    @classmethod
    def show_sample(cls,screen):
        plt.figure(1)
        plt.imshow(screen.cpu().squeeze(0).permute(1, 2, 0).numpy(),
                   interpolation='none')
        plt.title('Example extracted screen')
        plt.show()
        
    def save_context(self):
        serial_context = self.context.serialize(save_secret_key=True)
        f = open(self.cp.PATH+'/context.bytes', 'wb')
        f.write(serial_context)
        f.close()
        self._sk_= self.context.secret_key()
        self.context.make_context_public()
        
    def _switch_context(self, path):
        file = open(path, 'rb')
        serial_context = file.read()
        self.context = ts.Context.load(serial_context)
        
    def send_models_to_cp(self, FILE, CTX, EP=0):
        if not os.path.exists(CTX):
            raise FileNotFoundError("Context missing from path")
        if not os.path.exists(FILE):
            raise FileNotFoundError("Model missing from path")
        if self.cp is None:
            raise RuntimeError("No active platform found. Please hook one")
        self._switch_context(CTX)
        self.env.reset()
        init_screen = self.get_screen()
        User.show_sample(init_screen)
        _, _, screen_height, screen_width = init_screen.shape
        # Get number of actions from gym action space
        n_actions = self.env.action_space.n
        self.cp.prepare_model(screen_height,screen_width,n_actions)
        param = self.cp.load_model(FILE, EP)
        self._run(*param)
    
    def begin_operation(self):
        if self.cp is None:
            raise RuntimeError("No active platform found. Please hook one")
        if not self.cp.params_set:
            raise RuntimeError("Parameters are not set on hooked platform")
        self.env.reset()
        init_screen = self.get_screen()
        User.show_sample(init_screen)
        _, _, screen_height, screen_width = init_screen.shape
        # Get number of actions from gym action space
        n_actions = self.env.action_space.n
        self.cp.prepare_model(screen_height,screen_width,n_actions)
        self._run()
        
    def _run(self, prev_ep=0, prev_test=0):
        self.cp.make_dir()
        self.save_context()
        num_episodes = self.cp.EPISODE
        if num_episodes>=50:
            num_episodes+=(num_episodes//50)*10
        st = time()
        test_episodes = prev_test
        max_t = 0
        max_t_ep = 0
        for i_episode in range(prev_ep,num_episodes):
            self.env.reset()
            last_screen = self.get_screen()
            current_screen = self.get_screen()
            state = current_screen - last_screen
            test_phase = False
            test_c = i_episode%50
            if i_episode>=50 and test_c < 10:
                test_phase = True
                if test_c == 0:
                    self.cp.load_enc_net()
            for t in count():
                if test_phase:
                    state = self.encrypt_input(state)
                action = self.cp.select_action(state,test_phase)
                _, reward, done, _ = self.env.step(action.item())
                last_screen = current_screen
                current_screen = self.get_screen()
                if not done:
                    next_state = current_screen - last_screen
                else:
                    next_state = None
                if not test_phase:
                    self.cp.save_transition(state, action, next_state, reward)
                    self.cp.optimize_model()
                state = next_state
                if done:
                    if t>max_t:
                        max_t=t
                        max_t_ep=i_episode
                    if test_phase:
                        self.cp.update_duration_enc(t+1,test_episodes)
                    else:
                        self.cp.update_duration(t+1,i_episode+1-test_episodes)
                                                                                      
                    break
            # Update the target network, copying all weights and biases in DQN
            self.cp.update_model(i_episode+1-test_episodes)
            User.show_timer(st, i_episode+1)
            if test_phase:
                test_episodes+=1
            if i_episode>0 and (i_episode+1)%60==0:
                self.cp.save_model(i_episode, max_t, max_t_ep, test_episodes)

        print('Complete')
        self.env.render()
        self.env.close()
        plt.ioff()
        self.cp.save_info(max_t, max_t_ep)
        self.cp.save_model(i_episode, max_t, max_t_ep, test_episodes)
        
    def encrypt_input(self, state):
        kernel_shape = self.cp.policy_net.conv1.kernel_size
        stride = self.cp.policy_net.conv1.stride[0]
        enc_state = ts.im2col_encoding(
                self.context, state.view(30, 30).tolist(), kernel_shape[0],
                kernel_shape[1], stride
                )
        return enc_state
        
    def process_output(self, predictions):
        pred = predictions.decrypt(self._sk_)
        return torch.FloatTensor([pred])
    
    def process_batch_output(self, batch):
        return torch.cat(tuple(map(lambda s: Variable(self.process_output(s),requires_grad=True),
                                          batch)))
    @classmethod    
    def plot_durations(cls, episode_durations, eps, fig):
        plt.figure(fig)
        plt.clf()
        durations_t = torch.tensor(episode_durations, dtype=torch.float)
        plt.title('Training...')
        plt.xlabel('Episode c='+str(eps))
        plt.ylabel('Duration')
        plt.plot(durations_t.numpy())
        if len(durations_t) >= 20:
            means = durations_t.unfold(0, 20, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(19), means))
            plt.plot(means.numpy())
        plt.pause(0.001)  # pause a bit so that plots are updated
        if User.is_ipython:
            User.display.clear_output(wait=True)
            User.display.display(plt.gcf())
        plt.show()
            
    @classmethod
    def show_timer(cls, st, eps):
        seconds = time()-st
        elapsed = ""
        if(seconds>=3600):
            elapsed = str(seconds//3600)+" Hours  :  "
            seconds%=3600
        if(seconds>=60):
            elapsed = elapsed+str(seconds//60)+" Minutes  :  "
            seconds%=60
        elapsed = elapsed+str(round(seconds, 3))+" Seconds  :  "
        print('Episode complete: {}     Time Elapsed: {}\n'.format(eps, elapsed))
        
        
        

class Cloud:
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))
    
    def __init__(self):
        self.params_set = False
        self.steps_done = 0
        self.client = None
        
    def set_parameters(self,
                       EPISODE,
                       BATCH_SIZE = 64, 
                       GAMMA = 0.99, 
                       EPS_START = 0.9,
                       EPS_END = 0.05, 
                       EPS_DECAY = 500, 
                       TARGET_UPDATE = 10
                        ):
        self.EPISODE = EPISODE
        self.BATCH_SIZE = BATCH_SIZE 
        self.GAMMA = GAMMA 
        self.EPS_START = EPS_START
        self.EPS_END = EPS_END 
        self.EPS_DECAY = EPS_DECAY 
        self.TARGET_UPDATE = TARGET_UPDATE
        self.params_set = True
    
    def prepare_model(self, height, width, n_actions ):
        self.policy_net = model.Network(height, width, n_actions).to(Cloud.device)
        self.target_net = model.Network(height, width, n_actions).to(Cloud.device)
        self.policy_net_enc = model.Network(height, width, n_actions, mode='encrypted').to(Cloud.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.policy_net_enc.eval()
        self.optimizer = optim.RMSprop(self.policy_net.parameters())
        self.memory = model.ReplayMemory(10000,Cloud.Transition)
        self.episode_durations = [0]
        self.enc_episode_durations = [0]
        self.n_actions = n_actions
        self.model_id = uuid.uuid4().hex
        
    def make_dir(self):
        PATH = os.getcwd()+'/prev_models'
        if not os.path.exists(PATH): 
            os.mkdir(PATH)  
        PATH = PATH+'/'+self.model_id
        if not os.path.exists(PATH): 
            os.mkdir(PATH)
        self.PATH = PATH
        
    def select_action(self, state, test):
        if test:
            with torch.no_grad():
                out = self.policy_net_enc(*state)
                return self.client.process_output(out).max(1)[1].view(1, 1)
        else:
            sample = random.random()
            eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
                math.exp(-1. * self.steps_done / self.EPS_DECAY)
            self.steps_done += 1
            if sample > eps_threshold:
                with torch.no_grad():
                    # t.max(1) will return largest column value of each row.
                    # second column on max result is index of where max element was
                    # found, so we pick action with the larger expected reward.
                    out = self.policy_net(state)
                    return out.max(1)[1].view(1, 1)
            else:
                return torch.tensor([[random.randrange(self.n_actions)]], device=Cloud.device, dtype=torch.long)
    
    def optimize_model(self):
        if len(self.memory) < self.BATCH_SIZE:
            return           
        transitions = self.memory.sample(self.BATCH_SIZE)
        batch = Cloud.Transition(*zip(*transitions))
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=Cloud.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        state_action_values = self.policy_net(state_batch)
        state_action_values = state_action_values.gather(1, action_batch)
        next_state_values = torch.zeros(self.BATCH_SIZE, device=Cloud.device)
        target_net_output = self.target_net(non_final_next_states)
        next_state_values[non_final_mask] = target_net_output.max(1)[0].detach()
        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        
    def load_enc_net(self):
        self.policy_net_enc.load_state_dict(self.policy_net.state_dict()) 
        
    def save_transition(self, state, action, next_state, reward):
        reward = torch.tensor([reward], device=Cloud.device)
        self.memory.push(state, action, next_state, reward)
        
    def update_duration(self,t,eps):
        self.episode_durations.append(t)
        User.plot_durations(self.episode_durations,eps, 2)
    
    def update_duration_enc(self, t, eps):
        self.enc_episode_durations.append(t)
        User.plot_durations(self.enc_episode_durations,eps, 3)
        
    def update_model(self, i_episode):
        if i_episode % self.TARGET_UPDATE == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def save_model(self, episode, t_max, t_max_ep, test):
        checkpoint = {
                'model_id': self.model_id,
                'policy_state': self.policy_net.state_dict(),
                'target_state': self.target_net.state_dict(),
                'policy_state_enc': self.policy_net_enc.state_dict(),
                'optimizer_state': self.optimizer.state_dict(),
                'episode_duration': self.episode_durations,
                'enc_episode_duration': self.enc_episode_durations,
                'n_episode': self.EPISODE,
                'i_episode': episode,
                'batch': self.BATCH_SIZE, 
                'gamma': self.GAMMA, 
                'eps_s': self.EPS_START,
                'eps_e': self.EPS_END, 
                'eps_d': self.EPS_DECAY, 
                'target_up': self.TARGET_UPDATE,
                'steps_done': self.steps_done,
                'test': test
            }
        torch.save(checkpoint,self.PATH+'/'+str(episode+1)+'.pth')
        f = open(self.PATH+'/'+str(episode+1)+' info.txt', 'w')
        f.write('Max Score: '+str(t_max)+'\tIn Episode: '+str(t_max_ep+1))
        f.write('\nModel id: '+self.model_id+'\tEpisodes Ran: '+str(episode+1))
        f.flush()
        f.close()
        
    def load_model(self, FILE, ep):
        check = torch.load(FILE)
        self.model_id = check['model_id']
        self.policy_net.load_state_dict(check['policy_state'])
        self.target_net.load_state_dict(check['target_state'])
        self.optimizer.load_state_dict(check['optimizer_state'])
        self.policy_net_enc.load_state_dict(check['policy_state_enc'])
        self.BATCH_SIZE = check['batch']
        self.GAMMA = check['gamma']
        self.EPS_START = check['eps_s']
        self.EPS_END = check['eps_e']
        self.EPS_DECAY = check['eps_d']
        self.TARGET_UPDATE = check['target_up']
        self.steps_done = check['steps_done']
        self.episode_durations = check['episode_duration']
        self.enc_episode_durations = check['enc_episode_duration']
        self.params_set = True
        self.EPISODE = check['n_episode']+ep
        i_episode = check['i_episode']
        test = check['test']
        User.plot_durations(self.episode_durations, i_episode, 2)
        User.plot_durations(self.enc_episode_durations, test, 3)
        return i_episode , test
            
        
    def save_info(self, t, t_ep):
        f = open(os.getcwd()+'/model_info.txt', 'a')
        f.write('Model: '+self.model_id+'  t_max = '+str(t)+'\tt_max_ep = '+str(t_ep)+'\n')
        f.flush()
        f.close()
        if t>self.get_prev_max():
            f = open("model_max.txt", 'w')
            f.write('Max T : '+str(t))
            f.write('\nModel : '+self.model_id)
            f.write('\nIn Episode : '+str(t_ep))
            f.write('\nTotal Ep : '+str(self.EPISODE))
            f.flush()
            f.close()
    
    def get_prev_max(self):
        try:
            f = open("model_max.txt", 'r+')
            return int(f.readline()[len('Max T : '):])
        except FileNotFoundError:
            return 0
    
    def change_dir(self, model):
        os.rmdir(self.PATH)
        PATH = self.PATH[:(-1*len(self.model_id))]+model
        if not os.path.exists(PATH): 
            os.mkdir(PATH)
        self.PATH = PATH

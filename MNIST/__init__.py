from typing import Tuple, Optional

import torch
import torch.nn.functional as F
import torch.utils.data
from torch import nn

from utils import gather


class DenoiseDiffusion_CEM:


    def __init__(self, eps_model: nn.Module, n_steps: int, schedule: str, device: torch.device):

        super().__init__()
        self.eps_model = eps_model
        self.schedule = schedule
        
        if self.schedule=='exp':

            self.beta = 1-torch.exp(-10.0/n_steps*torch.ones([n_steps,])).to(device)
            self.alpha = 1. - self.beta
            self.alpha_bar = torch.cumprod(self.alpha, dim=0)
            self.n_steps = n_steps
            self.sigma2 = self.beta

            tmp1 = torch.tensor(0)*torch.arange(0,1,1).to(device)
            tmp2 = 0.0001*torch.exp(torch.log(torch.tensor(10/0.0001))*(1/(n_steps-1))*torch.arange(0,n_steps,1)).to(device)
            self.t_bw = torch.cat((tmp1, tmp2), dim=0).to(device)
        if self.schedule=='fixed':
            self.beta = torch.linspace(0.0001, 0.02, n_steps).to(device)
            self.alpha = 1. - self.beta
            self.alpha_bar = torch.cumprod(self.alpha, dim=0)
            self.n_steps = n_steps
            self.sigma2 = self.beta
        


    def q_xt_x0(self, x0: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:


        mean = gather(self.alpha_bar, t) ** 0.5 * x0
        var = 1 - gather(self.alpha_bar, t)
        
        return mean, var

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, eps: Optional[torch.Tensor] = None):



        if eps is None:
            eps = torch.randn_like(x0)

        mean, var = self.q_xt_x0(x0, t)

        return mean + (var ** 0.5) * eps

    def p_sample(self, xt: torch.Tensor, t: torch.Tensor, timeIndex: bool=False, schedule: str='exp'):
        
        if schedule=='exp':
            t_bw_current = gather(self.t_bw, t)
            t_bw_next = gather(self.t_bw, t-1)
            dt = t_bw_current-t_bw_next
            alpha_bar = torch.exp(-t_bw_current)
            if timeIndex:
                eps_theta = self.eps_model(xt, t)
            else:
                eps_theta = self.eps_model(xt, -torch.log(alpha_bar).reshape([xt.shape[0],]))
            alpha = torch.exp(-dt)
            score_func = xt/(1-alpha_bar) - (alpha_bar**0.5)/(1-alpha_bar)*eps_theta
            mean = alpha**(-0.5)*(xt-dt*score_func)
            var = 1-alpha
            eps = torch.randn(xt.shape, device=xt.device)
        elif schedule=='fixed':
            alpha_bar = gather(self.alpha_bar, t)
            if timeIndex:
                eps_theta = self.eps_model(xt, t)
            else:
                eps_theta = self.eps_model(xt, -torch.log(alpha_bar).reshape([xt.shape[0],]))
            alpha = gather(self.alpha, t)
            dt = -torch.log(alpha)
            score_func = xt/(1-alpha_bar) - (alpha_bar**0.5)/(1-alpha_bar)*eps_theta
            mean = alpha**(-0.5)*(xt-dt*score_func)
            var = 1-alpha
            eps = torch.randn(xt.shape, device=xt.device)
        # Sample
        return mean + (var ** .5) * eps

    def loss(self, x0: torch.Tensor, timeIndex: bool=False, noise: Optional[torch.Tensor] = None):

        batch_size = x0.shape[0]

        t = torch.randint(0, self.n_steps, (batch_size,), device=x0.device, dtype=torch.long)


        if noise is None:
            noise = torch.randn_like(x0)


        xt = self.q_sample(x0, t, eps=noise)
        alpha_bar = gather(self.alpha_bar, t)
        if timeIndex:
            eps_theta = self.eps_model(xt, t)
        else:
            eps_theta = self.eps_model(xt, -torch.log(torch.tensor(alpha_bar)).reshape([batch_size,]))
        # weighting function
        lambda_func = 1.0/(1.0/alpha_bar-1)

        return torch.mean(lambda_func * (x0-eps_theta)**2)
    


    
    
class DenoiseDiffusion_DDPM:


    def __init__(self, eps_model: nn.Module, n_steps: int, schedule: str, device: torch.device):

        super().__init__()
        self.eps_model = eps_model
        self.schedule = schedule
        
        if self.schedule=='exp':

            self.beta = 1-torch.exp(-10.0/n_steps*torch.ones([n_steps,])).to(device)
            self.alpha = 1. - self.beta
            self.alpha_bar = torch.cumprod(self.alpha, dim=0)
            self.n_steps = n_steps
            self.sigma2 = self.beta

            tmp1 = torch.tensor(0)*torch.arange(0,1,1).to(device)
            tmp2 = 0.0001*torch.exp(torch.log(torch.tensor(10/0.0001))*(1/(n_steps-1))*torch.arange(0,n_steps,1)).to(device)
            self.t_bw = torch.cat((tmp1, tmp2), dim=0).to(device)
        if self.schedule=='fixed':
            self.beta = torch.linspace(0.0001, 0.02, n_steps).to(device)
            self.alpha = 1. - self.beta
            self.alpha_bar = torch.cumprod(self.alpha, dim=0)
            self.n_steps = n_steps
            self.sigma2 = self.beta

    def q_xt_x0(self, x0: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:



        mean = gather(self.alpha_bar, t) ** 0.5 * x0

        var = 1 - gather(self.alpha_bar, t)
        
        return mean, var

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, eps: Optional[torch.Tensor] = None):

        if eps is None:
            eps = torch.randn_like(x0)

        mean, var = self.q_xt_x0(x0, t)

        return mean + (var ** 0.5) * eps

    def p_sample(self, xt: torch.Tensor, t: torch.Tensor, timeIndex: bool=False, schedule: str='exp'):
        if schedule=='exp':
            t_bw_current = gather(self.t_bw, t)
            t_bw_next = gather(self.t_bw, t-1)
            dt = t_bw_current-t_bw_next
            alpha_bar = torch.exp(-t_bw_current)
            if timeIndex:
                eps_theta = self.eps_model(xt, t)
            else:  
                eps_theta = self.eps_model(xt, -torch.log(alpha_bar).reshape([xt.shape[0],]))

            alpha = torch.exp(-dt)
            eps_coef = (1 - alpha) / (1 - alpha_bar) ** .5
            mean = 1 / (alpha ** 0.5) * (xt - eps_coef * eps_theta)
            var = 1-alpha
            
        elif schedule=='fixed':
            alpha_bar = gather(self.alpha_bar, t)
            eps_theta = self.eps_model(xt, -torch.log(torch.tensor(alpha_bar)).reshape([xt.shape[0],]))
            alpha = gather(self.alpha, t)
            eps_coef = (1 - alpha) / (1 - alpha_bar) ** .5
            mean = 1 / (alpha ** 0.5) * (xt - eps_coef * eps_theta)
            var = gather(self.sigma2, t)
            
        eps = torch.randn(xt.shape, device=xt.device)

        return mean + (var ** .5) * eps
        


    
    
    
    

    def loss(self, x0: torch.Tensor, timeIndex: bool=False, noise: Optional[torch.Tensor] = None):

        batch_size = x0.shape[0]

        t = torch.randint(0, self.n_steps, (batch_size,), device=x0.device, dtype=torch.long)


        if noise is None:
            noise = torch.randn_like(x0)


        xt = self.q_sample(x0, t, eps=noise)
        alpha_bar = gather(self.alpha_bar, t)

        if timeIndex:
            eps_theta = self.eps_model(xt, t)
        else:

            eps_theta = self.eps_model(xt, -torch.log(torch.tensor(alpha_bar)).reshape([batch_size,]))


        return F.mse_loss(noise, eps_theta)

        
        

        

class DenoiseDiffusion_SGM:


    def __init__(self, eps_model: nn.Module, n_steps: int, schedule: str, device: torch.device):

        super().__init__()
        self.eps_model = eps_model
        self.schedule = schedule
        
        if self.schedule=='exp':

            self.beta = 1-torch.exp(-10.0/n_steps*torch.ones([n_steps,])).to(device)
            self.alpha = 1. - self.beta
            self.alpha_bar = torch.cumprod(self.alpha, dim=0)
            self.n_steps = n_steps
            self.sigma2 = self.beta

            tmp1 = torch.tensor(0)*torch.arange(0,1,1).to(device)
            tmp2 = 0.0001*torch.exp(torch.log(torch.tensor(10/0.0001))*(1/(n_steps-1))*torch.arange(0,n_steps,1)).to(device)
            self.t_bw = torch.cat((tmp1, tmp2), dim=0).to(device)
        if self.schedule=='fixed':
            self.beta = torch.linspace(0.0001, 0.02, n_steps).to(device)
            self.alpha = 1. - self.beta
            self.alpha_bar = torch.cumprod(self.alpha, dim=0)
            self.n_steps = n_steps
            self.sigma2 = self.beta

    def q_xt_x0(self, x0: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:



        mean = gather(self.alpha_bar, t) ** 0.5 * x0

        var = 1 - gather(self.alpha_bar, t)
        
        return mean, var

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, eps: Optional[torch.Tensor] = None):
        """
        #### Sample from $q(x_t|x_0)$

        \begin{align}
        q(x_t|x_0) &= \mathcal{N} \Big(x_t; \sqrt{\bar\alpha_t} x_0, (1-\bar\alpha_t) \mathbf{I} \Big)
        \end{align}
        """


        if eps is None:
            eps = torch.randn_like(x0)


        mean, var = self.q_xt_x0(x0, t)

        return mean + (var ** 0.5) * eps

    def p_sample(self, xt: torch.Tensor, t: torch.Tensor, timeIndex: bool=False, schedule: str='exp'):
        if schedule=='exp':
            t_bw_current = gather(self.t_bw, t)
            t_bw_next = gather(self.t_bw, t-1)
            dt = t_bw_current-t_bw_next
            alpha_bar = torch.exp(-t_bw_current)
            if timeIndex:
                eps_theta = self.eps_model(xt, t)
            else:
                eps_theta = self.eps_model(xt, -torch.log(alpha_bar).reshape([xt.shape[0],]))
            alpha = torch.exp(-dt)
            eps_coef = (1 - alpha) 
            mean = 1 / (alpha ** 0.5) * (xt - eps_coef * eps_theta)
            var = 1-alpha
        elif schedule=='fixed':
            alpha_bar = gather(self.alpha_bar, t)
            if timeIndex:
                eps_theta = self.eps_model(xt, t)
            else:
                eps_theta = self.eps_model(xt, -torch.log(alpha_bar).reshape([xt.shape[0],]))
            alpha = gather(self.alpha, t)
            eps_coef = (1 - alpha) 
            mean = 1 / (alpha ** 0.5) * (xt - eps_coef * eps_theta)
            var = 1-alpha
        
        eps = torch.randn(xt.shape, device=xt.device)

        return mean + (var ** .5) * eps

    def loss(self, x0: torch.Tensor, timeIndex: bool=False, noise: Optional[torch.Tensor] = None):


        batch_size = x0.shape[0]

        t = torch.randint(0, self.n_steps, (batch_size,), device=x0.device, dtype=torch.long)


        if noise is None:
            noise = torch.randn_like(x0)


        xt = self.q_sample(x0, t, eps=noise)
        alpha_bar = gather(self.alpha_bar, t)

        if timeIndex:
            eps_theta = self.eps_model(xt, t)
        else:

            eps_theta = self.eps_model(xt, -torch.log(torch.tensor(alpha_bar)).reshape([batch_size,]))


        lambda_func = 1 - alpha_bar
        return torch.mean(lambda_func * (noise/torch.sqrt(1-alpha_bar)-eps_theta)**2)
    
    
    
    
    
    
    
    

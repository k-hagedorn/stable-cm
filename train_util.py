import torch as th
import numpy as np


def mean_flat(x):
    """
    Take the mean over all non-batch dimensions.
    """
    return th.mean(x, dim=list(range(1, len(x.size()))))


class TrainLoop():
    def __init__(self, sigma_data, P_mean, P_std, batch_size, c):
        self.sigma_data = sigma_data
        self.P_mean = P_mean
        self.P_std = P_std
        self.batch_size = batch_size
        self.c = c
  
    def model_wrapper(self,scaled_x_t, t):
        pred = self.avg_model(scaled_x_t, t.flatten(), return_logvar=False)
        # If you want the model to be conditioned on class label (or anything else), just add it as an additional argument:
        # You do not need to change anything else in the algorithm.
        # pred, logvar = model(scaled_x_t, t.flatten(), class_label, return_logvar=True)
        return pred
            
    def d_model(self, model, x, t, dxt):
        tangent = th.cos(t) * th.sin(t) * self.sigma_data
        out = th.autograd.functional.jvp(model, (x/self.sigma_data, t), ((th.cos(t) * th.sin(t)).view(-1,1,1,1) * dxt, tangent))
        out2 = out[0] + out[1]
        return out2
    
    def sample_t(self):
        sigma_max = 80.*th.ones((self.batch_size,))
        t = th.arctan(sigma_max/self.sigma_data)
        return t
        
        
    def compute_g(self, avg_model, xt, t, dxt, r):
        v_x = th.cos(t) * th.sin(t) * dxt / self.sigma_data
        v_t = th.cos(t) * th.sin(t)
        model, model_grad = th.autograd.functional.jvp(
                avg_model,
                (xt / self.sigma_data, t),
                (v_x, v_t),
        )
        model_grad = model_grad.detach()
        
        g = -th.cos(t) * th.cos(t) * (self.sigma_data * model - dxt)
        second_term = -r * (th.cos(t) * th.sin(t) * xt + self.sigma_data * model_grad)
        g_norm = th.linalg.vector_norm(g, dim=(1, 2, 3), keepdim=True)
        g_norm = g_norm * np.sqrt(g_norm.numel() / g.numel())  
        g = g / (g_norm + 0.1)
        g = g + second_term
        return g, model

    def compute_d_xt_distill(self, pretrain, xt, t):
        return self.sigma_data * pretrain(xt/self.sigma_data, t)
    
    def compute_d_xt_train(self, x1, t, noise):
        return th.cos(t)*noise - th.sin(t)*x1
    
    def compute_t(self):
        sigma = th.randn(self.batch_size).reshape(-1, 1, 1, 1)
        sigma = (sigma * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        t = th.arctan(sigma/self.sigma_data)
        return t, weight
    
    def loss_fn(self, g, logvar, train, teach, weight):
        logvar = logvar.view(-1, 1, 1, 1)
        l2 = (weight / logvar.exp())  * th.square(train - teach - g) + logvar
        return l2
       
    def distill_loss(self, x1, r, model, avg_model, pretrain_model):
        t, weight = self.compute_t()
        t = t.to(device=x1.device)
        weight = weight.to(device=x1.device)
        noise = th.randn_like(x1) * self.sigma_data
        xt = th.cos(t) * x1 + th.sin(t) * noise
        dxt = self.compute_d_xt_distill(pretrain_model, xt, t)
        model_out, logvar = model(xt/self.sigma_data, t.flatten(), return_logvar=True)
        g, avg_model_out = self.compute_g(avg_model, xt, t, dxt, r)
        loss = self.loss_fn(g=g, logvar=logvar, train=model_out, teach=avg_model_out, weight=weight)
        return loss
    
    def train_loss(self, x1, r, model, avg_model):
        t, weight = self.compute_t()
        t = t.to(device=x1.device)
        weight = weight.to(device=x1.device)
        noise = th.normal(mean=0.0, std=self.sigma_data, size=x1.shape).to(device=x1.device)
        xt = th.cos(t) * x1 + th.sin(t) * noise
        dxt = self.compute_d_xt_train(x1, t, noise)
        model_out, logvar = model(xt/self.sigma_data, t.flatten(), return_logvar=True)
        g, avg_model_out = self.compute_g(avg_model, xt, t, dxt, r)
        loss = self.loss_fn(g=g, logvar=logvar, train=model_out, teach=avg_model_out, weight=weight)
        return loss
        
        
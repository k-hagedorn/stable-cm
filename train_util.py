import torch as th


def mean_flat(x):
    """
    Take the mean over all non-batch dimensions.
    """
    return th.mean(x, dim=list(range(1, len(x.size()))))


class TrainLoop():
    def __init__(self, sigma_data, P_mean, P_std, pretrain_model, batch_size, c):
        self.sigma_data = sigma_data
        self.P_mean = P_mean
        self.P_std = P_std
        self.pretrain = pretrain_model
        self.batch_size = batch_size
        self.c = c
        self.dim = (pretrain_model.in_channels*pretrain_model.input_size**2 / self.batch_size) * th.ones((batch_size,)).reshape(-1, 1, 1, 1)
    
    
    def d_model(self, model, x, t, dxt):
        tangent = th.cos(t) * th.sin(t) * self.sigma_data
        out = th.autograd.functional.jvp(model, (x/self.sigma_data, t), ((th.cos(t) * th.sin(t)).view(-1,1,1,1) * dxt, tangent))
        out2 = out[0] + out[1]
        return out2
    
    def sample_t(self):
        sigma_max = 80.*th.ones((self.batch_size,))
        t = th.arctan(sigma_max/self.sigma_data)
        return t
        
        
    def compute_g(self, model, xt, t, d_xt, r):
        g = -(th.cos(t)**2).view(-1,1,1,1)*(self.sigma_data * model(xt/self.sigma_data, t) - d_xt) - (r * th.cos(t)*th.sin(t)).view(-1,1,1,1)*(xt+ self.sigma_data*self.d_model(model=model,x=xt, t=t, dxt=d_xt))
        g = g.div(th.norm(g) + self.c)
        return g

    def compute_d_xt_distill(self, xt, t):
        return self.sigma_data * self.pretrain(xt/self.sigma_data, t)
    
    def compute_d_xt_train(self, x1, t, noise):
        return th.cos(t).view(-1,1,1,1)*noise - th.sin(t).view(-1,1,1,1)*x1
    
    def compute_t(self):
        
        tau = th.normal(mean=self.P_mean, std=self.P_std, size=(self.batch_size,))
        t = th.arctan(tau.exp()/self.sigma_data)
        return t
    
    def loss_fn(self, g,t,weight,train,teach):
        w = weight(t)
        l = w.exp() / self.dim.to(device=train.device)
        l2 = mean_flat(l * (train - teach - g)**2 - w)
        return l2
       
    def distill_loss(self, x1, r, model, avg_model, weight_fn):
        t = self.compute_t().to(device=model.device)
        noise = th.normal(mean=0.0, std=self.sigma_data, size=x1.shape).to(device=model.device)
        xt = th.cos(t).view(-1,1,1,1) * x1 + th.sin(t).view(-1,1,1,1) * noise
        dxt = self.compute_d_xt_distill(xt, t)
        g = self.compute_g(avg_model, xt, t, dxt, r)
        
        model_out = model(xt/self.sigma_data, t)
        avg_model_out = avg_model(xt/self.sigma_data, t)
        
        loss = self.loss_fn(g=g, t=t, weight=weight_fn, train=model_out, teach=avg_model_out)
        return loss
    
    def train_loss(self, x1, r, model, avg_model, weight_fn):
        t = self.compute_t().to(device=model.device)
        noise = th.normal(mean=0.0, std=self.sigma_data, size=x1.shape).to(device=model.device)
        xt = th.cos(t).view(-1,1,1,1) * x1 + th.sin(t).view(-1,1,1,1) * noise
        dxt = self.compute_d_xt_train(x1, t, noise)
        g = self.compute_g(avg_model, xt, t, dxt, r)
        
        model_out = model(xt/self.sigma_data, t)
        avg_model_out = avg_model(xt/self.sigma_data, t)
        
        loss = self.loss_fn(g=g, t=t, weight=weight_fn, train=model_out, teach=avg_model_out)
        return loss
        
        
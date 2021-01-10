import math         
import torch 

class SC_SGD(torch.optim.Optimizer) :
    def __init__(self, params, convex = False, lr=1e-2, eps=1e-9, weight_decay=1e-2, correct_bias=True) :

        if lr < 0.0 : raise ValueError("Invalid LR {} Allowed >= 0.0".format(lr))
        if not eps >= 0.0 : raise ValueError("Invalid epsilon {} Allowed >= 0.0".format(eps))

        defaults = dict(lr=lr, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias, convex = convex)
        super(SC_SGD, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None) :
        loss = None
        if closure is not None :
            with torch.enable_grad() :  
                loss = closure()

        for group in self.param_groups :
            for p in group['params'] :
                if p.grad is None : continue
                grad = p.grad
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 1

                #state['step'] += 1 
                if group['convex'] : 
                    alpha = group['lr']/(state['step'])# + group['eps'])
                else : 
                    alpha = group['lr']
                
                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])
                
                p.add_(grad, alpha = -alpha)

        return loss 

class SC_RMSprop(torch.optim.Optimizer):

    def __init__(self, params, convex = False, lr=1e-2, alpha=0.9, eps1=0.1, eps2=1, weight_decay=1e-2):
        if not 0.0 <= lr : raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps1 : raise ValueError("Invalid epsilon value: {}".format(eps1))
        if not 0.0 <= eps2 : raise ValueError("Invalid epsilon value: {}".format(eps2))
        if not 0.0 <= weight_decay : raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if not 0.0 <= alpha : raise ValueError("Invalid alpha value: {}".format(alpha))

        
        defaults = dict(lr=lr, convex = convex, alpha=alpha, eps1=eps1, eps2=eps2, weight_decay=weight_decay)
        super(SC_RMSprop, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SC_RMSprop, self).__setstate__(state)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 1
                    state['square_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                square_avg = state['square_avg']
                alpha = group['alpha']
                beta = 1 - alpha*(1/state['step'])

                if group['convex'] : 
                    lr = group['lr'] / state['step'] 
                else : 
                    lr = group['lr']

                

                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])

                square_avg.mul_(beta).addcmul_(grad, grad, value=1 - beta)

                eps_replaced = torch.exp(-square_avg.mul(group['eps1']*state['step']))
                eps_replaced.mul_(group['eps2'])
                avg = square_avg.add(eps_replaced/state['step'])
                state['step'] += 1
                p.addcmul_(grad, 1/avg, value=-lr)
        return loss

class SC_Adagrad(torch.optim.Optimizer):

    def __init__(self, params, convex = False, lr=1e-2, eps1=0.1, eps2=1, weight_decay=1e-2):
        if not 0.0 <= lr : raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps1 : raise ValueError("Invalid epsilon value: {}".format(eps1))
        if not 0.0 <= eps2 : raise ValueError("Invalid epsilon value: {}".format(eps2))
        if not 0.0 <= weight_decay : raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        
        defaults = dict(lr=lr, convex = convex, eps1=eps1, eps2=eps2, weight_decay=weight_decay)
        super(SC_Adagrad, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SC_Adagrad, self).__setstate__(state)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 1
                    state['square_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                if group['convex'] : 
                    lr = group['lr'] #/ state['step'] 
                else : 
                    lr = group['lr']

                square_avg = state['square_avg']
                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])

                square_avg.addcmul_(grad, grad)
                
                eps_replaced = torch.exp(-square_avg.mul(group['eps1']))
                eps_replaced.mul_(group['eps2'])

                avg = square_avg.add(eps_replaced)
                p.addcdiv_(grad, avg, value=-lr)

        return loss

class SAdam(torch.optim.Optimizer):
    
    def __init__(self, params, beta_1=0.9, lr=0.01, delta=1e-2, xi_1=0.1, xi_2=0.1, gamma=0.1, weight_decay=1e-2):
        defaults = dict(lr=lr, beta_1=beta_1, delta=delta, xi_1=xi_1, xi_2=xi_2, gamma=gamma, weight_decay=weight_decay)
        super(SAdam, self).__init__(params, defaults)
    
    def __setstate__(self, state):
        super(SAdam, self).__setstate__(state)
    
    @torch.no_grad()
    def step(self,closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]

                if len(state)==0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['hat_g_t'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state['v_t'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    
                hat_g_t, v_t = state['hat_g_t'], state['v_t']
                beta_1, gamma, delta = group['beta_1'], group['gamma'], group['delta']
                lr, xi_1, xi_2 = group['lr'], group['xi_1'], group['xi_2']
                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])
                state['step'] += 1
                t = state['step']
                gamma = 1-0.9/(t)
                hat_g_t.mul_(beta_1).add_(grad, alpha=1 - beta_1)
                v_t.mul_(gamma).addcmul_(grad, grad, value=1-gamma)
                if t==20:
                    print("HERE")
                ## vthat = vt + I*delta/t
                denom = t*v_t + delta
                p.addcmul_(hat_g_t, 1/denom, value = -lr)

        return loss

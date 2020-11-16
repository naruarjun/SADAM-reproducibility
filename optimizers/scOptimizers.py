import math         
import torch 


class SC_SGD(torch.optim.Optimizer) :
    def __init__(self, params, lr=1e-3, eps=1e-6, wt_decay=0.0, correct_bias=True):

        if lr < 0.0 : raise ValueError("Invalid LR {} Allowed >= 0.0".format(lr))
        if not eps >= 0.0 : raise ValueError("Invalid epsilon {} Allowed >= 0.0".format(eps))

        defaults = dict(lr=lr, eps=eps, weight_decay=wt_decay, correct_bias=correct_bias)
	steps = 0
        super(SC_SGD, self).__init__(params, defaults)

    def step(self, closure=None) :
        loss = None
        if closure is not None :
            with torch.enable_grad() :	
		loss = closure()

        for group in self.param_groups :
            for p in group['params'] :
                if p.grad is None : continue
                grad = p.grad

		self.steps = self.steps + 1 
		alpha = -group['lr']/math.sqrt(self.steps + eps)
		p.add_(grad, alpha=alpha)

	return loss 

class SC_RMSprop(Optimizer):

    def __init__(self, params, lr=1e-2, alpha=0.99, eps1=0.1, eps2=0.1, weight_decay=0):
        if not 0.0 <= lr : raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps1 : raise ValueError("Invalid epsilon value: {}".format(eps1))
        if not 0.0 <= eps2 : raise ValueError("Invalid epsilon value: {}".format(eps2))
        if not 0.0 <= weight_decay : raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if not 0.0 <= alpha : raise ValueError("Invalid alpha value: {}".format(alpha))

        
        defaults = dict(lr=lr, alpha=alpha, eps1=eps1, eps2=eps2, weight_decay=weight_decay)
        super(RMSprop, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(RMSprop, self).__setstate__(state)

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
                    state['step'] = 0
                    state['square_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                square_avg = state['square_avg']
                alpha = group['alpha']
                beta = 1 - alpha.mul(1/state['step'])

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])

                square_avg.mul_(beta).addcmul_(grad, grad, value=1 - beta)

                eps_replaced = torch.exp(-square_avg.mul(group['eps1']*state['step']))
                eps_replaced.mul_(group['eps2'])

                avg = square_avg.add(eps_replaced)
                lr = group['lr']
                lr.mul_(1/state['step'])
                p.addcdiv_(grad, avg, value=-lr)
        return loss

class SC_Adagrad(Optimizer):

    def __init__(self, params, lr=1e-2, eps1=0.1, eps2=0.1, weight_decay=0):
        if not 0.0 <= lr : raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps1 : raise ValueError("Invalid epsilon value: {}".format(eps1))
        if not 0.0 <= eps2 : raise ValueError("Invalid epsilon value: {}".format(eps2))
        if not 0.0 <= weight_decay : raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        
        defaults = dict(lr=lr, eps1=eps1, eps2=eps2, weight_decay=weight_decay)
        super(RMSprop, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(RMSprop, self).__setstate__(state)

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
                    state['step'] = 0
                    state['square_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                square_avg = state['square_avg']
                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])

                square_avg.addcmul_(grad, grad)
                
                eps_replaced = torch.exp(-square_avg.mul(group['eps1']))
                eps_replaced.mul_(group['eps2'])

                avg = square_avg.add(eps_replaced)
                lr = group['lr']
                lr.mul_(1/state['step'])
                p.addcdiv_(grad, avg, value=-lr)

        return loss

import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.optim import Adam
from moosvgd import get_gradient
from pymoo.factory import get_problem
from pymoo.util.plotting import plot
from zdt_functions import *
import time
from pymoo.factory import get_performance_indicator

cur_problem = 'zdt3'
run_num = 0

if __name__ == '__main__':
    x = torch.rand((50, 30))
    x.requires_grad = True
    optimizer = Adam([x], lr=5e-4)
    
    ref_point = get_ref_point(cur_problem)
    hv = get_performance_indicator('hv', ref_point=ref_point)
    iters = 10000
    start_time = time.time()
    hv_results = []
    for i in range(iters):
        loss_1, loss_2 = loss_function(x, problem=cur_problem)
        pfront = torch.cat([loss_1.unsqueeze(1), loss_2.unsqueeze(1)], dim=1)
        pfront = pfront.detach().cpu().numpy()
        hvi = hv.calc(pfront)
        hv_results.append(hvi)
         
        if i%1000 == 0:
            problem = get_problem(cur_problem)
            x_p = problem.pareto_front()[:, 0]
            y_p = problem.pareto_front()[:, 1]
            plt.scatter(x_p, y_p, c='r')

            plt.scatter(loss_1.detach().cpu().numpy(),loss_2.detach().cpu().numpy())
            plt.savefig('figs/%s_%d.png'%(cur_problem, i))
            plt.close()
        
        loss_1.sum().backward(retain_graph=True)
        grad_1 = x.grad.detach().clone()
        x.grad.zero_()

        loss_2.sum().backward(retain_graph=True)
        grad_2 = x.grad.detach().clone()
        x.grad.zero_()
       
        # Perforam gradient normalization trick 
        grad_1 = torch.nn.functional.normalize(grad_1, dim=0)
        grad_2 = torch.nn.functional.normalize(grad_2, dim=0)

        optimizer.zero_grad()
        losses = torch.cat([loss_1.unsqueeze(1), loss_2.unsqueeze(1)], dim=1)
        x.grad = get_gradient(grad_1, grad_2, x, losses)
        optimizer.step()
        
        x.data = torch.clamp(x.data.clone(), min=1e-6, max=1.-1e-6)

    print(i, 'time:', time.time()-start_time, 'hv:', hvi, loss_1.sum().detach().cpu().numpy(), loss_2.sum().detach().cpu().numpy())

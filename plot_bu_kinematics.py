#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 21 10:43:21 2022

read and plot bu_kinematicts data

@author: mlisakov


"""


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import re 
from scipy.optimize import curve_fit
import matplotlib.animation as animation

import Model as mo

def read_comp(file):
    """read com position and parameters from a file"""
    names=['date', 'mjd', 'flux', 'flux_err', 'r','r_err',
                                 'theta', 'theta_radio', 'theta_err', 'x', 'y', 'x_err', 'y_err',
                                 'size', 'size_err', 'ratio', 'phi', 'tb', 'tb_flag']
    
    df = pd.read_csv(file,names=names, header=None, skiprows=1, index_col=False)
        
    df.loc[:, names[1:]] = df.loc[:, names[1:]].astype(float)
    
    return df



base = '/homes/mlisakov/data/nrao530/data/bu_kinematics/Knots/'
files = ['A0.csv',  'A1.csv',  'B1.csv',  'B2.csv',  'B3.csv',  'B4.csv',  'B5.csv',  'B6.csv',  'B7.csv',  'D.csv',  'T2.csv',  'T6.csv']
files = [ 'A0.csv',   'B1.csv',  'B2.csv',  'B3.csv',  'B4.csv',  'B5.csv',  'B6.csv',   'D.csv'] # which exhibit motion and seem to be good for kinem

files = [a+b for a,b in zip([base]*len(files), files)]
# colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
# colors = [a+b for a,b in zip(['tab:']*len(colors), colors)]

colors = {'B1':'tab:blue', 'B2':'tab:orange', 'B3':'tab:green', 'B4':'tab:red', 'B5':'tab:purple', 'B6':'tab:brown', 'D':'tab:pink', 'A0':'tab:gray', 
          'A1':'tab:olive', 'B7':'tab:cyan', 'T2':'tab:grey', 'T6':'yellow', }

if False:    
    fig, ax = plt.subplots(3,4)
    [ax1.set_xlim(-0.5, 0.5) for ax1 in ax.reshape(12)]
    [ax1.set_ylim(-0.1, 2.7) for ax1 in ax.reshape(12)]
    
    for i, y in enumerate(np.arange(2008, 2019)):
        
        axi = ax[int(i/4)][int(i - int(i/4)*4)]
        
        for f in files:
            a = re.search("\/([ABDT]\d*)\.", f)
            component = a.group(1)
            df = read_comp(f)
            
            df = df.loc[(df.date > y)& (df.date < y+1),:]
            if df.index.size == 0:
                continue
            
            
            
            # fig, ax = plt.subplots(1,1)
            # fig.suptitle(component)
            axi.plot(df.x, df.y, '-o', label=component, color=colors[component])
            # ax.text(df.loc[df.index.min(), 'x'], df.loc[df.index.min(), 'y'], df.loc[df.index.min(), 'date'].astype(int))
            # ax.text(df.loc[df.index.max(), 'x'], df.loc[df.index.max(), 'y'], df.loc[df.index.max(), 'date'].astype(int))
            axi.legend()
        
        axi.invert_xaxis()
        axi.set_title(y)
            
            
    # fig.legend()
    fig.suptitle("NRAO 530 components in xy plane")
    

if False:
    # add my model to 2017 plot
    fig, ax = plt.subplots(1,1)
    m = mo.Model()
    m.read_model('/homes/mlisakov/data/nrao530/models/1730-130.q1.2017_04_04.mdl_final')
    m.shift_model(-0.006, 0.024)
    m.plot_mas(ax)
    y=2008
    for f in files:
        a = re.search("\/([ABDT]\d*)\.", f)
        component = a.group(1)
        df = read_comp(f)
        
        df = df.loc[(df.date > y)& (df.date < y+11),:]
        if df.index.size == 0:
            continue
        
        
        
        # fig, ax = plt.subplots(1,1)
        # fig.suptitle(component)
        ax.plot(df.x, df.y, '-o', label=component, color=colors[component])
        # ax.text(df.loc[df.index.min(), 'x'], df.loc[df.index.min(), 'y'], df.loc[df.index.min(), 'date'].astype(int))
        # ax.text(df.loc[df.index.max(), 'x'], df.loc[df.index.max(), 'y'], df.loc[df.index.max(), 'date'].astype(int))
        ax.legend()
    
    ax.invert_xaxis()
    ax.set_title(y)
    






# plot component's radial distance versus time for those which exhibit speed-ups
# files = [ 'B1.csv',  'B2.csv',  'B3.csv',  'B5.csv'] # which exhibit motion and seem to be good for kinem
# files = ['B1.csv']
# files = [a+b for a,b in zip([base]*len(files), files)]


def func(x, a, b):
    return x*a + b

import torch
def broken_linear(x, x_break, a1, b1, a2, b2):
    b2 = a1*x_break + b1 - a2*x_break # to make sure they are connected 
    y = torch.where(x <= x_break, a1*x + b1, a2*x + b2)
    return y

def double_broken_linear(x, x_break1, x_break2, a1, b1, a2, b2, a3, b3):
    
    b2 = a1*x_break1 + b1 - a2*x_break1 # to make sure they are connected 
    b3 = a2*x_break2 + b2 - a2*x_break2 # to make sure they are connected 
    y = torch.where(x <= x_break1, a1*x + b1, a2*x + b2)
    y = torch.where(x >= x_break2, a3*x + b3)
    return y


# def broken_linear_fixed_break(x, x_break, a1, b1, a2, b2):
#     b2 = a1*x_break + b1 - a2*x_break # to make sure they are connected 
#     y = torch.where(x <= x_break, a1*x + b1, a2*x + b2)
#     return y


    
def linear(x, a1, b1):
    y = a1*x + b1
    return y

def test_broken():
    x = torch.tensor(np.arange(-10,10.), requires_grad=True)
    x_break = torch.tensor([1.], requires_grad=True)
    a1 = torch.tensor([-2.], requires_grad=True)
    b1 = torch.tensor([0.], requires_grad=True)
    a2 = torch.tensor([3.], requires_grad=True)
    b2 = torch.tensor([6.], requires_grad=True)
    y = broken_linear(x, x_break, a1, b1, a2, b2)
    
    fig, ax = plt.subplots(1,1)
    ax.plot(x.detach().numpy(), y.detach().numpy(), '-o')

    return 


def mse(yhat, y):
    sigma = torch.sum((yhat - y)**2)
    return sigma/len(y)



mus = np.array([]) # array of proper motions from the fits

for f in files:
    

    z = 0.902
    fig, ax = plt.subplots(2,1, figsize=[6,12])
    a = re.search("\/([ABDT]\d*)\.", f)
    component = a.group(1)
    
    print('===============================')
    print('COMPONENT {}'.format(component))
    print('===============================')

    
    
    df = read_comp(f)
    
    # df = df[df.r <= 1.0]
    
    # separation vs time    
    ax[0].plot(df.date, df.r, 'o', label=component, color=colors[component])
    ax[0].set_xlabel('Date [years]')
    ax[0].set_ylabel('Separation from the core [mas]')
    # ax[0].set_ylim([0, 1.0])
    # x = df.loc[df.r <= 10.4, 'date'].values 
    # y = df.loc[df.r <= 10.4, 'r'].values
    # y_err = df.loc[df.r <= 10.4, 'r_err'].values
    # popt, pcov = curve_fit(func, x, y, sigma=y_err)
    # perr = np.sqrt(np.diag(pcov))
    # x_fit = np.array([x.min(), x.max()])
    # y_fit = func(x_fit, *popt)
    # ax[0].plot(x_fit, y_fit, '-', label='ax+b: a = {:.2f} +- {:.2f} mas/year'.format(popt[0], perr[0]))
    # mus = np.append(mus, popt[0])
    
    # print('INNER 0.4 mas: ax+b : a = {:.2f} +- {:.2f} mas/year'.format(popt[0], perr[0]))

    
    # fit with pytorch
    # bring dates to zero
    # first fit a single linear slope with pytorch
    refdate = df.date.min()
    x = torch.tensor(df.date.values - refdate ).requires_grad_()
    y = torch.tensor(df.r.values).requires_grad_()
    a1 = torch.tensor([1.0], requires_grad=True)
    b1 = torch.tensor([0.0], requires_grad=True)
    yhat = linear(x, a1, b1)
    C = mse(yhat, y)
    C.backward()
    optimizer = torch.optim.SGD([a1, b1], lr=1e-2)
    optimizer.step()
    
    line, = ax[0].plot(x.detach().numpy() + refdate, yhat.detach().numpy(), 'b--')
    
    
    epochs = int(1000)
    for epoch in range(epochs):
        optimizer.zero_grad()
        yhat = linear(x, a1, b1)
        C = mse(yhat, y)
        C.backward()
        optimizer.step()
        
        if (epoch > 0 and int(np.log2(epoch)) == np.log2(epoch)) or epoch == epochs-1:
            line.set_ydata(yhat.detach().numpy())
            fig.canvas.draw()
            fig.canvas.flush_events()
        # print('LINEAR: Epoch {}, cost {:.3f}, a1 grad {:.3f}'.format(epoch, C.item(), a1.grad.item()))
    
    
    line, = ax[0].plot(x.detach().numpy() + refdate, yhat.detach().numpy(), 'b--', label='ax+b: a = {:.2f} mas/year'.format(a1.item()))
    
    print('LINEAR: C = {}'.format(C.item()))
    
    
    # set initial parameters: x_break, a1, b1, a2, b2
    # refdate = df.date.min()
    # x = torch.tensor(df.date.values - refdate ).requires_grad_()
    # y = torch.tensor(df.r.values).requires_grad_()

    xmean = x.mean().detach().numpy()
    x_break = torch.tensor(xmean, requires_grad=True)
    a1 = torch.tensor(a1.detach().numpy(), requires_grad=True)
    a2 = torch.tensor(a1.detach().numpy()+1., requires_grad=True)
    b1 = torch.tensor(b1.detach().numpy(), requires_grad=True)
    b2 = torch.tensor(b1.detach().numpy()+1., requires_grad=True)

    yhat = broken_linear(x, x_break, a1, b1, a2, b2)
    line1, = ax[0].plot(x.detach().numpy() + refdate, yhat.detach().numpy(), 'r--')
    
    C = mse(yhat, y)
    C.backward()
    optimizer = torch.optim.SGD([x_break, a1, b1, a2], lr=1e-2)
    optimizer.step()

    C = mse(broken_linear(x, x_break, a1, b1, a2, b2), y)
    
    epochs = int(1e4)
    for epoch in range(epochs):
        optimizer.zero_grad()
        yhat = broken_linear(x, x_break, a1, b1, a2, b2)
        C = mse(yhat, y)
        C.backward()
        optimizer.step()
        if epoch > 0 and int(np.log2(epoch)) == np.log2(epoch):
            line1.set_ydata(yhat.detach().numpy())
            fig.canvas.draw()
            fig.canvas.flush_events()

        # print('Epoch {}, cost {:.3f}, a1 grad {:.3f}, a2 grad {:.3f}'.format(epoch, C.item(), a1.grad.item(), a2.grad.item()))
    
    
    
    ax[0].plot(x.detach().numpy() + refdate, yhat.detach().numpy(), 'r--' , label='a1={:.2f}, a2={:.2f}'.format(a1.item(), a2.item()))
    # ax[0].axvline(xmean+refdate)

    
    print('BROKEN LINEAR: C = {}'.format(C.item()))
    print('Component {}. Break occurs at {:.2f} mas'.format(component, (a1*x_break + b1).item()))
    # date_break = df.loc[(df.r - x_break.item()).abs() ==  (df.r - x_break.item()).abs().min(), 'date'].values
    # print('              Break occurs on {}'.format(date_break))
    print('              Break occurs on {:.2f}'.format(x_break.item()+refdate))
    print('Proper motion mu = BEFORE: {:.2f}, AFTER: {:.2f}'.format(a1.item(), a2.item()))
    print('beta_app = BEFORE: {:.2f} , AFTER = {:.2f}'.format(a1.item()*7.82*3.26*(1+z), a2.item()*7.82*3.26*(1+z)))
    
    print('Speed-up factor = {:.1f}'.format(a2.item() / a1.item()))
    
    
    fig.legend()





    if True:
        # flux vs time. It is wrong to interpret flux(r) breaks as smth physical. Just a component starts to move faster, that's why the break. 
        ax[1].plot(df.date, df.flux, 'o', label=component, color=colors[component])
        ax[1].set_ylim(df.flux.min(), df.flux.max())
        
        ax[1].set_xlabel('Time [years]')
        ax[1].set_ylabel('Flux [Jy]')
        ax[1].set_yscale('log')  
            
        # fit flux(date) with a broken linear 
        
        # x is defines already
        yf = torch.tensor(np.log10(df.flux.values), requires_grad=True)
        xf_break = torch.tensor(x_break.item(), requires_grad=True)
        # af1 = torch.tensor(a1.detach().numpy(), requires_grad=True)
        # af2 = torch.tensor(a1.detach().numpy()+1., requires_grad=True)
        # bf1 = torch.tensor(b1.detach().numpy(), requires_grad=True)
        # bf2 = torch.tensor(b1.detach().numpy()+1., requires_grad=True)
        
        af1 = torch.tensor([-1.], requires_grad=True)
        af2 = torch.tensor([-2.], requires_grad=True)
        bf1 = torch.tensor([0.], requires_grad=True)
        bf2 = torch.tensor([0.], requires_grad=True)
        
        
        yfhat = broken_linear(x, xf_break, af1, bf1, af2, bf2)
        linef1, = ax[1].plot(x.detach().numpy() + refdate, np.power(10, yfhat.detach().numpy()), 'r--')
    
        CF = mse(yfhat, yf)
        CF.backward()
        optimizer = torch.optim.SGD([xf_break, af1, bf1, af2], lr=1e-2)
        optimizer.step()
    
        CF = mse(broken_linear(x, xf_break, af1, bf1, af2, bf2), yf)
        
        epochs = int(3e4)
        for epoch in range(epochs):
            optimizer.zero_grad()
            yfhat = broken_linear(x, xf_break, af1, bf1, af2, bf2)
            CF = mse(yfhat, yf)
            CF.backward()
            optimizer.step()
            if epoch > 0 and int(np.log2(epoch)) == np.log2(epoch):
                linef1.set_ydata(np.power(10, yfhat.detach().numpy()))
                fig.canvas.draw()
                fig.canvas.flush_events()
    
        
        print('              Break occurs on {:.2f}'.format(xf_break.item() + refdate))

    


    if False:
        # size vs time. Expect to see a break
    
        ax[1].plot(df.date, df['size'], 'o', label=component, color=colors[component])
        ax[1].set_ylim(df['size'].min(), df['size'].max())
        
        ax[1].set_xlabel('Time [years]')
        ax[1].set_ylabel('Size [mas]')
        # ax[1].set_yscale('log')  
            
        # fit flux(date) with a broken linear 
        
        # x is defines already
        x = torch.tensor(df.date.values - refdate ).requires_grad_()
        yf = torch.tensor(df['size'].values, requires_grad=True)
        xf_break = torch.tensor(x_break.item(), requires_grad=True)
        # af1 = torch.tensor(a1.detach().numpy(), requires_grad=True)
        # af2 = torch.tensor(a1.detach().numpy()+1., requires_grad=True)
        # bf1 = torch.tensor(b1.detach().numpy(), requires_grad=True)
        # bf2 = torch.tensor(b1.detach().numpy()+1., requires_grad=True)
        
        af1 = torch.tensor([0.001], requires_grad=True)
        af2 = torch.tensor([1.], requires_grad=True)
        bf1 = torch.tensor([0.1], requires_grad=True)
        bf2 = torch.tensor([-1.], requires_grad=True)
        
        
        yfhat = broken_linear(x, xf_break, af1, bf1, af2, bf2)
        
        linef1, = ax[1].plot(x.detach().numpy() + refdate, yfhat.detach().numpy(), 'r--')
    
        CF = mse(yfhat, yf)
        CF.backward()
        # optimizer = torch.optim.SGD([xf_break, af1, bf1, af2], lr=1e-2)
        optimizer = torch.optim.SGD([af1, bf1, af2], lr=1e-2)
        optimizer.step()
    
        CF = mse(broken_linear(x, xf_break, af1, bf1, af2, bf2), yf)
        
        epochs = int(3e4)
        for epoch in range(epochs):
            optimizer.zero_grad()
            yfhat = broken_linear(x, xf_break, af1, bf1, af2, bf2)
            CF = mse(yfhat, yf)
            CF.backward()
            optimizer.step()
            if epoch > 0 and int(np.log2(epoch)) == np.log2(epoch):
                linef1.set_ydata(yfhat.detach().numpy())
                
                fig.canvas.draw()
                fig.canvas.flush_events()
    
        
        print('              Break occurs on {:.2f}'.format(xf_break.item() + refdate))
    
        

    
    
    fig.tight_layout()



# calculate beta from mu
# beta = mu [mas/year] * 7.82 [pc/mas]* 3.26 [ly/pc] * (1+z) = [ly/year] -- value in speeds of light 
# (1+z) factor comes from the luminosity distance to angular distance relation. 
z=0.902

betas = mus * 7.82*3.26*(1+z)
beta = np.median(betas)
print('beta in 0--0.4 mas region is {:.1f}'.format(beta))

# same components but flux(r)


# for f in files:
#     fig, ax = plt.subplots(1,1)
#     a = re.search("\/([ABDT]\d*)\.", f)
#     component = a.group(1)
#     df = read_comp(f)
    
#     ax.plot(df.r, df.flux, 'o', label=component, color=colors[component])
#     ax.set_xlabel('Separation from the core [mas]')
#     ax.set_ylabel('Flux [Jy]')
#     # ax.set_ylim([0, 1.0])
    
#     x = df.loc[df.r <= 0.4, 'r'].values
#     y = df.loc[df.r <= 0.4, 'flux'].values
#     y_err = df.loc[df.r <= 0.4, 'flux_err'].values
    
#     # popt, pcov = curve_fit(func, x, y, sigma=y_err)
#     # perr = np.sqrt(np.diag(pcov))
#     # x_fit = np.array([x.min(), x.max()])
#     # y_fit = func(x_fit, *popt)
    
#     # ax.plot(x_fit, y_fit, '-', label='ax+b: a = {:.2f} mas/year'.format(popt[0]))
    
#     # mus = np.append(mus, popt[0])
#     fig.legend()









# # all in one plot, shifted and stretched in time
# fig, ax = plt.subplots(1,1)
# # timeshift = 
# for f in files:
#     a = re.search("\/([ABDT]\d*)\.", f)
#     component = a.group(1)
#     df = read_comp(f)
    
#     ax.plot((df.date - df.date.min()) / (df.date.max() - df.date.min()), df.r, 'o', label=component, color=colors[component])
    
#     ax.set_xlabel('Date [years]')
#     ax.set_ylabel('Separation from the core [mas]')
#     ax.set_ylim([0, 1.0])
    
#     fig.legend()









# # 1730-130_accelerations.csv  1730-130_gaussians.tab     1730-130_stationary.csv  1730-130_structure.tab  1730_Rplot.eps  1730_Tplot.eps  DifmapModels  Knots  README

base = '/homes/mlisakov/data/nrao530/data/bu_kinematics'

phys = pd.read_csv('{}/1730-130_physicalparameters.csv'.format(base))

speed = pd.read_csv('{}/1730-130_speeds.csv'.format(base)) 


# plot apparent speed of components in a given range of separations from the core as a function of the position angle. 


figt, axt = plt.subplots(1,1)

# inner 0.5 mas
knots = ['B1', 'B2', 'B3', 'B5']
segments = [1, 1, 1, 1]
df005  =pd.DataFrame([])
for i,k in enumerate(knots):
    # axt.plot(speed.loc[(speed.Knot == k) & (speed.Segment == segments[i]), 'phi' ], speed.loc[(speed.Knot == k) & (speed.Segment == segments[i]), 'beta'], '--o')
    df005 = df005.append(speed.loc[(speed.Knot == k) & (speed.Segment == segments[i]), : ])

df005.sort_values('phi', inplace=True)
axt.plot(df005.phi, df005.beta, 'r-o', label='0.0 - 0.5 mas')


# 0.5 -- 1 mas
knots = ['B1', 'B2', 'B3', 'B5', 'B6']
segments = [2, 1, 3, 1, 1]
df0510  =pd.DataFrame([])
for i,k in enumerate(knots):
    # axt.plot(speed.loc[(speed.Knot == k) & (speed.Segment == segments[i]), 'phi' ], speed.loc[(speed.Knot == k) & (speed.Segment == segments[i]), 'beta'], '-s')
    df0510 = df0510.append(speed.loc[(speed.Knot == k) & (speed.Segment == segments[i]), : ])

df0510.sort_values('phi', inplace=True)
axt.plot(df0510.phi, df0510.beta, 'g-o', label='0.5 - 1.0 mas')

# 1 -- 1.5 mas
knots = ['B3', 'B4', 'B6']
segments = [4, 1, 1]
df1015  =pd.DataFrame([])
for i,k in enumerate(knots):
    # axt.plot(speed.loc[(speed.Knot == k) & (speed.Segment == segments[i]), 'phi' ], speed.loc[(speed.Knot == k) & (speed.Segment == segments[i]), 'beta'], '-s')
    df1015 = df1015.append(speed.loc[(speed.Knot == k) & (speed.Segment == segments[i]), : ])

df1015.sort_values('phi', inplace=True)
axt.plot(df1015.phi, df1015.beta, 'b-o', label='1.0 - 1.5 mas')















# fig, ax = plt.subplots(1,1)
# ax.set_xlim(-0.5,0.5)
# for i in speed.index:
#     print('Component {}'.format(speed.loc[i, 'Knot']))
    
#     if speed.loc[i, 'Segment'] > 1:
#         print(speed.loc[i,:])
#         x0 = x0+dx
#         y0 = y0+dy  # plot the vector from the end of the previous one if the same c omponent
#     else:
#         x0=0
#         y0=0
        
#     dx = speed.loc[i, 'mu'] * np.cos(np.deg2rad(speed.loc[i, 'phi']+90))
#     dy = speed.loc[i, 'mu'] * np.sin(np.deg2rad(speed.loc[i, 'phi']+90))
    
#     print(x0, y0, dx, dy)
    
#     ax.arrow(x0,y0, dx,  dy)
#     ax.text(x0+dx, y0+dy, speed.loc[i, 'Knot'])






# gau = pd.read_csv('{}/1730-130_gaussians.tab'.format(base), sep='\s+')
# gau.loc[:, 'x'] = gau.loc[:, 'R'] * np.cos(np.deg2rad(gau.loc[:, 'theta']+90)) 
# gau.loc[:, 'y'] = gau.loc[:, 'R'] * np.sin(np.deg2rad(gau.loc[:, 'theta']+90)) 

# fig,ax = plt.subplots(1,1)

# colors = cm.rainbow(np.linspace(0, 1, len(gau.Epoch.unique())))
# epochs = gau.Epoch.unique()

# epocol = pd.DataFrame(index=epochs, data=colors)

# # ax.plot(gau.x, gau.y, 'o')

# for i,e in enumerate(epochs):
#     idx = gau.loc[gau.Epoch == e].index    
#     ax.scatter(gau.loc[idx, 'x'], gau.loc[idx, 'y'], color=colors[i])
    
# ax.scatter(gau.x, gau.y, c=)

# for i in gau.index:
    
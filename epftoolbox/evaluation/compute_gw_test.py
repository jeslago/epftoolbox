import os
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt
import cmap
from gw import gwtest

def mean_daily_errors(forecast, price):
    return np.mean(np.abs(forecast - price), axis=1)

dataset = 'PJM'

data = pd.read_csv(os.path.join('Final forecasts', f'Top1_All_forecasts_{dataset}.csv'), index_col=0)

# data['DNN 1_'] = data['DNN 2'].copy()
# data['DNN 2'] = data['DNN 1'].copy()
# data['DNN 1'] = data['DNN 1_'].copy()
# del data['DNN 1_'] 

errors = data.copy()

errors = errors.iloc[::24, :]
del errors['Real price']

errors = errors[['Lasso 56', 'Lasso 84', 'Lasso 1092', 'Lasso 1456', 'Lasso Ensemble',
                 'DNN 1', 'DNN 2', 'DNN 3', 'DNN 4', 'DNN Ensemble']]
for model in errors.columns:
    errors.loc[:, model] = mean_daily_errors(data[model].to_numpy().reshape(-1, 24), 
                                             data['Real price'].to_numpy().reshape(-1, 24))


pvals = pd.DataFrame(index=errors.columns, columns=errors.columns)

for model1 in pvals.index:
    for model2 in pvals.columns:
        pvals.loc[model1, model2] = gwtest(errors.loc[:, model1].values, errors.loc[:, model2].values)

cmap = mpl.colors.ListedColormap(cmap.colors)

labels = [r'LEAR$_{56}$', r'LEAR$_{84}$', r'LEAR$_{1092}$', r'LEAR$_{1456}$', r'LEAR$_\mathrm{ens}$',
          r'DNN$_{1}$', r'DNN$_{2}$', r'DNN$_{3}$', r'DNN$_{4}$', r'DNN$_\mathrm{ens}$']

ticklabels = [r'$\textrm{' + e + '}$' for e in labels]



# fig = plt.figure(figsize=[4.7, 4])
# if dataset not in {'PJM', 'FR', 'DE'}:
#     # left panels
#     ax = plt.axes([.27, .22, .7, .7])
# else:
#     # right panel
#     ax = plt.axes([.03, .22, .9, .7])
# mappable = plt.imshow(np.float32(pvals.values), cmap=cmap, vmin=0, vmax=0.1)

# import tikzplotlib
# tikzplotlib.save("NP1.tex")

# plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], ticklabels, rotation=90.)

# if dataset not in {'PJM', 'FR'}:
#     plt.yticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], ticklabels)
# else:
#     plt.yticks([])

# plt.plot(list(range(10)), list(range(10)), 'wx')

# if dataset in {'PJM', 'FR', 'DE'}:
#     plt.colorbar(mappable)
# plt.title(r'$\textrm{' + dataset + '}$')

# # plt.savefig(f'GW_{dataset}.png', dpi=300)
# plt.savefig(f'GW_{dataset}.eps')
# plt.show()


plt.rc('text', usetex=True)
plt.rc('axes', labelsize=14)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=14)    # fontsize of the tick labels
plt.rc('ytick', labelsize=14)    # fontsize of the tick labels
plt.rc('axes', titlesize=16)  # fontsize of the figure title


fig = plt.figure(figsize=[4.7, 4])
ax = plt.axes([.27, .22, .7, .7])

mappable = plt.imshow(np.float32(pvals.values), cmap=cmap, vmin=0, vmax=0.1)


plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], ticklabels, rotation=90.)
plt.yticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], ticklabels)


plt.plot(list(range(10)), list(range(10)), 'wx')
plt.colorbar(mappable)
plt.title(r'$\textrm{' + dataset + '}$')



# plt.savefig(f'GW_{dataset}.png', dpi=300)
plt.savefig(f'GW_{dataset}.eps')
plt.show()


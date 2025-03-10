import os
import pandas as pd
import matplotlib.pyplot as plt


def extractor(cluster, path_data='data/', dyn='Dyn/'):
    if cluster == 'gc': cluster_path = 'GC_chi01_output_noclusterevolv/'
    elif cluster == 'nsc': cluster_path = 'NSC_chi01_output_noclusterevolv/'
    elif cluster == 'ysc': cluster_path = 'YSC_chi01_output_noclusterevolv/'
    holder = []
    for z in os.listdir((new_path := path_data + cluster_path + dyn)):
        dfz = pd.read_csv(new_path + z + '/nth_generation.txt',
                      delimiter=' ',
                      skiprows=1,
                      usecols=[0,1, 2, 3, 4, 9, 13, 14, 15, 16, 17, 25, 27],
                      names=['ID', 'M1', 'M2', 'S1', 'S2', 't_pair', 't_elapsed','kick','Mrem', 'Srem', 'esca_v', 'Mcluster', 'gen'])
        dfz = dfz[dfz['t_elapsed'] != 136000]
        dfz['Z'] = float(z)
        dfz['ID'] = dfz['ID'] + dfz['Z']
        holder.append(dfz)
    df = pd.concat(holder, ignore_index=True)
    df['gen_max'] = df.groupby('ID')['gen'].transform('max')
    return df

#def RFPlot(forest, train_feats):
#    fig, ax = plt.subplots(1, 2, figsize=(20, 7))
#    
#    importances = pd.Series(forest.feature_importances_, index=list(train_feats)).sort_values(ascending=False)
#    importances.plot.bar(yerr=np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0), capsize=3, ax=ax[0])
#    plt.ylim(bottom=0)
#    plt.show()
#    
#    forest.fit(x_train[(important_features := list(importances.index)[:2])], y_train)
#    display = DecisionBoundaryDisplay.from_estimator(forest, x_test[important_features], response_method="predict", alpha=.5, ax=ax[1])
#    display.ax_.scatter(x_test[important_features[0]], x_test[important_features[1]], edgecolor='k', c=y_test, lw=0.5, marker='.')
#    plt.show()
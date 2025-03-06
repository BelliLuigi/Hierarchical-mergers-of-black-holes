import os
from pandas import read_csv, concat
import matplotlib.pyplot as plt
import matplotlib.colors as colors
#from matplotlib.colors import TwoSlopeNorm
import numpy as np



def extractor(cluster, path_data='data/', dyn='Dyn/'):  # This function gathers all the data in one single Dataframe specific for a type of Cluster
    if cluster == 'gc': cluster_path = 'GC_chi01_output_noclusterevolv/'
    elif cluster == 'nsc': cluster_path = 'NSC_chi01_output_noclusterevolv/'
    elif cluster == 'ysc': cluster_path = 'YSC_chi01_output_noclusterevolv/'
    holder = []
    for z in os.listdir((new_path := path_data + cluster_path + dyn)):
        dfz = read_csv(new_path + z + '/nth_generation.txt',
                      delimiter=' ',
                      skiprows=1,
                      usecols=[0,1, 2, 3, 4, 9, 13, 14, 15, 16, 17, 25, 27],
                      names=['ID', 'M1', 'M2', 'S1', 'S2', 't_pair', 't_elapsed','kick','Mrem', 'Srem', 'esca_v', 'Mcluster', 'gen'])
        dfz = dfz[dfz['t_elapsed'] != 136000]
        dfz['Z'] = float(z)
        holder.append(dfz)
    return concat(holder, ignore_index=True)



def hist2dgraph( x,y, nxbin,nybin, title, axx, axy, vmin, vmax, scale_x='linear', scale_y = 'linear'):
    #
    #This Function makes a 2D histogram, binning and showing the ax scales either linear scale or log scale. This method is to prefer again a scatter plot
    # bc it shows the info of density better.
    # x = data.COL1
    # y = data.Col2
    # nxbin, nybin = number of desired bins
    # title = title of graphs, must be a string
    # axx = X label, must be a string
    # axy = Y label, string
    # v min = lower threshold on the frequency, usually 1 but we could consider to increase it by some orders of magnitude, especially when dealing with NSC
    # vmax = upper threshold on frequency/density. Suggested values: {ysc: 1e3, gc : 1e6, nsc: 1e7}
    # scale_x, scale_y = mode of binning the interval and of displaying ticks
    # 
    #This is useful with a colorbar on the side, to add colorbar, do:
    #cbar_ax = fig.add_axes([0.92, 0.11, 0.02, 0.77])  # Adjust position [left, bottom, width, height]
    #fig.colorbar(hist[3], cax=cbar_ax, label="Count Density")
    #plt.show()
    #
    # Something like that. Be aware that this does not work w/ graphs that have to deal w/ Z. 
    # You can furter customize the graph by adding lines after the function has been called.
    #
    if scale_x == 'linear':
        xbin= np.linspace(x.min(),x.max(),nxbin)
        plt.xscale('linear')
    elif scale_x == 'log':
        xbin= np.logspace(np.log10(x.min()),np.log10(x.max()),nxbin)
        plt.xscale('log')
    if scale_y == 'linear':
        ybin= np.linspace(y.min(),y.max(),nybin)
        plt.yscale('linear')
    elif scale_y == 'log':
        ybin= np.logspace(np.log10(y.min()),np.log10(y.max()),nybin)
        plt.yscale('log')
    hist = plt.hist2d(x,y,bins=(xbin,ybin), cmap="Blues", norm=colors.LogNorm( vmin=vmin, vmax=vmax))
    plt.title(title)
    plt.xlabel(axx)
    plt.ylabel(axy)
    return hist

def normer_col(df,i): # This is a normer and it needs to the following function: fast_covariance_matrix
    media = df[i].mean()
    std = df[i].std()
    coso = (df[i] - media)/std
    return coso


def fast_covariance_matrix(cluster, colormap, list_of_reordered_columns=['M1', 'M2','Mrem','Srem','S2','S1','gen','Z', 'kick', 't_elapsed','t_pair', 'esca_v', 'Mcluster'] ):
    #####
    # Suggested colormap w/ 3 colors to make the center 0, so uncorrelation, more visible. Like 'bwr'
    #####
    
    df = extractor(cluster).drop(columns='ID')
    df = df[list_of_reordered_columns]
    heade = list(df.columns)

    # Normalize the DataFrame
    normalized_df = (df - df.mean()) / df.std()
    TwoSlopeNorm = colors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    # Compute the covariance matrix
    cov_matrix = np.cov(normalized_df.values, rowvar=False)
    norm = TwoSlopeNorm
    # Plotting the covariance matrix
    plt.imshow(cov_matrix, cmap=colormap, norm = norm)
    plt.colorbar(label='Covariance')#, ticks=np.arange(-0.99,1.01,0.25), boundaries=np.linspace(-1,1.01,200))#, boundaries=np.linspace(-1,1,2000))
    cluster = cluster.upper()
    plt.title(f'Covariance Matrix for {cluster}')
    plt.xticks(ticks=np.arange(len(heade)), labels=heade, rotation=50)
    plt.yticks(ticks=np.arange(len(heade)), labels=heade)

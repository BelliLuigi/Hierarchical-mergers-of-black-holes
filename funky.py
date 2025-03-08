import os
from pandas import read_csv, concat


def extractor(cluster, path_data='data/', dyn='Dyn/'):
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
        dfz['ID'] = dfz['ID'] + dfz['Z']
        holder.append(dfz)
    df = concat(holder, ignore_index=True)
    df['gen_max'] = df.groupby('ID')['gen'].transform('max')
    return df

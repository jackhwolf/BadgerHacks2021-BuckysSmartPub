if __name__ == '__main__':
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler

    beers = pd.read_csv('beers.csv', index_col=0)
    beers = beers.dropna()
    names, beers = beers['name'], beers[['abv', 'ibu']]
    if 'style' in beers.columns:
        s = beers['style']
        beers['style'] = OrdinalEncoder().fit_transform(s.values.reshape(-1,1))
    beers = beers.values
    for j in range(beers.shape[1]):
        beers[:,[j]] = MinMaxScaler().fit_transform(beers[:,[j]]) * 2 - 1
    np.save('beers_processed.npy', beers)
    np.save('beers_processed_names.npy', np.array(names))

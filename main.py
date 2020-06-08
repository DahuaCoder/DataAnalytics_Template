import numpy as np                      # User guide: https://numpy.org/doc/stable/user/index.html
import pandas as pd                     # User guide: https://pandas.pydata.org/docs/user_guide/index.html
import matplotlib.pyplot as plt         # example charts: https://matplotlib.org/gallery/index.html
import seaborn as sns                   # example charts: https://seaborn.pydata.org/examples/index.html
import plotly.graph_objects as go       # required for interactive plotly charts: https://plotly.com/python/getting-started/
pass                                    # user guide / API https://seaborn.pydata.org/api.html


def read_data():
    # read in data
    df = pd.read_csv('input/911.csv')

    print(df.head())
    print(df.info())

    # What are the top 5 zip codes for 911 calls??
    print(df['zip'].value_counts().head(5))

    # What are the top 5 townships (twp) for 911 calls?
    print(df['twp'].value_counts().head(5))

    # Take a look at the 'title' column, how many unique title codes are there?
    df['title'].nunique()
    df['title'].unique()


def cleanse_and_check_data():
    # groupby
    data = {'Company': ['GOOG', 'GOOG', 'MSFT', 'MSFT', 'FB', 'FB'],
            'Person': ['Sam', 'Charlie', 'Amy', 'Vanessa', 'Carl', 'Sarah'],
            'Sales': [200, 120, 340, 124, 243, 350]}
    df = pd.DataFrame(data)
    by_comp = df.groupby("Company")
    by_comp.mean()
    by_comp.std()
    by_comp.min()
    by_comp.describe()              # print all fields / possible fuctions: count, mean, std, min, 25%, 50%, 75%, max

    # Pivot tables
    data = {'A': ['foo', 'foo', 'foo', 'bar', 'bar', 'bar'],
            'B': ['one', 'one', 'two', 'two', 'one', 'one'],
            'C': ['x', 'y', 'x', 'y', 'x', 'y'],
            'D': [1, 3, 2, 5, 4, 1]}

    df = pd.DataFrame(data)
    df.pivot_table(values='D', index=['A', 'B'], columns=['C'], aggfunc=len)

    # Join: same as the Alteryx join
    left = pd.DataFrame({'A': ['A0', 'A1', 'A2'],
                         'B': ['B0', 'B1', 'B2']},
                        index=['K0', 'K1', 'K2'])

    right = pd.DataFrame({'C': ['C0', 'C2', 'C3'],
                          'D': ['D0', 'D2', 'D3']},
                         index=['K0', 'K2', 'K3'])

    left.join(right)
    left.join(right, how='outer')

    # merge: similar to a join but we can choose on which column to merge
    left = pd.DataFrame({'key1': ['K0', 'K0', 'K1', 'K2'],
                         'key2': ['K0', 'K1', 'K0', 'K1'],
                         'A': ['A0', 'A1', 'A2', 'A3'],
                         'B': ['B0', 'B1', 'B2', 'B3']})

    right = pd.DataFrame({'key1': ['K0', 'K1', 'K1', 'K2'],
                          'key2': ['K0', 'K0', 'K0', 'K0'],
                          'C': ['C0', 'C1', 'C2', 'C3'],
                          'D': ['D0', 'D1', 'D2', 'D3']})

    pd.merge(left, right, how='outer', on=['key1', 'key2'])
    pd.merge(left, right, how='inner', on=['key1', 'key2'])
    pd.merge(left, right, how='right', on=['key1', 'key2'])
    pd.merge(left, right, how='left', on=['key1', 'key2'])

    # concatenate
    BAC = pd.read_csv('input/BAC.csv')
    C = pd.read_csv('input/C.csv')
    GS = pd.read_csv('input/GS.csv')
    JPM = pd.read_csv('input/JPM.csv')
    MS = pd.read_csv('input/MS.csv')
    WFC = pd.read_csv('input/WFC.csv')

    tickers = 'BAC C GS JPM MS WFC'.split()

    bank_stocks = pd.concat([BAC, C, GS, JPM, MS, WFC], axis=1, keys=tickers)
    bank_stocks.columns.names = ['Bank Ticker', 'Stock Info']

    # set index
    BAC.set_index('Date', inplace=True)                 # set column Date as index
    BAC.reset_index(inplace=True)                 # reset index to 0, 1, 2, ...
    BAC.set_index('Date', inplace=True)                 # set column Date as index as we need this later on



    # indexing
    row_index = BAC.loc['2015-01-02']                   # return row with index 2015-01-02
    row_position = bank_stocks.iloc[0]                  # return row at position 0
    column = BAC['Close']                               # return column 'Close'
    column_multi_index = bank_stocks['BAC']['Close']    # return column BAC/Close
    rows_condition = BAC[BAC['Close'] > 40]             # return all rows where Close > 40

    # multi level indexing
    print(bank_stocks.xs(key='Close', axis=1, level='Stock Info').max())
    print(bank_stocks['BAC']['Close'])

    # find and fill NaN
    df = pd.DataFrame({'A': [1, 2, np.nan],
                       'B': [5, np.nan, np.nan],
                       'C': [1, 2, 3]})
    df.dropna()
    df.dropna(axis=1)
    df.dropna(thresh=2)
    df.fillna(value='FILL VALUE')
    df['A'].fillna(value=df['A'].mean())

    # apply lambda function
    df = pd.DataFrame({'col1': [1, 2, 3, 4], 'col2': [444, 555, 666, 444], 'col3': ['abc', 'def', 'ghi', 'xyz']})
    df['col1'].apply(lambda x : x**2)

    pass


def write_data():
    BAC = pd.read_csv('input/BAC.csv')
    BAC.to_csv('output/BAC.csv')
    pass


def create_DataFrame():
    # DataFrame with 4 columsn A, B, C, and D
    # df = pd.DataFrame(np.random.randn(100, 4), columns='A B C D'.split())

    # DataFrame with two columns 'Category' and 'Values'
    # df2 = pd.DataFrame({'Category': ['A', 'B', 'C'], 'Values': [32, 43, 50]})
    pass


def plot_dataframe_charts():
    # load example datasets
    tips = sns.load_dataset('tips')
    flights = sns.load_dataset('flights')
    iris = sns.load_dataset('iris')

    # Histogram
    # tips['total_bill'].hist()

    # barplot (stacked)
    # df = pd.DataFrame(np.random.rand(5, 4) * 10, index='A B C D E'.split(), columns='W X Y Z'.split())
    # df.plot.bar(stacked=True)

    # scatter plot
    # df = pd.DataFrame(np.random.randn(900, 3), columns=['a', 'b', 'c'])
    # df.plot.scatter(x='a',y='b',c='c',cmap='coolwarm')
    # df.plot.scatter(x='a', y='b', s=df['c'] * 200)

    # Hex bins
    # df = pd.DataFrame(np.random.randn(1000, 2), columns=['a', 'b'])
    # df.plot.hexbin(x='a', y='b', gridsize=25, cmap='Oranges')

    ###################################################################################################
    # show charts
    # plt.show()
    pass


def plot_seaborn_charts():
    # load example datasets
    tips = sns.load_dataset('tips')
    flights = sns.load_dataset('flights')
    iris = sns.load_dataset('iris')

    ###################################################################################################
    # Distribution plots
    # sns.distplot(tips['total_bill'],rwidth=0.9)
    # sns.distplot(tips['total_bill'],kde=False,bins=30,rwidth=0.9)
    # sns.jointplot(x='total_bill',y='tip',data=tips,kind='scatter')
    # sns.jointplot(x='total_bill',y='tip',data=tips,kind='hex')

    ###################################################################################################
    # Grid plots
    # pairplot - plot pairwise relationships across an entire dataframe
    # sns.pairplot(tips,hue='sex',palette='coolwarm')

    # PairGrid: map to upper,lower, and diagonal
    # g = sns.PairGrid(iris)
    # g.map_diag(plt.hist)
    # g.map_upper(plt.scatter)
    # g.map_lower(sns.kdeplot)

    # FacetGrid
    # g = sns.FacetGrid(tips, col="time",  row="smoker",hue='sex')
    # # Notice hwo the arguments come after plt.scatter call
    # g = g.map(plt.scatter, "total_bill", "tip").add_legend()

    ###################################################################################################
    # Categorical plots
    # sns.barplot(x='sex', y='total_bill', data=tips)
    # sns.countplot(x='sex', data=tips)
    # sns.boxplot(x='day', y='total_bill', data=tips, hue='smoker')
    # sns.violinplot(x='day', y='total_bill', data=tips, hue='smoker', split=True)
    # sns.barplot(x='sex', y='total_bill', data=tips)
    # sns.barplot(x='sex', y='total_bill', data=tips, ci=None)                   # remove error bars

    # sns.swarmplot(x='day',y='total_bill', data=tips, hue='sex')

    ###################################################################################################
    # Heatmaps
    # tc = tips.corr()
    # sns.heatmap(tc, annot=True, cmap='coolwarm')

    # hm = flights.pivot_table(index='month', columns='year', values='passengers')
    # sns.heatmap(hm, cmap='magma',linecolor='white', linewidth=1)

    # sns.clustermap(hm, cmap='coolwarm',standard_scale=1)

    ###################################################################################################
    # Regression plots
    # sns.lmplot(x='total_bill', y='tip', data=tips, col='sex')

    ###################################################################################################
    # Set style and color
    # plt.style.use('ggplot')                  # popular style package for R
    # sns.set_style('darkgrid')                # parameters: {darkgrid, whitegrid, dark, white, ticks}
    # sns.despine()                            # removes spines / axis
    # plt.figure(figsize=(12, 3))              # chart size and format
    # sns.set_context('poster',font_scale=2)   # charts size and font size: parameter: {paper, notebook, talk, poster}
    # colormaps: https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html


    ###################################################################################################
    # show charts
    plt.show()


def plot_interactive():
    # this method uses plotly and cufflinks libraries
    # pip install plotly
    # pip install cufflinks

    # Scatter plot
    # df = pd.DataFrame(np.random.randn(100, 4), columns='A B C D'.split())
    # fig = go.Figure(data=go.Scatter(x=df['A'], y=df['B'], mode='markers'))
    # fig.write_html('scatter_plot.html', auto_open=True)

    # there are lots of other cool interactive charts including 3D graphs
    # https://plotly.com/python/basic-charts/


    # interactive maps
    # data = dict(type='choropleth',
    #             locations=['AZ', 'CA', 'NY'],
    #             locationmode='USA-states',
    #             colorscale='Portland',
    #             text=['text1', 'text2', 'text3'],
    #             marker=dict(line=dict(color='rgb(255,255,255)', width=2)),
    #             z=[1.0, 2.0, 3.0],
    #             colorbar={'title': 'Colorbar Title'})
    # layout = dict(title='Example data by State',
    #               geo=dict(scope='usa',
    #                        showlakes=True,
    #                        lakecolor='rgb(85,173,240)'))
    # choromap = go.Figure(data=[data], layout=layout)
    # choromap.write_html('choromap_plot.html', auto_open=True)

    # interactive world map
    # df = pd.read_csv('2014_World_GDP')
    # data = dict(
    #     type='choropleth',
    #     colorscale = 'ylorbr',
    #     locations=df['CODE'],
    #     z=df['GDP (BILLIONS)'],
    #     text=df['COUNTRY'],
    #     colorbar={'title': 'GDP Billions US'},)
    # layout = dict(
    #     title='2014 Global GDP',
    #     geo=dict(
    #         showframe=False,
    #         projection={'type': 'mercator'}))
    # choromap2 = go.Figure(data=[data], layout=layout)
    # choromap2.write_html('choromap_plot.html', auto_open=True)
    pass


def main():
    cleanse_and_check_data()


if __name__ == "__main__":
    main()

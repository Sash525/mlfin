#
# Python Module with Class 
# for Vectorized Backtesting
# of Momentum-Based Strategies 
# 
import numpy as np
import pandas as pd

print('223344')

class MomVectorBacktester(object):
    ''' Class for vectorized backtesting of
    momentum-based trading strategies.

    Attributes
    ==========
    datafile: str
        path to file with trade data

    symbol: str
        RIC (financial instrument) to work with
    
    start: str
        start date for data selection
    
    end: str
        end date for data selection

    iday: bool
        true, if trade data is intraday

    amount: int, float
        amount to be invested to beginning
    
    tc: float
        proportional transaction costs (e.g., 0.5% = 0.005) per trade
    

    Metods
    ======
    get_data:
        retrieves and prepares the base data set

    run_strategy:
        runs the backtest for momentum-based strategy

    plot_results:
        plot the performance of the strategy compared to the symbol
    '''

    def __init__(self, datafile, symbol, start, end, iday, amount, tc):
        self.datafile = datafile
        self.symbol = symbol
        self.start = start
        self.end = end
        self.iday = iday
        self.amount = amount
        self.tc = tc
        self.results = None
        self.get_data()

    def get_data(self):
        ''' Retrieves and prepares the data
        '''
        
        if self.iday == True:
            raw = pd.read_csv(self.datafile,
            encoding='utf-8',
            index_col=False,
            skiprows=1,
            header= None,
            names= ['date', 'time','open','high', 'low', 'close', 'tv', 'vol', 'spread'],
            sep='\t')
            
            raw['datetime'] = raw['date'] + ' ' + raw['time']
            raw.reset_index(drop=True, inplace=True)
            raw = raw.set_index('datetime')
            raw = pd.DataFrame(raw['close'])
            raw = raw.loc[self.start:self.end]
            raw.rename(columns = {'close': 'price'}, inplace=True)
            raw['return'] = np.log(raw / raw.shift(1))
            self.data = raw
        
        else:
            raw = pd.read_csv(self.datafile,
            encoding='utf-8',
            index_col=0,
            skiprows=1,
            header= None,
            parse_dates=True,
            names= ['open','high', 'low', 'close', 'tv', 'vol', 'spread'],
            sep='\t')
            
            raw = pd.DataFrame(raw['close'])
            raw = raw.loc[self.start:self.end]
            raw.rename(columns = {'close': 'price'}, inplace=True)
            raw['return'] = np.log(raw['price']/raw['price'].shift(1))
            self.data = raw

    def run_strategy(self, momentum=1):
        ''' Backtests the trading strategy
        '''

        self.momentum = momentum
        data = self.data.copy().dropna()
        data['position'] = np.sign(data['return'].rolling(momentum).mean())
        data['strategy'] = data['position'].shift(1) * data['return']
        
        # determine when a trade takes place 
        # определение места, где (когда) происходит сделка
        data.dropna(inplace=True)
        trades = data['position'].diff().fillna(0) != 0

        # subtract transaction costs from return when trade takes place 
        # вычитаем торговые расходы, когда происходит сделка
        data['strategy'][trades] -= self.tc
        data['creturns'] = self.amount * data['return'].cumsum().apply(np.exp)
        data['cstrategy'] = self.amount * \
            data['strategy'].cumsum().apply(np.exp)
        self.results = data

        # absolute performance of the strategy
        aperf = self.results['cstrategy'].iloc[-1]

        # out -/underperformance of the strategy
        # превышение / невыполение стратегии
        operf = aperf - self.results['creturns'].iloc[-1]
        return round(aperf, 2), round(operf, 2)
    
    def plot_results(self):
        '''Plots the cumulative performance of the trading
        strategy compared to the symbol
        '''
        if self.results is None:
            print('No results to plot yet. Run a strategy')
        title = '%s | TC = %.4f' % (self.symbol, self.tc)
        self.results[['creturns', 'cstrategy']].plot(title=title,
        figsize=(18, 8))

if __name__ == '__main__':
    mombt = MomVectorBacktester('XAU=', '2010-1-1', '2020-12-31',
    10000, 0.0)
    print(mombt.run_strategy)
    print(mombt.run_strategy(momentum=2))
    mombt = MomVectorBacktester('XAU=', '2010-1-1', '2020-12-31',
    10000, 0.001)
    print(mombt.run_strategy(momentum=2))







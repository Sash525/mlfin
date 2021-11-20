#
# Python Module with Class
# for Vectorized Backtesting
# of Mean-Reversion Strategies
# 
#
from Momentum_Backtesting import *

class MRVectorBacktester(MomVectorBacktester):
    '''Class for the vectorized backtesting of
    mean reversion-based trading trading strategies.
    
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

    def run_strategy(self, SMA, threshold):
        '''Backtests the trading strategy
        '''
        data = self.data.copy().dropna()
        data['sma'] = data['price'].rolling(SMA).mean()
        data['distance'] = data['price'] - data['sma']
        data.dropna(inplace=True)
        
        # sell signals
        data['position'] = np.where(data['distance'] > threshold,
        -1, np.nan)

        # buy signals
        data['position'] = np.where(data['distance'] < -threshold,
        1, data['position'])

        # crossing current price and SMA (zero distance)
        # пересечение цены закрытия и средней (нулевое расстояние)
        data['position'] = np.where(data['distance'] * data['distance'].shift(1) < 0,
        0, data['position'])
        data['position'] = data['position'].ffill().fillna(0)
        data['strategy'] = data['position'].shift(1) * data['return']

        # determine when a trade takes place
        # определить, когда происходит сделка
        trades = data['position'].diff().fillna(0) != 0

        # subtract transaction costs from return when trade takes place
        # вычитаем торговые издержки, когда происходит торговля
        data['strategy'][trades] -= self.tc
        data['creturns'] = self.amount * \
            data['return'].cumsum().apply(np.exp)
        data['cstrategy'] = self.amount * \
            data['strategy'].cumsum().apply(np.exp)
        self.results = data

        # absolute performance of thr strategy
        aperf = self.results['cstrategy'].iloc[-1]

        # out-/underperformance of strategy
        operf = aperf - self.results['creturns'].iloc[-1]
        return round(aperf, 2), round(operf, 2)
        
if __name__ == '__main__':
    mrbt = MRVectorBacktester('GSX', '2010-1-1', '2020-12-31',
    10000, 0)
    print(mrbt.run_strategy(SMA=25, threshold=5))
    mrbt = MRVectorBacktester('GDX', '2010-1-1', '2020-12-31',
    10000, 0.001)
    print(mrbt.run_strategy(SMA=25, threshold=5))
    mrbt = MRVectorBacktester('GLD', '2010-1-1', '2020-12-31',
    10000, 0.001)
    print(mrbt.run_strategy(SMA=42, threshold=7.5))

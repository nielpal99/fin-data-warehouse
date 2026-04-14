{{ config(materialized='view') }}

SELECT ticker, name, asset_class
FROM (
    VALUES
        ('VTI',      'US Total Market ETF',       'Equity'),
        ('VOO',      'S&P 500 ETF',               'Equity'),
        ('QQQ',      'Nasdaq 100 ETF',             'Equity'),
        ('SPY',      'S&P 500 ETF',               'Equity'),
        ('IWM',      'Russell 2000 ETF',           'Equity'),
        ('BND',      'Total Bond Market ETF',      'Fixed Income'),
        ('TLT',      '20+ Year Treasury ETF',      'Fixed Income'),
        ('GLD',      'Gold ETF',                   'Commodity'),
        ('BTC-USD',  'Bitcoin',                    'Crypto'),
        ('ETH-USD',  'Ethereum',                   'Crypto'),
        ('DX-Y.NYB', 'US Dollar Index',            'Macro')
) AS t(ticker, name, asset_class)

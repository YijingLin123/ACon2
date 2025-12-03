import os, sys
import base64
import hashlib
import hmac
import json
import pandas as pd
#sfrom datetime import datetime, timedelta
import pickle
import time
import numpy as np
from urllib.parse import urlencode

COINGECKO_SYMBOL_MAP = None
COINAPI_PERIOD_ID_MAP = {
    1: '1SEC',
    5: '5SEC',
    10: '10SEC',
    15: '15SEC',
    30: '30SEC',
    60: '1MIN',
    120: '2MIN',
    300: '5MIN',
    600: '10MIN',
    900: '15MIN',
    1800: '30MIN',
    3600: '1HRS',
    7200: '2HRS',
    21600: '6HRS',
    43200: '12HRS',
    86400: '1DAY',
}
ALPHAVANTAGE_INTERVAL_MAP = {
    60: '1min',
    300: '5min',
    900: '15min',
    1800: '30min',
    3600: '60min',
}
KRAKEN_INTERVAL_MAP = {
    60: 1,
    300: 5,
    900: 15,
    1800: 30,
    3600: 60,
    14400: 240,
    86400: 1440,
    604800: 10080,
    1296000: 21600,
}
KRAKEN_PAIR_MAP = {
    ('ETH', 'USD'): 'XETHZUSD',
    ('BTC', 'USD'): 'XXBTZUSD',
    ('ETH', 'EUR'): 'XETHZEUR',
    ('BTC', 'EUR'): 'XXBTZEUR',
}
BITFINEX_RESOLUTION = {
    60: '1m',
    300: '5m',
    900: '15m',
    1800: '30m',
    3600: '1h',
    10800: '3h',
    21600: '6h',
    43200: '12h',
    86400: '1D',
    604800: '7D',
    1209600: '14D',
    2592000: '1M',
}
KUCOIN_INTERVALS = {
    60: '1min',
    180: '3min',
    300: '5min',
    900: '15min',
    1800: '30min',
    3600: '1hour',
    7200: '2hour',
    14400: '4hour',
    21600: '6hour',
    43200: '12hour',
    86400: '1day',
    604800: '1week',
}
def sanity_check(data, time_start, time_end, time_step_sec):
    if time_start is None or time_end is None:
        return data
    t_start = data[0]['time']
    # assert(data[0]['time'] == time_start)
    # for i, d in enumerate(data):
    #     assert d['time'] == t_start + np.timedelta64(time_step_sec, 's')*i, f'time_obs ({d["time"]}) != time_exp ({t_start + np.timedelta64(time_step_sec, "s")*i})'
    # assert(data[-1]['time'] == time_end)
    for i in range(len(data)-1):
        assert data[i]['time'] != data[i+1]['time'], f"should hold {data[i]['time']} != {data[i+1]['time']}"
    
    return data


def request_batch(name, time_start, time_end, time_step_sec, request_func, postprocess_func):
    
    print(f'[{name}] start time =', time_start, ', end time =', time_end)
    data_pk = []

    t_end = time_start
    f_end = False
    while not f_end:
        t_start = t_end
        t_end = t_start

        for _ in range(99): # do not request too much at once
            if t_end + np.timedelta64(time_step_sec, 's') > time_end:
                f_end = True
                break
            t_end += np.timedelta64(time_step_sec, 's')

        print(f'[{name}, request] start time = {t_start}, end time = {t_end}')
        data_req_i = request_func(t_start, t_end, time_step_sec)
        data_pk_i = postprocess_func(data_req_i)
        data_pk += data_pk_i
        t_end += np.timedelta64(time_step_sec, 's')

        time.sleep(0.05)

    return sanity_check(data_pk, time_start, time_end, time_step_sec)
def _load_coingecko_symbol_map():
    global COINGECKO_SYMBOL_MAP
    if COINGECKO_SYMBOL_MAP is not None:
        return COINGECKO_SYMBOL_MAP
    import requests
    resp = requests.get('https://api.coingecko.com/api/v3/coins/list?include_platform=false', timeout=10)
    if resp.status_code != 200:
        raise ValueError(f'获取Coingecko代币列表失败: {resp.status_code} {resp.text}')
    data = resp.json()
    COINGECKO_SYMBOL_MAP = {}
    for item in data:
        symbol = item.get('symbol')
        coin_id = item.get('id')
        if symbol and coin_id and symbol.upper() not in COINGECKO_SYMBOL_MAP:
            COINGECKO_SYMBOL_MAP[symbol.upper()] = coin_id
    if not COINGECKO_SYMBOL_MAP:
        raise ValueError('Coingecko代币列表为空')
    return COINGECKO_SYMBOL_MAP

def get_from_coingecko(pair0, pair1, time_start=None, time_end=None, time_step_sec=None):
    import requests
    if time_start is None or time_end is None or time_step_sec is None:
        raise ValueError('Coingecko 抓取需要提供 time_start, time_end, time_step_sec')

    now_utc = np.datetime64(pd.Timestamp.utcnow(), 's')
    earliest_allowed = now_utc - np.timedelta64(365, 'D')
    if time_end < earliest_allowed:
        raise ValueError('Coingecko公共API仅支持最近365天的数据，请缩短时间范围')
    if time_start < earliest_allowed:
        print(f'[coingecko] 请求起始时间早于允许范围，自动调整为 {earliest_allowed}')
        time_start = earliest_allowed

    symbol_map = _load_coingecko_symbol_map()
    coin_id = symbol_map.get(pair0.upper())
    if coin_id is None:
        raise ValueError(f'在Coingecko代币列表中找不到 {pair0}')
    vs_currency = pair1.lower()

    def to_unix(ts):
        return int(pd.Timestamp(ts).timestamp())

    chunk_days = 90
    chunk_delta = np.timedelta64(chunk_days, 'D')
    data_pk = []
    cur_start = time_start

    while cur_start < time_end:
        cur_end = min(time_end, cur_start + chunk_delta)
        params = {
            'vs_currency': vs_currency,
            'from': to_unix(cur_start),
            'to': to_unix(cur_end)
        }
        url = f'https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart/range'
        resp = requests.get(url, params=params, timeout=15)
        if resp.status_code != 200:
            raise ValueError(f'Coingecko接口返回错误: {resp.status_code} {resp.text}')
        json_data = resp.json()
        prices = json_data.get('prices', [])
        if not prices:
            raise ValueError(f'Coingecko未返回价格数据: {json_data}')

        df = pd.DataFrame(prices, columns=['time', 'price'])
        df['time'] = pd.to_datetime(df['time'], unit='ms', utc=True)
        df.set_index('time', inplace=True)
        df.sort_index(inplace=True)
        freq = f'{int(time_step_sec)}S'
        df_resampled = df.resample(freq).ffill().dropna()
        ts_start = pd.Timestamp(cur_start, tz='UTC')
        ts_end = pd.Timestamp(cur_end, tz='UTC')
        df_resampled = df_resampled[(df_resampled.index >= ts_start) &
                                    (df_resampled.index <= ts_end)]

        for t, price in df_resampled['price'].items():
            if t.to_numpy() < np.datetime64(time_start) or t.to_numpy() > np.datetime64(time_end):
                continue
            data_pk.append({'time': np.datetime64(t.to_datetime64()),
                            'price': float(price)})

        cur_start = cur_end
        time.sleep(0.2)

    if not data_pk:
        raise ValueError('Coingecko未生成任何数据，请检查参数')

    data_pk.sort(key=lambda x: x['time'])
    return sanity_check(data_pk, time_start, time_end, time_step_sec)

def _extract_av_close(entry, market):
    fields = [
        f'4b. close ({market.upper()})',
        f'4a. close ({market.upper()})',
        '4. close',
        '4a. close (USD)',
        '4b. close (USD)',
    ]
    for f in fields:
        val = entry.get(f)
        if val is None:
            continue
        try:
            return float(val)
        except ValueError:
            continue
    return None

# Kraken public OHLC endpoint: https://api.kraken.com/0/public/OHLC
def get_from_kraken(pair0, pair1, time_start=None, time_end=None, time_step_sec=None):
    import requests
    if time_start is None or time_end is None or time_step_sec is None:
        raise ValueError('Kraken 抓取需要提供 time_start, time_end, time_step_sec')
    step = int(time_step_sec)
    interval = KRAKEN_INTERVAL_MAP.get(step)
    if interval is None:
        supported = ', '.join(str(k) for k in sorted(KRAKEN_INTERVAL_MAP.keys()))
        raise ValueError(f'Kraken 只支持以下秒级间隔: {supported}')

    pair = KRAKEN_PAIR_MAP.get((pair0.upper(), pair1.upper()))
    if pair is None:
        pair = f'{pair0.upper()}{pair1.upper()}'
    start_ts = int(pd.Timestamp(time_start).timestamp())
    end_ts = int(pd.Timestamp(time_end).timestamp())
    since_ts = start_ts
    data_pk = []

    while since_ts <= end_ts:
        params = {
            'pair': pair,
            'interval': interval,
            'since': since_ts,
        }
        resp = requests.get('https://api.kraken.com/0/public/OHLC', params=params, timeout=15)
        payload = resp.json()
        errors = payload.get('error', [])
        if errors:
            raise ValueError(f'Kraken 接口返回错误: {errors}')
        result = payload.get('result', {})
        next_since = int(result.get('last', since_ts + step))
        pair_key = next((k for k in result.keys() if k != 'last'), None)
        ohlc_rows = result.get(pair_key, []) if pair_key else []

        for row in ohlc_rows:
            if len(row) < 5:
                continue
            ts = np.datetime64(int(row[0]), 's')
            if ts < time_start or ts > time_end:
                continue
            try:
                close_price = float(row[4])
            except (TypeError, ValueError):
                continue
            data_pk.append({'time': ts, 'price': close_price})

        if next_since <= since_ts:
            break
        since_ts = next_since
        time.sleep(1)

    if not data_pk:
        raise ValueError('Kraken 未返回所需时间范围的数据')

    data_pk.sort(key=lambda x: x['time'])
    return sanity_check(data_pk, time_start, time_end, time_step_sec)


def get_from_bitfinex(pair0, pair1, time_start=None, time_end=None, time_step_sec=None):
    import requests
    if time_start is None or time_end is None or time_step_sec is None:
        raise ValueError('Bitfinex 抓取需要提供 time_start, time_end, time_step_sec')

    step = int(time_step_sec)
    resolution = BITFINEX_RESOLUTION.get(step)
    if resolution is None:
        supported = ', '.join(str(k) for k in sorted(BITFINEX_RESOLUTION.keys()))
        raise ValueError(f'Bitfinex 只支持以下秒级步长: {supported}')

    symbol = f't{pair0.upper()}{pair1.upper()}'
    limit = 1000
    millis_end = int(pd.Timestamp(time_end).timestamp() * 1000)
    millis_start = int(pd.Timestamp(time_start).timestamp() * 1000)
    data_pk = []
    cur_end = millis_end

    session = requests.Session()
    session.headers.update({'User-Agent': 'acon-price-fetcher'})
    while cur_end > millis_start:
        params = {
            'limit': limit,
            'sort': -1,
            'end': cur_end
        }
        interval = resolution
        url = f'https://api-pub.bitfinex.com/v2/candles/trade:{interval}:{symbol}/hist'
        verify = True
        for attempt in range(2):
            try:
                resp = session.get(url, params=params, timeout=15, verify=verify)
                break
            except requests.exceptions.SSLError as exc:
                if not verify or attempt == 1:
                    raise
                print('[bitfinex] SSL 握手失败，尝试关闭证书校验重试一次')
                verify = False
        if resp.status_code != 200:
            raise ValueError(f'Bitfinex 接口返回错误: {resp.status_code} {resp.text}')
        candles = resp.json()
        if not candles:
            break

        for candle in candles:
            if len(candle) < 6:
                continue
            ts_ms = candle[0]
            close_price = candle[2]  # Bitfinex: [MTS, OPEN, CLOSE, HIGH, LOW, VOLUME]
            ts = np.datetime64(pd.Timestamp(ts_ms, unit='ms', tz='UTC').to_datetime64())
            if ts < time_start or ts > time_end:
                continue
            data_pk.append({'time': ts, 'price': float(close_price)})

        earliest_ms = candles[-1][0]
        if earliest_ms <= millis_start:
            break
        cur_end = earliest_ms - step * 1000
        time.sleep(0.5)

    if not data_pk:
        raise ValueError('Bitfinex 未返回所需时间范围的数据')

    data_pk.sort(key=lambda x: x['time'])
    return sanity_check(data_pk, time_start, time_end, time_step_sec)

def get_from_kucoin(pair0, pair1, time_start=None, time_end=None, time_step_sec=None):
    import requests
    if time_start is None or time_end is None or time_step_sec is None:
        raise ValueError('KuCoin 抓取需要提供 time_start, time_end, time_step_sec')

    step = int(time_step_sec)
    interval = KUCOIN_INTERVALS.get(step)
    if interval is None:
        supported = ', '.join(str(k) for k in sorted(KUCOIN_INTERVALS.keys()))
        raise ValueError(f'KuCoin 只支持以下秒级间隔: {supported}')

    quote = 'USDT' if pair1.upper() == 'USD' else pair1.upper()
    symbol = f'{pair0.upper()}-{quote}'
    limit = 1500
    start_ts = int(pd.Timestamp(time_start).timestamp())
    end_ts = int(pd.Timestamp(time_end).timestamp())
    cur_end = end_ts
    data_pk = []

    while cur_end >= start_ts:
        params = {
            'symbol': symbol,
            'type': interval,
            'endAt': cur_end,
            'startAt': start_ts
        }
        resp = requests.get('https://api.kucoin.com/api/v1/market/candles', params=params, timeout=15)
        if resp.status_code != 200:
            raise ValueError(f'KuCoin 接口返回错误: {resp.status_code} {resp.text}')
        payload = resp.json()
        if payload.get('code') != '200000':
            raise ValueError(f'KuCoin 返回错误: {payload}')
        candles = payload.get('data', [])
        if not candles:
            break

        for candle in candles:
            if len(candle) < 7:
                continue
            ts = np.datetime64(int(candle[0]), 's')
            if ts < time_start or ts > time_end:
                continue
            try:
                close_price = float(candle[2])
            except (TypeError, ValueError):
                continue
            data_pk.append({'time': ts, 'price': close_price})

        earliest = int(candles[-1][0])
        if earliest <= start_ts:
            break
        cur_end = earliest - step
        time.sleep(0.2)

    if not data_pk:
        raise ValueError('KuCoin 未返回所需时间范围的数据')

    data_pk.sort(key=lambda x: x['time'])
    return sanity_check(data_pk, time_start, time_end, time_step_sec)

def get_from_alphavantage(pair0, pair1, time_start=None, time_end=None, time_step_sec=None):
    import requests
    if time_start is None or time_end is None or time_step_sec is None:
        raise ValueError('Alpha Vantage 抓取需要提供 time_start, time_end, time_step_sec')

    api_key = (os.environ.get('ALPHAVANTAGE_API_KEY') or
               os.environ.get('ALPHAVANTAGE_KEY') or
               os.environ.get('ALPHAVANTAGE_TOKEN') or
               'demo')
    base_params = {
        'symbol': pair0.upper(),
        'market': pair1.upper(),
        'apikey': api_key
    }

    def fetch(params):
        resp = requests.get('https://www.alphavantage.co/query', params=params, timeout=20)
        if resp.status_code != 200:
            raise ValueError(f'Alpha Vantage 接口返回错误: {resp.status_code} {resp.text}')
        payload = resp.json()
        if 'Error Message' in payload:
            raise ValueError(f'Alpha Vantage 返回错误: {payload["Error Message"]}')
        if 'Note' in payload:
            raise ValueError(f'Alpha Vantage 速率受限: {payload["Note"]}')
        return payload

    step = int(time_step_sec)
    payload = None
    interval = ALPHAVANTAGE_INTERVAL_MAP.get(step)
    if interval:
        params = dict(base_params, function='CRYPTO_INTRADAY', interval=interval, outputsize='full')
        payload = fetch(params)
        info_line = (payload.get('Information') or '').lower()
        if 'premium' in info_line:
            print('[alphavantage] Intraday 接口仅对付费用户开放，自动改用日线数据')
            payload = None
    elif step < 86400:
        print('[alphavantage] 不支持此分辨率，自动改用日线数据')

    if payload is None:
        params = dict(base_params, function='DIGITAL_CURRENCY_DAILY')
        payload = fetch(params)

    series = None
    for key, value in payload.items():
        if isinstance(value, dict) and 'time series' in key.lower():
            series = value
            break
    if not series:
        raise ValueError(f'Alpha Vantage 未返回时间序列: {payload}')

    data_pk = []
    for ts_str, entry in series.items():
        try:
            ts = pd.Timestamp(ts_str, tz='UTC')
        except ValueError:
            continue
        t = np.datetime64(ts.to_datetime64())
        if t < time_start or t > time_end:
            continue
        price = _extract_av_close(entry, pair1)
        if price is None:
            continue
        data_pk.append({'time': t, 'price': price})

    if not data_pk:
        raise ValueError('Alpha Vantage 未返回所需时间范围的数据')

    data_pk.sort(key=lambda x: x['time'])
    return sanity_check(data_pk, time_start, time_end, time_step_sec)

# go to this webpage claim your api key: https://www.alphavantage.co/support/#api-key
def get_from_cryptocompare(pair0, pair1, time_start=None, time_end=None, time_step_sec=None):
    import requests
    if time_start is None or time_end is None or time_step_sec is None:
        raise ValueError('CryptoCompare 抓取需要提供 time_start, time_end, time_step_sec')
    if time_step_sec <= 0:
        raise ValueError('time_step_sec 必须为正')

    fsym = pair0.upper()
    tsym = pair1.upper()
    headers = {}
    api_key = os.environ.get('CRYPTOCOMPARE_API_KEY')
    if api_key:
        headers['authorization'] = f'Apikey {api_key}'

    def to_unix(ts):
        return int(pd.Timestamp(ts).timestamp())

    start_ts = to_unix(time_start)
    end_ts = to_unix(time_end)
    step = int(time_step_sec)

    endpoint_map = {
        60: ('https://min-api.cryptocompare.com/data/v2/histominute', 60),
        3600: ('https://min-api.cryptocompare.com/data/v2/histohour', 3600),
        86400: ('https://min-api.cryptocompare.com/data/v2/histoday', 86400),
    }

    data_pk = []

    if step in endpoint_map:
        url, granularity = endpoint_map[step]
        limit = 2000
        cur_end = end_ts

        while cur_end >= start_ts:
            max_points = max(1, (cur_end - start_ts) // granularity)
            count = min(limit - 1, max_points)
            params = {
                'fsym': fsym,
                'tsym': tsym,
                'toTs': cur_end,
                'limit': count
            }
            resp = requests.get(url, params=params, headers=headers, timeout=10)
            payload = resp.json()
            if resp.status_code != 200 or payload.get('Response') != 'Success':
                raise ValueError(f'CryptoCompare接口返回错误: {payload}')
            candles = payload.get('Data', {}).get('Data', [])
            if not candles:
                break
            for candle in candles:
                t = np.datetime64(candle['time'], 's')
                if t < time_start or t > time_end:
                    continue
                data_pk.append({'time': t, 'price': float(candle['close'])})
            earliest = np.datetime64(candles[0]['time'], 's')
            if earliest <= time_start:
                break
            cur_end = int((earliest - np.timedelta64(granularity, 's')).astype('int'))
            time.sleep(0.2)
    else:
        # fallback to pricehistorical for arbitrary steps
        url = 'https://min-api.cryptocompare.com/data/pricehistorical'
        cur_ts = start_ts
        while cur_ts <= end_ts:
            params = {'fsym': fsym, 'tsyms': tsym, 'ts': cur_ts}
            resp = requests.get(url, params=params, headers=headers, timeout=10)
            payload = resp.json()
            if resp.status_code != 200 or fsym not in payload or tsym not in payload.get(fsym, {}):
                raise ValueError(f'CryptoCompare pricehistorical接口返回错误: {payload}')
            price = payload[fsym][tsym]
            data_pk.append({'time': np.datetime64(cur_ts, 's'), 'price': float(price)})
            cur_ts += step
            time.sleep(0.1)

    if not data_pk:
        raise ValueError('CryptoCompare未返回任何数据，请检查参数或API限制')

    data_pk.sort(key=lambda x: x['time'])
    return sanity_check(data_pk, time_start, time_end, time_step_sec)

    
if __name__ == '__main__':
    market = 'kucoin'
    # market = 'kraken'
    # market = 'bitfinex'
    # market = 'coingecko'
    # market = 'cryptocompare'
    pair0 = 'ETH'
    pair1 = 'USD'
    root = f'price_{pair0}_{pair1}'
    now_utc = pd.Timestamp('2025-12-03T00:00:00Z')
    time_end = np.datetime64('2025-12-02T23:59')
    time_start = np.datetime64('2025-11-27T00:00')
    time_step_sec = 60

    os.makedirs(root, exist_ok=True)
    
    if market == 'coingecko':
        data = get_from_coingecko(pair0, pair1, time_start, time_end, time_step_sec)
        pickle.dump(data, open(os.path.join(root, f'coingecko_{time_start}_{time_end}.pk'), 'wb'))
    elif market == 'cryptocompare':
        data = get_from_cryptocompare(pair0, pair1, time_start, time_end, time_step_sec)
        pickle.dump(data, open(os.path.join(root, f'cryptocompare_{time_start}_{time_end}.pk'), 'wb'))
    elif market == 'kraken':
        data = get_from_kraken(pair0, pair1, time_start, time_end, time_step_sec)
        pickle.dump(data, open(os.path.join(root, f'kraken_{time_start}_{time_end}.pk'), 'wb'))
    elif market == 'kucoin':
        data = get_from_kucoin(pair0, pair1, time_start, time_end, time_step_sec)
        pickle.dump(data, open(os.path.join(root, f'kucoin_{time_start}_{time_end}.pk'), 'wb'))
    elif market == 'bitfinex':
        data = get_from_bitfinex(pair0, pair1, time_start, time_end, time_step_sec)
        pickle.dump(data, open(os.path.join(root, f'bitfinex_{time_start}_{time_end}.pk'), 'wb'))

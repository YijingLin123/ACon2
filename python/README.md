conda create -n acon python=3.9 -y

conda activate acon

pip install -r requirements.txt
pip install pandas
pip install python-binance
pip install cbpro

通过coingecko/yahoo免费抓取过去365天的数据：python data/read_offchain_price.py 

python main.py \
    --exp_name eth_usd_demo \
    --data.name PriceDataset \
    --data.path data/price_ETH_USD/coingecko_2025-02-01T00:00_2025-06-30T23:59.pk \
    --data.start_time 2025-02-01T00:00 \
    --data.end_time 2025-02-01T23:59 \
    --model_ps.beta 1
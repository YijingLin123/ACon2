"""Flask API接口，用于执行agent_aggregation并返回median价格"""

from flask import Flask, request, jsonify
from decimal import Decimal
from agent_aggregation import build_agent_configs, get_aggregated_price

app = Flask(__name__)


@app.route('/aggregate_price', methods=['POST'])
def aggregate_price():
    """
    执行agent聚合并返回median价格。
    
    请求体格式（JSON）:
    {
        "data_paths": [
            "data/price_ETH_USD/coingecko_2025-11-27T00:00_2025-12-02T23:59.pk@1",
            "data/price_ETH_USD/cryptocompare_2025-11-27T00:00_2025-12-02T23:59.pk@1",
            "data/price_ETH_USD/kucoin_2025-11-27T00:00_2025-12-02T23:59.pk@1"
        ],
        "start_time": "2025-11-27T00:00",
        "end_time": "2025-11-27T00:59",
        "mode": "latest",
        "aggregation": "median",
        "currency": "USD"
    }
    
    返回格式:
    {
        "success": true,
        "median_price": "1234.56",
        "currency": "USD"
    }
    """
    try:
        data = request.get_json()
        
        # 解析请求参数
        data_paths = data.get('data_paths', [])
        if not data_paths:
            return jsonify({
                'success': False,
                'error': 'data_paths参数不能为空'
            }), 400
        
        start_time = data.get('start_time')
        end_time = data.get('end_time')
        mode = data.get('mode', 'latest')
        aggregation = data.get('aggregation', 'median')
        currency = data.get('currency', 'USD')
        
        # 构建agent配置
        configs = build_agent_configs(data_paths)
        
        # 获取聚合价格
        timestamp_hint = end_time or start_time
        query = '请查询币价并返回报价'
        
        agg_price = get_aggregated_price(
            query=query,
            agent_configs=configs,
            currency=currency,
            timestamp_hint=timestamp_hint,
            aggregation=aggregation,
            truncation_ratio=0.2,
            mode=mode,
            start_time=start_time,
            end_time=end_time
        )
        
        if agg_price is None:
            return jsonify({
                'success': False,
                'error': '无法获取聚合价格'
            }), 500
        
        # 返回结果
        return jsonify({
            'success': True,
            'median_price': str(agg_price),
            'currency': currency
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/health', methods=['GET'])
def health():
    """健康检查接口"""
    return jsonify({'status': 'ok'})


if __name__ == '__main__':
    # 默认运行在5000端口
    app.run(host='0.0.0.0', port=5000, debug=True)


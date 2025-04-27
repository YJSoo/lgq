from flask import Flask, request, jsonify,render_template
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

app = Flask(__name__)

# 定义重要特征
features = [
    'device_usage_days',
    'avg_monthly_usage_mins',
    'total_calls_lifetime',
    'monthly_usage_pct_change',
    'total_work_months',
    'lifetime_avg_monthly_fee',
    'avg_monthly_fee',
    'avg_incoming_voice_calls',
    'avg_missed_voice_calls',
    'monthly_fee_pct_change',
    'region',
    'current_phone_price'
]

threshold = 0.44  # 最佳阈值
rf_model = None  # 全局模型变量


# 加载并训练模型
def train_model():
    global rf_model
    data = pd.read_csv('data/train2.csv')

    X = data[features]
    y = data['is_churn']

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    rf_model = RandomForestClassifier(
        n_estimators=50,
        max_depth=5,
        random_state=42,
        class_weight='balanced'
    )
    rf_model.fit(X_train, y_train)

    y_pred_proba = rf_model.predict_proba(X_val)[:, 1]
    y_pred = (y_pred_proba >= threshold).astype(int)

    f1 = f1_score(y_val, y_pred)

    print(f"模型训练完成，F1_Score: {round(f1, 4)}，阈值: {threshold}")


@app.before_first_request
def before_first_request_func():
    train_model()

@app.route('/')
def index():
    return render_template('index.html',features=features)
# 单条预测接口
@app.route('/predict_churn/', methods=['POST'])
def predict_churn():
    global rf_model
    if rf_model is None:
        return jsonify({"code": 500, "message": "模型未加载，请稍后再试"}), 500

    data = request.get_json()

    # 检查必要字段
    required_fields = features + ['customer_id', 'expected_income']
    if not all(field in data for field in required_fields):
        return jsonify({"code": 400, "message": "请求数据缺少必要字段"}), 400

    # 构建输入
    input_features = pd.DataFrame([{k: data[k] for k in features}])

    pred_proba = rf_model.predict_proba(input_features)[:, 1][0]
    pred_label = int(pred_proba >= threshold)

    risk = "高" if pred_proba > 0.7 else ("中" if pred_proba > 0.4 else "低")
    customer_type = "高净值客户" if data['expected_income'] >= 5 and data['avg_monthly_fee'] >= 100 else "中小微客户"

    return jsonify({
        "code": 200,
        "message": "success",
        "data": {
            "customer_id": data['customer_id'],
            "预测流失": "是" if pred_label == 1 else "否",
            "流失概率": round(pred_proba, 3),
            "风险评分": risk,
            "客户类型": customer_type
        }
    })


# 批量预测接口
@app.route('/predict_churn_batch/', methods=['POST'])
def predict_churn_batch():
    global rf_model
    if rf_model is None:
        return jsonify({"code": 500, "message": "模型未加载，请稍后再试"}), 500

    data = request.get_json()
    customers = data.get('customers', [])

    if not customers:
        return jsonify({"code": 400, "message": "请求数据缺少 customers 列表"}), 400

    results = []

    for customer in customers:
        if not all(field in customer for field in features + ['customer_id', 'expected_income']):
            continue  # 跳过字段不全的客户

        input_features = pd.DataFrame([{k: customer[k] for k in features}])

        pred_proba = rf_model.predict_proba(input_features)[:, 1][0]
        pred_label = int(pred_proba >= threshold)

        risk = "高" if pred_proba > 0.7 else ("中" if pred_proba > 0.4 else "低")
        customer_type = "高净值客户" if customer['expected_income'] >= 5 and customer['avg_monthly_fee'] >= 100 else "中小微客户"

        results.append({
            "customer_id": customer['customer_id'],
            "预测流失": "是" if pred_label == 1 else "否",
            "流失概率": round(pred_proba, 3),
            "风险评分": risk,
            "客户类型": customer_type
        })

    return jsonify({"code": 200, "message": "success", "data": results})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)

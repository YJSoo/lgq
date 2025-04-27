from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from io import StringIO
from typing import List

app = FastAPI(
    title="电信客户流失预测API",
    docs_url="/docs",  # 明确启用 Swagger UI
    openapi_url="/openapi.json",  # 确保 OpenAPI 规范可访问
)

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
rf_model = None  # 先定义模型变量，后续赋值


# 在应用启动时加载并训练模型
@app.on_event("startup")
async def startup():
    global rf_model

    # 读取训练数据文件 train2.csv
    data = pd.read_csv('train2.csv')

    X = data[features]
    y = data['is_churn']

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 训练模型
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        class_weight='balanced'
    )
    rf_model.fit(X_train, y_train)

    y_pred_proba = rf_model.predict_proba(X_val)[:, 1]
    y_pred = (y_pred_proba >= threshold).astype(int)

    f1 = f1_score(y_val, y_pred)

    print(f"模型训练完成，F1_Score: {round(f1, 4)}，阈值: {threshold}")


# --------- 接口2：单条数据预测 ---------
class CustomerData(BaseModel):
    customer_id: int
    device_usage_days: int
    avg_monthly_usage_mins: float
    total_calls_lifetime: int
    monthly_usage_pct_change: float
    total_work_months: int
    lifetime_avg_monthly_fee: float
    avg_monthly_fee: float
    avg_incoming_voice_calls: float
    avg_missed_voice_calls: float
    monthly_fee_pct_change: float
    region: int
    current_phone_price: float
    expected_income: float  # 注意expected_income不是模型特征，只是传进来的，不参与预测


@app.post("/predict_churn/")
async def predict_churn(customer: CustomerData):
    global rf_model

    if rf_model is None:
        raise HTTPException(status_code=500, detail="模型未加载，请稍后再试")

    # 将输入转换为DataFrame
    input_features = pd.DataFrame([{
        "device_usage_days": customer.device_usage_days,
        "avg_monthly_usage_mins": customer.avg_monthly_usage_mins,
        "total_calls_lifetime": customer.total_calls_lifetime,
        "monthly_usage_pct_change": customer.monthly_usage_pct_change,
        "total_work_months": customer.total_work_months,
        "lifetime_avg_monthly_fee": customer.lifetime_avg_monthly_fee,
        "avg_monthly_fee": customer.avg_monthly_fee,
        "avg_incoming_voice_calls": customer.avg_incoming_voice_calls,
        "avg_missed_voice_calls": customer.avg_missed_voice_calls,
        "monthly_fee_pct_change": customer.monthly_fee_pct_change,
        "region": customer.region,
        "current_phone_price": customer.current_phone_price
    }])

    # 预测流失概率
    pred_proba = rf_model.predict_proba(input_features)[:, 1][0]
    pred_label = int(pred_proba >= threshold)

    # 计算风险等级
    risk = "高" if pred_proba > 0.7 else ("中" if pred_proba > 0.4 else "低")

    # 计算客户类型
    customer_type = "高净值客户" if customer.expected_income >= 5 and customer.avg_monthly_fee >= 100 else "中小微客户"

    return {
        "code": 200,
        "message": "success",
        "data": {
            "customer_id": customer.customer_id,
            "预测流失": "是" if pred_label == 1 else "否",
            "流失概率": round(pred_proba, 3),
            "风险评分": risk,
            "客户类型": customer_type
        }
    }


# 批量预测接口
class CustomersData(BaseModel):
    customers: List[CustomerData]


@app.post("/predict_churn_batch/")
async def predict_churn_batch(customers_data: CustomersData):
    global rf_model

    if rf_model is None:
        raise HTTPException(status_code=500, detail="模型未加载，请稍后再试")

    results = []

    for customer in customers_data.customers:
        input_features = pd.DataFrame([{
            "device_usage_days": customer.device_usage_days,
            "avg_monthly_usage_mins": customer.avg_monthly_usage_mins,
            "total_calls_lifetime": customer.total_calls_lifetime,
            "monthly_usage_pct_change": customer.monthly_usage_pct_change,
            "total_work_months": customer.total_work_months,
            "lifetime_avg_monthly_fee": customer.lifetime_avg_monthly_fee,
            "avg_monthly_fee": customer.avg_monthly_fee,
            "avg_incoming_voice_calls": customer.avg_incoming_voice_calls,
            "avg_missed_voice_calls": customer.avg_missed_voice_calls,
            "monthly_fee_pct_change": customer.monthly_fee_pct_change,
            "region": customer.region,
            "current_phone_price": customer.current_phone_price
        }])

        # 预测流失概率
        pred_proba = rf_model.predict_proba(input_features)[:, 1][0]
        pred_label = int(pred_proba >= threshold)

        # 计算风险等级
        risk = "高" if pred_proba > 0.7 else ("中" if pred_proba > 0.4 else "低")

        # 计算客户类型
        customer_type = "高净值客户" if customer.expected_income >= 5 and customer.avg_monthly_fee >= 100 else "中小微客户"

        results.append({
            "customer_id": customer.customer_id,
            "预测流失": "是" if pred_label == 1 else "否",
            "流失概率": round(pred_proba, 3),
            "风险评分": risk,
            "客户类型": customer_type
        })

    return {"code": 200, "message": "success", "data": results}

# 直接运行时启动 uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000)
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import os
from PIL import Image

# 页面配置
st.set_page_config(
    page_title="胸痛风险预测器",
    page_icon="❤️",
    layout="wide"
)


# 加载模型和选定特征
@st.cache_resource
def load_model():
    model = joblib.load('model/lightgbm_risk_model.pkl')
    selected_features = joblib.load('model/selected_features.pkl')
    return model, selected_features


try:
    model, feature_names = load_model()
    model_loaded = True
except:
    st.error("加载模型失败。请确保模型文件存在于'model'目录中。")
    model_loaded = False

# Streamlit UI
st.title("急性胸痛风险分层")
st.markdown("此应用程序使用LightGBM分类器预测高风险心血管事件。")

# 侧边栏信息
with st.sidebar:
    st.header("关于")
    st.info(
        "这个临床决策支持工具运用机器学习对胸痛患者进行心血管风险分层。"
        "输入患者参数并点击'预测'进行评估。"
    )

    st.header("模型规格")
    st.markdown("""
    - **模型类型**: LightGBM分类器
    - **特征**: 7个临床参数
    - **目的**: 急性胸痛的早期风险分层
    - **验证**: AUC-ROC 0.909 (95% CI 0.887-0.928)
    """)

# 输入表单
st.subheader("患者临床参数")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("年龄(岁):", min_value=18, max_value=120, value=65)
    max_creatinine = st.number_input("肌酐峰值(mg/dL):", min_value=0.0, max_value=15.0, value=1.0, step=0.1)
    max_bun = st.number_input("尿素氮峰值(mg/dL):", min_value=0.0, max_value=200.0, value=20.0, step=0.1)
    max_glucose = st.number_input("血糖峰值(mg/dL):", min_value=0.0, max_value=700.0, value=120.0, step=1.0)

with col2:
    max_potassium = st.number_input("钾离子峰值(mmol/L):", min_value=2.0, max_value=8.0, value=4.0, step=0.1)
    max_troponin = st.number_input("肌钙蛋白峰值(ng/mL):", min_value=0.0, max_value=50.0, value=0.01, step=0.01)
    ethnicity = st.selectbox("种族:",
                             options=["白人", "黑人", "西班牙裔", "亚洲人", "其他"],
                             index=0)


# 种族编码
def encode_ethnicity(ethnicity_value):
    ethnicity_mapping = {
        "白人": "White",
        "黑人": "Black",
        "西班牙裔": "Hispanic",
        "亚洲人": "Asian",
        "其他": "Other"
    }
    
    encoded = {}
    english_value = ethnicity_mapping[ethnicity_value]
    
    ethnicities = ["White", "Black", "Hispanic", "Asian", "Other"]
    for eth in ethnicities:
        if eth != "White":  # 参考类别
            key = f"ethnicity_{eth}"
            encoded[key] = 1 if english_value == eth else 0
    return encoded


if st.button("预测") and model_loaded:
    # 准备输入数据
    ethnicity_encoded = encode_ethnicity(ethnicity)

    input_data = {
        'age': age,
        'max_creatinine': max_creatinine,
        'max_bun': max_bun,
        'max_glucose': max_glucose,
        'max_potassium': max_potassium,
        'max_troponin': max_troponin,
    }
    input_data.update(ethnicity_encoded)

    input_df = pd.DataFrame([input_data])

    # 确保特征对齐
    missing_features = [feat for feat in feature_names if feat not in input_df.columns]
    if missing_features:
        for feat in missing_features:
            input_df[feat] = 0

    input_df = input_df[feature_names]

    # 预测
    try:
        risk_probability = model.predict_proba(input_df)[0][1]
        predicted_class = 1 if risk_probability >= 0.5 else 0

        tab1, tab2 = st.tabs(["风险评估", "模型解释"])

        with tab1:
            st.subheader("风险分层")

            col1, col2 = st.columns([2, 3])

            with col1:
                st.metric(
                    label="事件概率",
                    value=f"{risk_probability:.1%}",
                    delta=None
                )

                if predicted_class == 1:
                    st.error("**高风险**: 发生重大不良心脏事件(MACE)的概率较高")
                else:
                    st.success("**低风险**: 急性冠脉综合征的可能性较低")

            with col2:
                fig, ax = plt.subplots(figsize=(5, 1))
                ax.barh([0], [100], color='lightgray', height=0.4)
                ax.barh([0], [risk_probability * 100],
                        color='#ff4b4b' if risk_probability >= 0.5 else '#2ecc71',
                        height=0.4)
                ax.set_xlim(0, 100)
                ax.set_ylim(-0.5, 0.5)
                ax.set_xticks([0, 25, 50, 75, 100])
                ax.set_yticks([])
                ax.set_xlabel('风险概率 (%)')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_visible(False)
                st.pyplot(fig)

            st.subheader("临床建议")
            if predicted_class == 1:
                st.markdown("""
                - 立即进行心脏科会诊
                - 连续监测心脏生物标志物
                - 每20-30分钟进行一次12导联心电图
                - 考虑进行高级影像学检查(CTA/血管造影)
                - 建议住院监测
                """)
            else:
                st.markdown("""
                - 72小时内进行门诊随访
                - 如有指征进行运动负荷试验
                - 风险因素修正咨询
                - 重新评估非心脏病因
                - 提供胸痛应对计划
                """)

        with tab2:
            st.subheader("特征贡献分析")

            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(input_df)

            if isinstance(shap_values, list):
                shap_values = shap_values[1]

            plt.figure(figsize=(10, 3))
            shap.force_plot(
                explainer.expected_value[1] if isinstance(explainer.expected_value,
                                                          np.ndarray) else explainer.expected_value,
                shap_values,
                input_df,
                matplotlib=True,
                show=False
            )
            plt.tight_layout()
            plt.savefig("temp_shap_plot.png", bbox_inches='tight', dpi=150)
            plt.close()

            st.image("temp_shap_plot.png")

            st.markdown("""
            **解释指南**:
            - 正SHAP值(红色)增加预测风险
            - 负SHAP值(蓝色)降低预测风险
            - 特征影响大小由条形长度表示
            - 基线值代表人群平均风险
            """)

    except Exception as e:
        st.error(f"预测错误: {str(e)}")

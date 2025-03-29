import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import os
from PIL import Image

# 设置页面配置
st.set_page_config(
    page_title="胸痛风险预测器",
    page_icon="❤️",
    layout="wide"
)


# 加载模型和选定的特征
@st.cache_resource
def load_model():
    model = joblib.load('model/lightgbm_risk_model.pkl')
    selected_features = joblib.load('model/selected_features.pkl')
    return model, selected_features


try:
    model, feature_names = load_model()
    model_loaded = True
except:
    st.error("无法加载模型。请确保模型文件存在于'model'目录中。")
    model_loaded = False

# Streamlit用户界面
st.title("高危胸痛预测器")
st.markdown("本应用使用LightGBM模型预测高危胸痛风险。")

# 创建侧边栏信息
with st.sidebar:
    st.header("关于")
    st.info(
        "本应用使用机器学习模型基于患者数据预测高危胸痛风险。"
        "在字段中输入患者信息并点击'预测'获取结果。"
    )

    st.header("模型信息")
    st.markdown("""
    - **模型类型**: LightGBM分类器
    - **特征**: 7个临床参数
    - **目的**: 胸痛患者的早期风险评估
    """)

# 创建输入表单
st.subheader("患者信息")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("年龄:", min_value=18, max_value=120, value=65)
    max_creatinine = st.number_input("最高肌酐值 (mg/dL):", min_value=0.0, max_value=15.0, value=1.0, step=0.1)
    max_bun = st.number_input("最高尿素氮 (mg/dL):", min_value=0.0, max_value=200.0, value=20.0, step=0.1)
    max_glucose = st.number_input("最高血糖值 (mg/dL):", min_value=0.0, max_value=700.0, value=120.0, step=1.0)

with col2:
    max_potassium = st.number_input("最高钾离子值 (mmol/L):", min_value=2.0, max_value=8.0, value=4.0, step=0.1)
    max_troponin = st.number_input("最高肌钙蛋白值 (ng/mL):", min_value=0.0, max_value=50.0, value=0.01, step=0.01)
    ethnicity = st.selectbox("民族:",
                             options=["白人", "黑人", "西班牙裔", "亚洲人", "其他"],
                             index=0)


# 处理民族特征的独热编码
def encode_ethnicity(ethnicity_value):
    # 创建一个字典存储独热编码值
    encoded = {}
    ethnicities = ["白人", "黑人", "西班牙裔", "亚洲人", "其他"]

    for eth in ethnicities:
        if eth != "白人":  # 假设'白人'是参考类别(drop_first=True)
            key = f"ethnicity_{eth}"
            encoded[key] = 1 if ethnicity_value == eth else 0

    return encoded


if st.button("预测") and model_loaded:
    # 准备输入数据
    ethnicity_encoded = encode_ethnicity(ethnicity)

    # 创建包含所有特征的字典
    input_data = {
        'age': age,
        'max_creatinine': max_creatinine,
        'max_bun': max_bun,
        'max_glucose': max_glucose,
        'max_potassium': max_potassium,
        'max_troponin': max_troponin,
    }

    # 添加编码后的民族特征
    input_data.update(ethnicity_encoded)

    # 转换为DataFrame
    input_df = pd.DataFrame([input_data])

    # 确保所有必需的特征都存在
    missing_features = [feat for feat in feature_names if feat not in input_df.columns]
    if missing_features:
        for feat in missing_features:
            input_df[feat] = 0

    # 重新排列列，以匹配模型期望的特征顺序
    input_df = input_df[feature_names]

    # 进行预测
    try:
        risk_probability = model.predict_proba(input_df)[0][1]
        predicted_class = 1 if risk_probability >= 0.5 else 0

        # 创建不同视图的选项卡
        tab1, tab2 = st.tabs(["预测结果", "模型解释"])

        with tab1:
            # 显示带有量表图表的预测结果
            st.subheader("风险评估")

            col1, col2 = st.columns([2, 3])

            with col1:
                st.metric(
                    label="风险概率",
                    value=f"{risk_probability:.1%}",
                    delta=None
                )

                if predicted_class == 1:
                    st.error("**高风险**: 患者有较高的不良心脏事件概率。")
                else:
                    st.success("**低风险**: 患者有较低的不良心脏事件概率。")

            with col2:
                # 创建自定义量表图
                fig, ax = plt.subplots(figsize=(5, 1))
                ax.barh([0], [100], color='lightgray', height=0.4)
                ax.barh([0], [risk_probability * 100], color='red' if risk_probability >= 0.5 else 'green', height=0.4)
                ax.set_xlim(0, 100)
                ax.set_ylim(-0.5, 0.5)
                ax.set_xticks([0, 25, 50, 75, 100])
                ax.set_yticks([])
                ax.set_xlabel('风险概率 (%)')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_visible(False)
                st.pyplot(fig)

            # 临床建议
            st.subheader("临床建议")
            if predicted_class == 1:
                st.markdown("""
                * 考虑立即进行心脏咨询
                * 密切监测生命体征
                * 评估急性冠状动脉综合征
                * 定期检查心电图和心脏生物标志物
                * 如有指征，考虑早期干预
                """)
            else:
                st.markdown("""
                * 继续标准评估
                * 考虑胸痛的非心脏原因
                * 根据临床怀疑进行适当的后续检查
                * 向患者宣教心脏风险因素
                """)

        with tab2:
            # 生成SHAP值
            st.subheader("特征重要性分析")

            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(input_df)

            # 如果SHAP值作为列表返回（对于二分类），选择正类
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # 索引1对应正类

            # 创建并显示SHAP力图
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

            # 显示SHAP图
            st.image("temp_shap_plot.png")

            st.markdown("""
            **如何解读**: 
            * 上面的图表显示了每个特征如何对风险预测产生贡献。
            * 红色特征将预测值推高（增加风险）。
            * 蓝色特征将预测值推低（降低风险）。
            * 每个特征条的大小表示它对这一特定预测的重要性。
            """)

    except Exception as e:
        st.error(f"预测出错: {str(e)}")


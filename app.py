import streamlit as st
import pandas as pd
import numpy as np
import pickle
import time
import os
import plotly.express as px

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Smart Car Price Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 2. LOAD ARTIFACTS
# ==========================================
@st.cache_resource
def load_artifacts():
    """
    Loads all pickled models, preprocessors, and the dataset.
    """
    try:
        with open("models.pkl", "rb") as f: models = pickle.load(f)
        with open("encoder.pkl", "rb") as f: encoder = pickle.load(f)
        with open("scaler.pkl", "rb") as f: scaler = pickle.load(f)
        with open("columns.pkl", "rb") as f: feature_columns = pickle.load(f)
        with open("cat_cols.pkl", "rb") as f: cat_cols = pickle.load(f)
        with open("num_cols.pkl", "rb") as f: num_cols = pickle.load(f)
        with open("metrics.pkl", "rb") as f: metrics = pickle.load(f)
        
        df = pd.read_csv("ford_clean_data.csv")
        return models, encoder, scaler, feature_columns, cat_cols, num_cols, metrics, df
    except Exception as e:
        st.error(f"Error loading artifacts: {e}. Please ensure you have run your training script first.")
        return None, None, None, None, None, None, None, None

models, encoder, scaler, feature_columns, cat_cols, num_cols, metrics, df = load_artifacts()

# Safe metric retriever for MAE, MSE, and R2
def get_metric(model_name, metric_key):
    metric_data = metrics.get(model_name, {})
    if isinstance(metric_data, dict):
        return metric_data.get(metric_key, 0)
    return metric_data

# Comprehensive Ford Car Images mapping
ford_images = {
    'Fiesta': 'https://images.unsplash.com/photo-1590362891991-f776e747a588?auto=format&fit=crop&w=800&q=80',
    'Focus': 'https://images.unsplash.com/photo-1552062635-420de4d57df7?auto=format&fit=crop&w=800&q=80',
    'Mustang': 'https://images.unsplash.com/photo-1584345604476-8ac5e3cd31cc?auto=format&fit=crop&w=800&q=80',
    'Kuga': 'https://images.unsplash.com/photo-1609521263047-f8f205293f24?auto=format&fit=crop&w=800&q=80',
    'Puma': 'https://images.unsplash.com/photo-1619682817481-e994891cd1f5?auto=format&fit=crop&w=800&q=80',
    'EcoSport': 'https://images.unsplash.com/photo-1606152421802-db97b9c7a11b?auto=format&fit=crop&w=800&q=80',
    'Mondeo': 'https://images.unsplash.com/photo-1580274455091-1cb117181c03?auto=format&fit=crop&w=800&q=80',
    'C-MAX': 'https://images.unsplash.com/photo-1603584173870-7f23fdae1b7a?auto=format&fit=crop&w=800&q=80',
    'S-MAX': 'https://images.unsplash.com/photo-1511919884226-fd3cad34687c?auto=format&fit=crop&w=800&q=80',
    'Edge': 'https://images.unsplash.com/photo-1551830116-d9818cf5e137?auto=format&fit=crop&w=800&q=80',
    'KA+': 'https://images.unsplash.com/photo-1616423640778-28d1b53229bd?auto=format&fit=crop&w=800&q=80',
    'Galaxy': 'https://images.unsplash.com/photo-1549317661-bd32c8ce0db2?auto=format&fit=crop&w=800&q=80',
    'B-MAX': 'https://images.unsplash.com/photo-1533473359331-0135ef1b58bf?auto=format&fit=crop&w=800&q=80'
}
default_image = 'https://images.unsplash.com/photo-1549317661-bd32c8ce0db2?auto=format&fit=crop&w=800&q=80'

if df is None:
    st.stop()

# ==========================================
# 3. MAIN APPLICATION UI
# ==========================================
st.title("🚗 Smart Ford Car Price Predictor")
st.markdown("Enter the specifications of the vehicle below to get an intelligent ensemble price prediction.")

# ==========================================
# 4. SIDEBAR - DYNAMIC USER INPUTS
# ==========================================
user_inputs = {}

with st.sidebar:
    st.header("Vehicle Specifications")
    
    st.subheader("Categorical Features")
    for col in cat_cols:
        options = sorted(df[col].dropna().unique())
        display_name = col.replace('_', ' ').title()
        user_inputs[col] = st.selectbox(display_name, options=options)
        
    st.markdown("---")
    
    st.subheader("Numerical Features")
    for col in ['year', 'mileage', 'tax', 'mpg', 'engineSize']:
        if col not in df.columns: continue
        
        display_name = col.replace('Size', ' Size').title()
        min_val = float(df[col].min())
        max_val = float(df[col].max())
        mean_val = float(df[col].median())
        
        # Enforce integers for everything, including engineSize as requested
        if col in ['year', 'mileage', 'tax', 'engineSize']:
            user_inputs[col] = st.slider(
                display_name, min_value=int(min_val), max_value=int(max_val), value=int(mean_val), step=1
            )
        else:
            # Only MPG remains a float
            user_inputs[col] = st.slider(
                display_name, min_value=min_val, max_value=max_val, value=mean_val, step=1.0
            )
            
    predict_btn = st.button("Predict Price 🚀", use_container_width=True, type="primary")

# ==========================================
# 5. PREDICTION LOGIC & SESSION STATE
# ==========================================
if predict_btn:
    with st.spinner('Applying preprocessing pipelines and running models...'):
        time.sleep(1) 
        
        # Preprocessing
        input_df = pd.DataFrame([user_inputs])
        encoded_cats = encoder.transform(input_df[cat_cols])
        encoded_df = pd.DataFrame(encoded_cats, columns=encoder.get_feature_names_out(cat_cols), index=input_df.index)
        
        numeric_cols_order = ['year', 'mileage', 'mpg', 'engineSize', 'tax'] 
        final_df = pd.concat([input_df[numeric_cols_order], encoded_df], axis=1)
        final_df[num_cols] = scaler.transform(final_df[num_cols])
        final_df = final_df[feature_columns]

        # Model Inference
        predictions = {}
        for name, model in models.items():
            pred = model.predict(final_df)[0]
            predictions[name] = pred
            
        # Evaluation & Blended Ensemble Price
        best_model_name = max(predictions.keys(), key=lambda k: get_metric(k, "R2"))
        ensemble_price = np.mean(list(predictions.values()))
        
        st.session_state['has_predicted'] = True
        st.session_state['best_model'] = best_model_name
        st.session_state['ensemble_price'] = ensemble_price
        st.session_state['best_r2'] = get_metric(best_model_name, "R2")
        
        st.session_state['results_df'] = pd.DataFrame({
            'Algorithm': list(predictions.keys()),
            'Predicted Price (₹)': [round(v, 2) for v in predictions.values()],
            'Historical R²': [get_metric(m, "R2") for m in predictions.keys()],
            'Historical MAE': [get_metric(m, "MAE") for m in predictions.keys()],
            'Historical MSE': [get_metric(m, "MSE") for m in predictions.keys()]
        })

# ==========================================
# 6. MAIN CONTENT AREA (IMAGE & RESULTS)
# ==========================================
col1, col2 = st.columns([1, 2])

with col1:
    selected_model_name = user_inputs.get('model', '').strip()
    img_url = ford_images.get(selected_model_name, default_image)
    st.image(img_url, caption=f"Selected Model: {selected_model_name if selected_model_name else 'Ford Car'}", use_container_width=True, clamp=True)

with col2:
    if st.session_state.get('has_predicted'):
        best_model_name = st.session_state['best_model']
        ensemble_price = st.session_state['ensemble_price']
        best_r2_score = st.session_state['best_r2']
        results_df = st.session_state['results_df']
        
        st.success(f"### Estimated Price: ₹ {ensemble_price:,.2f}")
        st.info(f"🧠 **How was this calculated?** This is an **Ensemble Average**, meaning we took the predictions from all {len(models)} algorithms and averaged them to give you a balanced, unbiased price estimate.")
        
        st.markdown("### Model Comparison")
        
        def highlight_best(row):
            if row['Algorithm'] == best_model_name:
                return ['background-color: #d4edda; color: #155724'] * len(row)
            return [''] * len(row)

        st.dataframe(
            results_df.style.apply(highlight_best, axis=1).format({
                'Predicted Price (₹)': "₹ {:,.2f}", 
                'Historical R²': "{:.4f}",
                'Historical MAE': "{:,.2f}",
                'Historical MSE': "{:,.2f}"
            }),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("👈 Please enter vehicle specifications and click **Predict Price 🚀** to see results.")

# ==========================================
# 7. INTERACTIVE VISUALIZATIONS
# ==========================================
if st.session_state.get('has_predicted'):
    st.markdown("---")
    
    # Graph Controls Layout
    graph_col1, graph_col2 = st.columns([3, 1])
    
    with graph_col1:
        st.markdown("### Prediction Variance Across Algorithms")
    with graph_col2:
        chart_type = st.selectbox(
            "Select Chart Type:", 
            ["Bar Chart", "Pie Chart", "Donut Chart", "Line Chart"],
            index=0
        )
    
    results_df = st.session_state['results_df']
    best_model_name = st.session_state['best_model']
    
    # Color Palette for Charts
    custom_colors = px.colors.qualitative.Vivid
    
    # Generate requested chart
    if chart_type == "Bar Chart":
        fig = px.bar(
            results_df, x='Algorithm', y='Predicted Price (₹)', 
            color='Algorithm', text_auto='.2s',
            color_discrete_sequence=custom_colors
        )
        fig.update_traces(textfont_size=12, textangle=0, textposition="outside", cliponaxis=False)
        
    elif chart_type == "Pie Chart":
        fig = px.pie(
            results_df, names='Algorithm', values='Predicted Price (₹)',
            color='Algorithm', color_discrete_sequence=custom_colors
        )
        fig.update_traces(textposition='inside', textinfo='percent+label', marker=dict(line=dict(color='#000000', width=1)))
        
    elif chart_type == "Donut Chart":
        fig = px.pie(
            results_df, names='Algorithm', values='Predicted Price (₹)',
            color='Algorithm', hole=0.45, color_discrete_sequence=custom_colors
        )
        fig.update_traces(textposition='inside', textinfo='percent+label', marker=dict(line=dict(color='#000000', width=1)))
        
    elif chart_type == "Line Chart":
        fig = px.line(
            results_df, x='Algorithm', y='Predicted Price (₹)', markers=True,
            color_discrete_sequence=['#ff0055']
        )
        fig.update_traces(line=dict(width=4), marker=dict(size=12, line=dict(width=2, color='white')))
        
    st.plotly_chart(fig, use_container_width=True)
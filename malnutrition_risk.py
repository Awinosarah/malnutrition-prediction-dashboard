import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import sys
import warnings
import time
import pickle

warnings.filterwarnings('ignore')

try:
    import geopandas as gpd
    geopandas_available = True
except ImportError:
    geopandas_available = False

st.set_page_config(layout="wide", page_title="Malnutrition Prediction")

file_path = "r/Users/sarahawino/Downloads/UpdatednewfromJ33.csv"
TARGET_VARIABLE = '108-NA01b1_2019. No. of new SAM admissions in ITC'
IDENTIFIER_COLUMNS = [
    'periodid', 'periodname', 'district', 'region',
    'periodid.1', 'periodname.1', 'periodcode.1', 'organisationunitname',

]

# List of columns to explicitly exclude from feature selection/modeling
EXCLUDED_FEATURES = [
    'orgunitlevel2', 
    '108-NA01b1_2019. No. of new SAM admissions in ITC.1',
    'extracted_month',
    'extracted_year'
]

# This list is now primarily for reference and correlation plotting in Tab 2.
TOP_FEATURES = [
    '108-CD06b. Pneumonia - Deaths',
    '108-CD06a. Pneumonia (Cases)',
    '108-EP01a1. Malaria Total - Cases',
    'CCH - Normalized difference vegetation index (MODIS_NDVI)',
    'CCH - Enhanced vegetation index (MODIS_EVI)',
    'CCH - Relative humidity (ERA5-Land)',
    'CCH - Heat stress (ERA5-HEAT)',
    'CCH - Air temperature (ERA5-Land)',
    'CCH - Precipitation (CHIRPS)',

    f'{TARGET_VARIABLE}_lag_3'
]

# --- Risk Level Assignment Function ---
def assign_risk_level(value, q25, q50, q75):
    if pd.isna(value):
        return 'No Data'
    elif value <= q25:
        return 'Low'
    elif value <= q50:
        return 'Moderate'
    elif value <= q75:
        return 'High'
    else:
        return 'Severe'

# --- Backend Functions for Data Types and Preprocessing ---
def convert_to_numeric(df, column):
    df[column] = pd.to_numeric(df[column], errors='coerce')
    return df

def add_lag_features(df, target_col, group_col, time_col, lags=[1, 2, 3]):
    df_with_lags = df.copy()
    if time_col in df_with_lags.columns and not df_with_lags[time_col].isna().all():
        df_with_lags = df_with_lags.sort_values(by=[group_col, time_col])
        for lag in lags:
            lag_col_name = f'{target_col}_lag_{lag}'
            df_with_lags[lag_col_name] = df_with_lags.groupby(group_col)[target_col].shift(lag)
    return df_with_lags

def handle_missing_values(df):
    """
    Use KNN Imputer for numeric columns and mode for categorical.
    """
    processed_df = df.copy()
    
    # Separate numeric and categorical columns
    numeric_cols = processed_df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = processed_df.select_dtypes(exclude=np.number).columns.tolist()
    
    # Impute numeric columns using KNN
    if numeric_cols:
        # Filter out columns with all NaN values
        valid_numeric_cols = [col for col in numeric_cols if processed_df[col].notna().any()]
        
        if valid_numeric_cols:
            imputer = KNNImputer(n_neighbors=5)
            imputed_data = imputer.fit_transform(processed_df[valid_numeric_cols])
            
            # Assign back using DataFrame to ensure column alignment
            processed_df[valid_numeric_cols] = pd.DataFrame(
                imputed_data, 
                columns=valid_numeric_cols, 
                index=processed_df.index
            )
        
    # Impute categorical columns using mode
    for col in categorical_cols:
        if processed_df[col].isnull().any():
            mode_val = processed_df[col].mode()[0] if not processed_df[col].mode().empty else 'Unknown'
            processed_df[col] = processed_df[col].fillna(mode_val)
            
    return processed_df

def preprocess_data(df):
    processed_df = df.copy()
    
    district_col = None
    if 'district' in df.columns:
        district_col = 'district'
    elif 'organisationunitname' in df.columns:
        district_col = 'organisationunitname'
        processed_df['district'] = processed_df['organisationunitname']
    elif 'ou' in df.columns:
        district_col = 'ou'
        processed_df['district'] = processed_df['ou']
    else:
        processed_df['district'] = 'Unknown District'
        district_col = 'district'
    
    # Ensure TARGET_VARIABLE is numeric before lag creation
    processed_df = convert_to_numeric(processed_df, TARGET_VARIABLE)
    
    cols_to_process_numeric = [col for col in processed_df.columns
                               if col not in IDENTIFIER_COLUMNS
                               and col != TARGET_VARIABLE
                               and col != 'periodname']
    for col in cols_to_process_numeric:
        processed_df = convert_to_numeric(processed_df, col)
    
    if 'periodname' in processed_df.columns:
        try:
            processed_df['period_datetime'] = pd.to_datetime(
                processed_df['periodname'], format='%b-%y', errors='coerce'
            )
            processed_df['extracted_month'] = processed_df['period_datetime'].dt.month
            processed_df['extracted_year'] = processed_df['period_datetime'].dt.year
        except Exception:
            processed_df['extracted_month'] = np.nan
            processed_df['extracted_year'] = np.nan
            processed_df['period_datetime'] = pd.NaT
    else:
        processed_df['extracted_month'] = np.nan
        processed_df['extracted_year'] = np.nan
        processed_df['period_datetime'] = pd.NaT
    
    if 'period_datetime' in processed_df.columns and not processed_df['period_datetime'].isna().all() and district_col:
        processed_df = add_lag_features(processed_df, TARGET_VARIABLE, district_col, 'period_datetime')
    
    processed_df = handle_missing_values(processed_df)
    
    return processed_df, district_col


# --- FUNCTION FOR MODEL SAVING ---
def save_model_components(model, scaler, features, base_name="malnutrition_predictor"):
    """Saves the trained model, scaler, and feature list using pickle."""
    try:
        # 1. Save the Model
        model_path = f'{base_name}_model.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
            
        # 2. Save the Scaler (if it exists)
        scaler_path = f'{base_name}_scaler.pkl'
        if scaler is not None:
            with open(scaler_path, 'wb') as f:
                pickle.dump(scaler, f)
        else:
            scaler_path = 'None (Scaling Skipped)'

        # 3. Save the Feature List
        features_path = f'{base_name}_features.pkl'
        with open(features_path, 'wb') as f:
            pickle.dump(features, f)

        return model_path, scaler_path, features_path
        
    except Exception as e:
        return f"Error: {str(e)}", None, None
# -------------------------------------


# --- Main Streamlit App ---
def main():
    st.title("Malnutrition Prediction Dashboard")
    st.markdown("""
    This application predicts malnutrition risk levels by district using machine learning models.
    **Feature selection is automatic** using all eligible input variables.
    """)

    tab1, tab2, tab3, tab4 = st.tabs([
        "Data Loading",
        "Exploratory Data Analysis",
        "Model Training",
        "District Risk Mapping"
    ])

    # ----- Tab 1: Data Loading -----
    with tab1:
        st.header("Data Loading")

        st.subheader("Data Source")
        data_source = st.radio("Choose data source:", ["Upload CSV file", "Use file path"])
        
        df = None
        
        if data_source == "Upload CSV file":
            uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)
                    st.success("✅ Successfully loaded data from uploaded file")
                except Exception as e:
                    st.error(f"❌ Error reading uploaded file: {str(e)}")
        else:
            st.info(f"Using file path: `{file_path}`")
            try:
                df = pd.read_csv(file_path)
                st.success(f"Successfully loaded data from `{file_path}`")
            except FileNotFoundError:
                st.error(f"❌ Error: CSV file not found at `{file_path}`. Please check the path or upload a file instead.")
            except Exception as e:
                st.error(f"❌ Error loading the CSV file: {str(e)}")

        if df is not None:
            if TARGET_VARIABLE not in df.columns:
                st.error(f"❌ Target variable '{TARGET_VARIABLE}' not found in the dataset.")
                st.write("Available columns:", df.columns.tolist())
                st.stop()

            st.subheader("Data Overview")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Rows", df.shape[0])
            with col2:
                st.metric("Total Columns", df.shape[1])
            with col3:
                st.metric("Missing Values", df.isna().sum().sum())
            
            st.dataframe(df.head(10), use_container_width=True)

            with st.spinner("Processing data..."):
                initial_row_count = len(df)
                processed_df, district_col = preprocess_data(df)
                final_row_count = len(processed_df)
            
            st.success("Data preprocessed successfully")
            
            st.subheader("Preprocessing Summary")
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Time Features Created:**")
                st.write("- Month extracted from period name")
                st.write("- Year extracted from period name")
                st.write("- Lag features (1-3 periods only)")
            with col2:
                st.write("**Data Quality:**")
                st.write(f"- Initial rows: {initial_row_count}")
                st.write(f"- Rows after initial imputation: {final_row_count}")
                st.write(f"- District column: `{district_col}`")
                st.write(f"- Numeric features: {len(processed_df.select_dtypes(include=[np.number]).columns)}")

            st.session_state['processed_df'] = processed_df
            st.session_state['target_variable'] = TARGET_VARIABLE
            st.session_state['district_col'] = district_col
        else:
            st.warning("Please upload a CSV file or ensure the file path is correct.")
    
    # ----- Tab 2: Exploratory Data Analysis -----
    with tab2:
        st.header("Exploratory Data Analysis")
        
        if 'processed_df' in st.session_state and 'target_variable' in st.session_state:
            processed_df = st.session_state['processed_df']
            target_variable = st.session_state['target_variable']
            
            all_cols = processed_df.columns.tolist()
            
            # Identify all eligible features
            potential_features = [
                col for col in all_cols 
                if pd.api.types.is_numeric_dtype(processed_df[col])
                and col != target_variable
                and col not in IDENTIFIER_COLUMNS
                and col not in EXCLUDED_FEATURES
            ]
            
            # Ensure lag features are included in the potential list
            for lag in range(1, 4):
                lag_col_name = f'{TARGET_VARIABLE}_lag_{lag}'
                if lag_col_name in processed_df.columns and lag_col_name not in potential_features:
                    potential_features.append(lag_col_name)

            if not potential_features:
                st.error("❌ No numeric features available for analysis after preprocessing.")
            else:
                st.subheader("1. Target Variable Distribution")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Mean", f"{processed_df[target_variable].mean():.2f}")
                with col2:
                    st.metric("Median", f"{processed_df[target_variable].median():.2f}")
                with col3:
                    st.metric("Std Dev", f"{processed_df[target_variable].std():.2f}")
                with col4:
                    st.metric("Max", f"{processed_df[target_variable].max():.2f}")
                
                fig = px.histogram(
                    processed_df,
                    x=target_variable,
                    nbins=30,
                    title=f'Distribution of {target_variable}',
                    labels={target_variable: target_variable, 'count': 'Frequency'}
                )
                fig.update_layout(bargap=0.1, height=400)
                st.plotly_chart(fig, use_container_width=True)
                
                st.subheader("2. Trend Analysis Over Time")
                
                if 'extracted_year' in processed_df.columns and 'extracted_month' in processed_df.columns:
                    processed_df_temp = processed_df.copy()
                    processed_df_temp['year_month'] = (
                        processed_df_temp['extracted_year'].astype(str) + '-' +
                        processed_df_temp['extracted_month'].astype(str).str.zfill(2)
                    )
                    
                    time_trend = processed_df_temp.groupby('year_month')[target_variable].agg(['mean', 'sum']).reset_index()
                    time_trend = time_trend.sort_values('year_month')
                    
                    fig = px.line(
                        time_trend,
                        x='year_month',
                        y='mean',
                        title='Target Variable Trend Over Time (Monthly Average)',
                        labels={'mean': f'Average {target_variable}', 'year_month': 'Year-Month'}
                    )
                    fig.update_xaxes(tickangle=45)
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No time columns available for trend analysis.")
                
                st.subheader("3. Top Correlated Features Analysis")
                st.markdown("""
                This visualization uses the top features for correlation, excluding the manual selection option.
                """)
                
                features_to_check = potential_features.copy()
                # Ensure the features used for the correlation plot are available and limited (e.g., top 10)
                available_top_features = [f for f in TOP_FEATURES if f in features_to_check]
                features_for_correlation = available_top_features[:min(10, len(available_top_features))]
                
                if features_for_correlation:
                    correlation_cols = features_for_correlation + [target_variable]
                    correlation_matrix = processed_df[correlation_cols].corr(method='spearman')
                    
                    fig, ax = plt.subplots(figsize=(14, 12))
                    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
                    
                    sns.heatmap(
                        correlation_matrix,
                        mask=mask,
                        annot=True,
                        cmap='coolwarm',
                        center=0,
                        square=True,
                        fmt='.3f',
                        cbar_kws={"shrink": .8},
                        ax=ax
                    )
                    ax.set_title(f'Spearman Correlation Matrix (Top Features)', fontsize=16, pad=20)
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    st.write("**Correlations with Target Variable:**")
                    target_correlations = correlation_matrix[target_variable].drop(target_variable).sort_values(key=abs, ascending=False)
                    
                    fig, ax = plt.subplots(figsize=(10, max(6, len(target_correlations) * 0.4)))
                    colors = ['#e74c3c' if x < 0 else '#3498db' for x in target_correlations.values]
                    bars = ax.barh(range(len(target_correlations)), target_correlations.values, color=colors)
                    ax.set_yticks(range(len(target_correlations)))
                    ax.set_yticklabels(target_correlations.index)
                    ax.set_xlabel('Correlation Coefficient', fontsize=12)
                    ax.set_title(f'Feature Correlations with {target_variable}', fontsize=14)
                    ax.grid(axis='x', alpha=0.3)
                    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
                    
                    for i, (bar, value) in enumerate(zip(bars, target_correlations.values)):
                        ax.text(
                            value + (0.01 if value >= 0 else -0.01), i, f'{value:.3f}',
                            va='center', ha='left' if value >= 0 else 'right'
                        )
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    correlation_df = pd.DataFrame({
                        'Feature': target_correlations.index,
                        'Correlation': target_correlations.values.round(4)
                    }).reset_index(drop=True)
                    
                    st.dataframe(correlation_df, use_container_width=True)
                else:
                    st.warning("No eligible features available for correlation analysis.")
        else:
            st.warning("Please load and preprocess your data in the 'Data Loading' tab first.")
    
    # ----- Tab 3: Model Training -----
    with tab3:
        st.header("Model Training")

        if 'processed_df' in st.session_state and 'target_variable' in st.session_state:
            processed_df = st.session_state['processed_df']
            target_variable = st.session_state['target_variable']
            
            all_cols = processed_df.columns.tolist()
            
            # --- AUTOMATIC FEATURE SELECTION ---
            # Identify all eligible features (Must be numeric, not identifier, NOT in EXCLUDED_FEATURES)
            selected_features = [
                col for col in all_cols 
                if pd.api.types.is_numeric_dtype(processed_df[col])
                and col != target_variable
                and col not in IDENTIFIER_COLUMNS
                and col not in EXCLUDED_FEATURES
            ]
            
            # Ensure all lag features are included
            for lag in range(1, 4):
                lag_col_name = f'{TARGET_VARIABLE}_lag_{lag}'
                if lag_col_name in processed_df.columns and lag_col_name not in selected_features:
                    selected_features.append(lag_col_name)

            if not selected_features:
                st.error("❌ No eligible features available for modeling after preprocessing and filtering.")
            else:
                st.subheader("Feature Selection (Automatic)")
                st.info(f"✅ Automatically using **{len(selected_features)}** features for training (all eligible, non-excluded features).")
                st.code('\n'.join(selected_features), language='text')

                st.subheader("Model Configuration")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    test_size = st.slider("Test set size (%):", 10, 50, 20) / 100
                    random_state = st.number_input("Random seed for reproducibility:", 0, 999, 42)
                
                with col2:
                    perform_scaling = st.checkbox("Standardize features", value=True)
                    st.info("Using Linear Regression model")
                
                X = processed_df[selected_features].copy()
                y = processed_df[target_variable].copy()

                # ----------------------------------------------------
                # 🚀 FIX: Explicitly Impute Remaining NaNs in selected features/target
                # ----------------------------------------------------
                
                data_for_imputation = pd.concat([X, y], axis=1)
                
                if data_for_imputation.isnull().any().any():
                    
                    imputer = KNNImputer(n_neighbors=5)
                    initial_rows = len(data_for_imputation)
                    
                    imputed_data = imputer.fit_transform(data_for_imputation)
                    
                    data_imputed = pd.DataFrame(
                        imputed_data, 
                        columns=data_for_imputation.columns, 
                        index=data_for_imputation.index
                    )
                    
                    # Filter out rows that are completely empty (safety check)
                    data_imputed.dropna(how='all', inplace=True)
                    
                    # Re-assign X and y from the newly imputed data
                    X = data_imputed[selected_features]
                    y = data_imputed[target_variable]
                    
                    if len(X) < initial_rows:
                        st.warning(f"⚠️ Warning: {initial_rows - len(X)} rows were dropped because they were entirely NaN even after KNNImputation (likely severe data sparsity).")

                    st.info("✅ Re-applied KNN Imputer to ensure no NaNs remain in selected features and target.")
                
                if X.empty:
                    st.error("❌ Data is empty after imputation check. Cannot proceed with train/test split.")
                    st.stop()
                
                # ----------------------------------------------------
                # END FIX
                # ----------------------------------------------------

                if perform_scaling:
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)
                    X = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
                else:
                    scaler = None
                
                try:
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size, random_state=random_state
                    )
                    
                    st.success(f"✅ Data split: {len(X_train)} training samples, {len(X_test)} testing samples")
                    
                    st.subheader("Model Training and Evaluation")
                    
                    model = LinearRegression()
                    
                    start_time = time.time()
                    model.fit(X_train, y_train)
                    training_time = time.time() - start_time
                    
                    y_pred = model.predict(X_test)
                    
                    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                    mae = mean_absolute_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Linear Regression Results:**")
                        st.metric("RMSE", f"{rmse:.4f}")
                        st.metric("MAE", f"{mae:.4f}")
                        st.metric("R² Score", f"{r2:.4f}")
                        st.metric("Training Time", f"{training_time:.4f}s")
                    
                    with col2:
                        fig, ax = plt.subplots(figsize=(8, 8))
                        ax.scatter(y_test, y_pred, alpha=0.5, edgecolors='k', linewidth=0.5)
                        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
                        ax.set_xlabel("Actual Values", fontsize=12)
                        ax.set_ylabel("Predicted Values", fontsize=12)
                        ax.set_title("Linear Regression: Actual vs Predicted", fontsize=14)
                        ax.grid(True, alpha=0.3)
                        st.pyplot(fig)
                    
                    st.write("**Model Coefficients (Feature Importance):**")
                    coefficients_df = pd.DataFrame({
                        'Feature': X.columns,
                        'Coefficient': model.coef_
                    }).sort_values('Coefficient', key=abs, ascending=False)
                    
                    fig, ax = plt.subplots(figsize=(10, max(6, len(coefficients_df) * 0.3)))
                    colors = ['#e74c3c' if x < 0 else '#3498db' for x in coefficients_df['Coefficient'].values]
                    bars = ax.barh(coefficients_df['Feature'], coefficients_df['Coefficient'], color=colors)
                    ax.set_xlabel('Coefficient Value', fontsize=12)
                    ax.set_title("Linear Regression: Feature Coefficients", fontsize=14)
                    ax.grid(axis='x', alpha=0.3)
                    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
                    
                    for i, (feature, value) in enumerate(zip(coefficients_df['Feature'], coefficients_df['Coefficient'])):
                        ax.text(
                            value + (0.01 if value >= 0 else -0.01), i, f'{value:.3f}',
                            va='center', ha='left' if value >= 0 else 'right'
                        )
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    st.dataframe(coefficients_df, use_container_width=True)
                    
                    st.write("**Residuals Analysis:**")
                    residuals = y_test - y_pred
                    
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
                    
                    ax1.scatter(y_pred, residuals, alpha=0.5, edgecolors='k', linewidth=0.5)
                    ax1.axhline(y=0, color='r', linestyle='--', lw=2)
                    ax1.set_xlabel("Predicted Values", fontsize=12)
                    ax1.set_ylabel("Residuals", fontsize=12)
                    ax1.set_title("Residuals vs Predicted Values", fontsize=14)
                    ax1.grid(True, alpha=0.3)
                    
                    ax2.hist(residuals, bins=30, edgecolor='black', alpha=0.7)
                    ax2.set_xlabel("Residuals", fontsize=12)
                    ax2.set_ylabel("Frequency", fontsize=12)
                    ax2.set_title("Distribution of Residuals", fontsize=14)
                    ax2.grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    results = pd.DataFrame({
                        'Model': ['Linear Regression'],
                        'RMSE': [rmse],
                        'MAE': [mae],
                        'R²': [r2],
                        'Training Time (s)': [training_time],
                        'Features Used': [len(selected_features)]
                    })
                    
                    csv = results.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Model Results",
                        data=csv,
                        file_name="linear_regression_results.csv",
                        mime="text/csv"
                    )
                    
                    st.session_state['trained_models'] = {'Linear Regression': model}
                    st.session_state['X_features'] = selected_features
                    st.session_state['scaler'] = scaler
                    st.session_state['X_test_df'] = X_test
                    st.session_state['y_test_series'] = y_test

                    # --- MODEL PERSISTENCE CODE ---
                    st.subheader("Model Persistence")
                    
                    model_path, scaler_path, features_path = save_model_components(
                        model, scaler, selected_features
                    )
                    
                    if isinstance(model_path, str) and model_path.startswith('Error'):
                        st.error(f"❌ Failed to save model components: {model_path}")
                    else:
                        st.success(f"✅ Model components saved locally for deployment:")
                        st.markdown(f"- **Model (PKL):** `{model_path}`")
                        st.markdown(f"- **Scaler (PKL):** `{scaler_path}`")
                        st.markdown(f"- **Features (PKL):** `{features_path}`")
                    # ----------------------------

                except Exception as e:
                    st.error(f"❌ Error during model training: {str(e)}")
        else:
            st.warning("⚠️ Please load and preprocess your data in the 'Data Loading' tab first.")
    
    # ----- Tab 4: District Risk Mapping -----
    with tab4:
        st.header("District Risk Mapping & Time Series Prediction")
        
        if ('processed_df' in st.session_state and
            'target_variable' in st.session_state and
            'district_col' in st.session_state):
            
            processed_df = st.session_state['processed_df']
            target_variable = st.session_state['target_variable']
            district_col = st.session_state['district_col']
            
            st.subheader("1. Current Risk Level by District")
            
            if district_col in processed_df.columns and 'period_datetime' in processed_df.columns:
                latest_data_per_district = processed_df.loc[
                    processed_df.groupby(district_col)['period_datetime'].idxmax()
                ]
                district_risk_current = latest_data_per_district[
                    [district_col, target_variable, 'periodname']
                ].copy()
                district_risk_current.rename(
                    columns={district_col: 'District', target_variable: 'Current_Risk_Score'},
                    inplace=True
                )
                
                if not district_risk_current['Current_Risk_Score'].empty and district_risk_current['Current_Risk_Score'].notna().any():
                    risk_bins_current = [
                        district_risk_current['Current_Risk_Score'].min() - 1,
                        district_risk_current['Current_Risk_Score'].quantile(0.25),
                        district_risk_current['Current_Risk_Score'].quantile(0.50),
                        district_risk_current['Current_Risk_Score'].quantile(0.75),
                        district_risk_current['Current_Risk_Score'].max() + 1
                    ]
                    
                    labels = ['Low', 'Moderate', 'High', 'Severe']
                    
                    district_risk_current['Risk Level'] = pd.cut(
                        district_risk_current['Current_Risk_Score'],
                        bins=risk_bins_current,
                        labels=labels,
                        include_lowest=True,
                        duplicates='drop'
                    )
                else:
                    district_risk_current['Risk Level'] = pd.Series(dtype='object')
                
                all_risk_categories = ['Low', 'Moderate', 'High', 'Severe', 'No Data']
                district_risk_current['Risk Level'] = pd.Categorical(
                    district_risk_current['Risk Level'],
                    categories=all_risk_categories,
                    ordered=True
                )
                
                district_risk_current = district_risk_current.sort_values('Current_Risk_Score', ascending=False)
                
                st.write("**District Risk Levels (Latest Available Month):**")
                st.dataframe(district_risk_current, use_container_width=True)
                
                col1, col2, col3, col4 = st.columns(4)
                risk_counts = district_risk_current['Risk Level'].value_counts()
                with col1:
                    st.metric("🟢 Low Risk", risk_counts.get('Low', 0))
                with col2:
                    st.metric("🟡 Moderate Risk", risk_counts.get('Moderate', 0))
                with col3:
                    st.metric("🟠 High Risk", risk_counts.get('High', 0))
                with col4:
                    st.metric("🔴 Severe Risk", risk_counts.get('Severe', 0))
                
                color_map = {
                    'Low': '#2ecc71',
                    'Moderate': '#f9e79f',
                    'High': '#e67e22',
                    'Severe': '#c0392b',
                    'No Data': '#95a5a6'
                }
                
                fig = px.bar(
                    district_risk_current,
                    x='District',
                    y='Current_Risk_Score',
                    color='Risk Level',
                    title='Current Malnutrition Risk by District',
                    color_discrete_map=color_map,
                    category_orders={"Risk Level": all_risk_categories}
                )
                fig.update_xaxes(tickangle=45)
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)

                st.markdown("---")
                st.subheader("2. Predictive Risk Mapping")
                
                if ('trained_models' in st.session_state and
                    'X_features' in st.session_state):
                    
                    trained_models = st.session_state['trained_models']
                    selected_features = st.session_state['X_features']
                    scaler = st.session_state.get('scaler', None)
                    
                    selected_model_name = 'Linear Regression'
                    selected_model = trained_models[selected_model_name]
                        
                    X_pred_latest = latest_data_per_district[selected_features].copy()
                    X_pred_latest_cleaned = X_pred_latest.dropna(subset=selected_features)
                    
                    if X_pred_latest_cleaned.empty:
                        st.error("❌ No valid data points for prediction.")
                    else:
                        if scaler is not None:
                            X_pred_scaled = pd.DataFrame(
                                scaler.transform(X_pred_latest_cleaned),
                                columns=X_pred_latest_cleaned.columns,
                                index=X_pred_latest_cleaned.index
                            )
                        else:
                            X_pred_scaled = X_pred_latest_cleaned

                        try:
                            predictions = selected_model.predict(X_pred_scaled)
                            
                            district_predicted_latest = latest_data_per_district.loc[X_pred_scaled.index].copy()
                            district_predicted_latest['Predicted_Risk_Score'] = predictions
                            
                            predicted_values = district_predicted_latest['Predicted_Risk_Score']
                            
                            if not predicted_values.empty and predicted_values.notna().any():
                                risk_bins_pred = [
                                    predicted_values.min() - 1,
                                    predicted_values.quantile(0.25),
                                    predicted_values.quantile(0.50),
                                    predicted_values.quantile(0.75),
                                    predicted_values.max() + 1
                                ]
                                
                                labels = ['Low', 'Moderate', 'High', 'Severe']
                                
                                district_predicted_latest['Predicted_Risk_Category'] = pd.cut(
                                    predicted_values,
                                    bins=risk_bins_pred,
                                    labels=labels,
                                    include_lowest=True,
                                    duplicates='drop'
                                )
                            else:
                                district_predicted_latest['Predicted_Risk_Category'] = 'No Data'
                            
                            all_predicted_categories = ['Low', 'Moderate', 'High', 'Severe', 'No Data']
                            district_predicted_latest['Predicted_Risk_Category'] = pd.Categorical(
                                district_predicted_latest['Predicted_Risk_Category'],
                                categories=all_predicted_categories,
                                ordered=True
                            )
                            
                            district_predicted_latest = district_predicted_latest.sort_values('Predicted_Risk_Score', ascending=False)
                            
                            st.write(f"**District Predicted Risk Levels using {selected_model_name}:**")
                            if 'District' not in district_predicted_latest.columns:
                                district_predicted_latest.rename(columns={district_col: 'District'}, inplace=True)

                            display_cols = ['District', 'periodname', target_variable, 'Predicted_Risk_Score', 'Predicted_Risk_Category']
                            st.dataframe(district_predicted_latest[display_cols], use_container_width=True)
                            
                            col1, col2, col3, col4 = st.columns(4)
                            pred_risk_counts = district_predicted_latest['Predicted_Risk_Category'].value_counts()
                            with col1:
                                st.metric("🟢 Predicted Low", pred_risk_counts.get('Low', 0))
                            with col2:
                                st.metric("🟡 Predicted Moderate", pred_risk_counts.get('Moderate', 0))
                            with col3:
                                st.metric("🟠 Predicted High", pred_risk_counts.get('High', 0))
                            with col4:
                                st.metric("🔴 Predicted Severe", pred_risk_counts.get('Severe', 0))
                            
                            st.subheader("Predicted Malnutrition Risk Map")
                            if geopandas_available:
                                uploaded_geojson_pred = st.file_uploader(
                                    "Upload GeoJSON file for predicted risk map",
                                    type=["geojson", "json"],
                                    key="geojson_map_predicted"
                                )
                                if uploaded_geojson_pred:
                                    try:
                                        geojson_pred = json.load(uploaded_geojson_pred)
                                        gdf_pred = gpd.GeoDataFrame.from_features(geojson_pred['features'])

                                        geojson_district_property_pred = None
                                        if not gdf_pred.empty:
                                            geojson_cols_pred = gdf_pred.columns.tolist()
                                            data_district_column_name = 'District'
                                            possible_geojson_props_pred = ['name', 'districtname', 'orgunitname', data_district_column_name]
                                            for prop in possible_geojson_props_pred:
                                                if prop in geojson_cols_pred:
                                                    geojson_district_property_pred = prop
                                                    break
                                            if not geojson_district_property_pred:
                                                st.warning("⚠️ Could not auto-identify GeoJSON district property.")

                                        if geojson_district_property_pred:
                                            merged_gdf_pred = gdf_pred.merge(
                                                district_predicted_latest[['District', 'Predicted_Risk_Score', 'Predicted_Risk_Category', 'periodname']],
                                                left_on=geojson_district_property_pred,
                                                right_on='District',
                                                how='left'
                                            )
                                            merged_gdf_pred['Predicted_Risk_Score'] = merged_gdf_pred['Predicted_Risk_Score'].fillna(np.nan)
                                            merged_gdf_pred['Predicted_Risk_Category'] = merged_gdf_pred['Predicted_Risk_Category'].fillna('No Data')
                                            
                                            merged_gdf_pred['Predicted_Risk_Category'] = pd.Categorical(
                                                merged_gdf_pred['Predicted_Risk_Category'],
                                                categories=all_predicted_categories,
                                                ordered=True
                                            )
                                            
                                            color_map_predicted = {
                                                'Low': '#2ecc71',
                                                'Moderate': '#f9e79f',
                                                'High': '#e67e22',
                                                'Severe': '#c0392b',
                                                'No Data': '#95a5a6'
                                            }

                                            fig_pred_map = px.choropleth(
                                                merged_gdf_pred,
                                                geojson=merged_gdf_pred.geometry,
                                                locations=merged_gdf_pred.index,
                                                color='Predicted_Risk_Category',
                                                color_discrete_map=color_map_predicted,
                                                hover_name=geojson_district_property_pred,
                                                hover_data={
                                                    'Predicted_Risk_Category': True,
                                                    'Predicted_Risk_Score': ':.2f',
                                                    'periodname': True,
                                                    geojson_district_property_pred: False,
                                                    'District': False
                                                },
                                                labels={
                                                    'Predicted_Risk_Category': 'Predicted Risk Category',
                                                    'Predicted_Risk_Score': 'Predicted Risk Score'
                                                },
                                                category_orders={"Predicted_Risk_Category": all_predicted_categories}
                                            )
                                            fig_pred_map.update_geos(fitbounds="locations", visible=False)
                                            fig_pred_map.update_layout(
                                                height=800,
                                                title=f"Predicted Malnutrition Risk Map ({selected_model_name})",
                                                margin={"r": 10, "t": 60, "l": 10, "b": 10}
                                            )
                                            fig_pred_map.update_traces(
                                                marker_line_width=2,
                                                marker_line_color='white',
                                                hovertemplate='<b>%{hovertext}</b><br>Risk Level: %{customdata[0]}<br>Score: %{customdata[1]:.2f}<br>Month: %{customdata[2]}<extra></extra>'
                                            )
                                            st.plotly_chart(fig_pred_map, use_container_width=True, config={'displayModeBar': True})
                                        else:
                                            st.warning("⚠️ Could not create predicted GeoJSON map.")
                                    except Exception as e:
                                        st.error(f"❌ Error loading or processing GeoJSON: {str(e)}")
                                else:
                                    st.info("ℹ️ Upload a GeoJSON file to visualize predicted risk on district boundaries.")
                            else:
                                st.warning("⚠️ GeoJSON mapping requires 'geopandas' library.")
                        except Exception as e:
                            st.error(f"❌ Error making predictions: {str(e)}")
                else:
                    st.info("ℹ️ Please train models in the 'Model Training' tab first.")

            st.markdown("---")
            st.subheader("3. Time Series Prediction Plot")
            
            if 'trained_models' in st.session_state and 'X_features' in st.session_state:
                trained_models = st.session_state['trained_models']
                selected_features = st.session_state['X_features']
                scaler = st.session_state.get('scaler', None)
                X_test = st.session_state.get('X_test_df')
                y_test = st.session_state.get('y_test_series')

                all_districts = processed_df[district_col].unique().tolist()
                
                selected_district = st.selectbox(
                    "Select a District to plot time series for:",
                    options=all_districts
                )

                selected_model_name_plot = 'Linear Regression'
                selected_model = trained_models[selected_model_name_plot]

                if selected_district and selected_model:
                    district_data = processed_df[processed_df[district_col] == selected_district].copy()
                    
                    if 'period_datetime' not in district_data.columns or district_data['period_datetime'].isna().all():
                        st.warning(f"⚠️ No valid time data for {selected_district}.")
                    else:
                        district_data = district_data.sort_values('period_datetime')

                        # Data for imputation (selected features + target)
                        X_plot_and_target = district_data[selected_features + [target_variable]]
                        
                        # Initialize imputed data to the original data in case no NaNs are found
                        X_plot_and_target_imputed = X_plot_and_target.copy()
                        
                        if X_plot_and_target.isnull().any().any():
                            try:
                                imputer = KNNImputer(n_neighbors=5)
                                imputed_data = imputer.fit_transform(X_plot_and_target)
                                X_plot_and_target_imputed = pd.DataFrame(
                                    imputed_data, 
                                    columns=X_plot_and_target.columns, 
                                    index=X_plot_and_target.index
                                )
                                st.info("ℹ️ Imputed data for time series prediction plot.")
                            except Exception as e:
                                st.error(f"❌ Failed to impute data for plotting: {e}. Falling back to clean rows only.")
                                X_plot_and_target_imputed = X_plot_and_target.dropna()

                        
                        X_plot_for_pred = X_plot_and_target_imputed[selected_features]
                        
                        if X_plot_for_pred.empty:
                            st.warning(f"⚠️ No valid data points available in the selected time range for {selected_district} after cleaning.")
                        else:
                            if scaler is not None:
                                X_for_prediction = pd.DataFrame(
                                    scaler.transform(X_plot_for_pred),
                                    columns=X_plot_for_pred.columns,
                                    index=X_plot_for_pred.index
                                )
                            else:
                                X_for_prediction = X_plot_for_pred

                            # Update the actual district_data with the imputed target values for plotting purposes
                            district_data.loc[X_plot_for_pred.index, target_variable] = X_plot_and_target_imputed.loc[X_plot_for_pred.index, target_variable]
                            
                            district_data['Predicted_Value'] = np.nan
                            try:
                                district_data.loc[X_for_prediction.index, 'Predicted_Value'] = selected_model.predict(X_for_prediction)
                            except Exception as e:
                                st.error(f"❌ Error making predictions: {e}")
                                st.stop()

                            if X_test is not None and y_test is not None and len(y_test) > 1:
                                std_dev_of_errors = np.std(y_test - selected_model.predict(X_test))
                            else:
                                std_dev_of_errors = processed_df[target_variable].std() * 0.1

                            district_data['Predicted_Upper'] = district_data['Predicted_Value'] + std_dev_of_errors
                            district_data['Predicted_Lower'] = district_data['Predicted_Value'] - std_dev_of_errors
                            district_data['Predicted_Lower'] = district_data['Predicted_Lower'].apply(lambda x: max(0, x))
                            
                            fig = go.Figure()

                            fig.add_trace(go.Scatter(
                                x=district_data['period_datetime'],
                                y=district_data[target_variable],
                                mode='lines+markers',
                                name='Actual Admissions',
                                line=dict(color='#e67e22', width=2),
                                marker=dict(size=6)
                            ))

                            fig.add_trace(go.Scatter(
                                x=district_data['period_datetime'],
                                y=district_data['Predicted_Value'],
                                mode='lines+markers',
                                name='Predicted Admissions',
                                line=dict(color='#3498db', width=2),
                                marker=dict(size=6)
                            ))

                            fig.add_trace(go.Scatter(
                                x=district_data['period_datetime'].tolist() + district_data['period_datetime'].tolist()[::-1],
                                y=district_data['Predicted_Upper'].tolist() + district_data['Predicted_Lower'].tolist()[::-1],
                                fill='toself',
                                fillcolor='rgba(52, 152, 219, 0.2)',
                                line=dict(color='rgba(255,255,255,0)'),
                                name='Prediction Interval',
                                hoverinfo="skip"
                            ))
                            
                            fig.update_layout(
                                title=f'{target_variable}<br>Actual vs. Predicted Over Time for {selected_district} (Linear Regression)',
                                xaxis_title="Time Period",
                                yaxis_title=target_variable,
                                hovermode="x unified",
                                height=500,
                                showlegend=True,
                                legend=dict(
                                    orientation="h",
                                    yanchor="bottom",
                                    y=1.02,
                                    xanchor="right",
                                    x=1
                                )
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                actual_mean = district_data[target_variable].mean()
                                st.metric("Average Actual", f"{actual_mean:.2f}")
                            with col2:
                                predicted_mean = district_data['Predicted_Value'].mean()
                                st.metric("Average Predicted", f"{predicted_mean:.2f}")
                            with col3:
                                difference = predicted_mean - actual_mean
                                st.metric("Difference", f"{difference:.2f}", delta=f"{difference:.2f}")
            else:
                st.info("Please train models to enable time series visualization.")
        else:
            st.warning("Please load and preprocess your data in the 'Data Loading' tab first.")

if __name__ == "__main__":
    main()
import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score
import plotly.express as px
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
# Configure Streamlit page settings
st.set_page_config(
    page_title="F1 Pit Stop Prediction & Analysis",
    page_icon="🏎️",
    layout="wide"
)

st.title("🏎️ F1 Pit Stop Prediction & Statistical Analysis")
st.markdown("Predict driver points and explore detailed pit-stop correlations and EDA.")

@st.cache_data
# Load all required CSV datasets into pandas DataFrames
def load_data():
    path = "/content/drive/MyDrive/archive/"
    races = pd.read_csv(f"{path}races.csv", na_values='\\N')
    results = pd.read_csv(f"{path}results.csv", na_values='\\N')
    pit_stops = pd.read_csv(f"{path}pit_stops.csv", na_values='\\N')
    drivers = pd.read_csv(f"{path}drivers.csv", na_values='\\N')
    constructors = pd.read_csv(f"{path}constructors.csv", na_values='\\N')
    circuits = pd.read_csv(f"{path}circuits.csv", na_values='\\N')
    status = pd.read_csv(f"{path}status.csv", na_values='\\N')
    return races, results, pit_stops, drivers, constructors, circuits, status

# Perform data cleaning and feature engineering on merged race and pit stop datasets
def preprocess_data(races, results, pit_stops, drivers, constructors, circuits, status):
    pit_stops['duration'] = pd.to_numeric(pit_stops['duration'], errors='coerce')
    if 'milliseconds' in pit_stops.columns:
        pit_stops['milliseconds'] = pd.to_numeric(pit_stops['milliseconds'], errors='coerce')
    if 'milliseconds' in results.columns:
        results['milliseconds'] = pd.to_numeric(results['milliseconds'], errors='coerce')
    results[['points','grid','position']] = results[['points','grid','position']].apply(pd.to_numeric, errors='coerce')
    rc = pd.merge(races, circuits, on='circuitId', how='left')
    df = pit_stops.merge(results, on=['raceId','driverId'], how='left')
    df = df.merge(drivers, on='driverId', how='left')
    df = df.merge(constructors, on='constructorId', how='left')
    df = df.merge(rc, on='raceId', how='left')
    df = df.merge(status, on='statusId', how='left')
    df = df.rename(columns={'name':'constructor_name','name_x':'race_name','name_y':'circuit_name'})
    if 'milliseconds' in pit_stops.columns:
        stats = pit_stops.groupby(['raceId','driverId']).agg(
            total_stops=('milliseconds','count'),
            avg_stop_duration=('milliseconds', lambda x: x.mean()/1000),
            earliest_lap=('lap','min'),
            latest_lap=('lap','max')
        ).reset_index()
    else:
        stats = pit_stops.groupby(['raceId','driverId']).agg(
            total_stops=('duration','count'),
            avg_stop_duration=('duration', lambda x: x.mean()),
            earliest_lap=('lap','min'),
            latest_lap=('lap','max')
        ).reset_index()
    df = df.merge(stats, on=['raceId','driverId'], how='left')
    df['position_change'] = df['grid'] - df['position']
    df['early_stop_flag'] = df['earliest_lap'] < 5
    df['late_stop_flag'] = df['latest_lap'] > 45
    df['stop_range'] = df['latest_lap'] - df['earliest_lap']
    df['avg_speed_per_stop'] = df['avg_stop_duration'] / df['total_stops']
    df['has_multiple_stops'] = df['total_stops'] > 1
    df['pit_timing_ratio'] = df['earliest_lap'] / df['latest_lap']
    df['efficiency_score'] = df['points'] / (df['total_stops'] + 1)
    df['lap_momentum'] = df['stop_range'] / df['total_stops']
    df['change_per_stop'] = df['position_change'] / df['total_stops'].replace(0, np.nan)
    return df

# Train multiple regression models to predict race position and compare them
# Returns the best performing model based on R² score
def train_models(df, use_feature_elimination=True):
    importances_df = None
    feats = ['grid', 'total_stops', 'avg_stop_duration', 'earliest_lap', 'pit_timing_ratio']
    cat = ['circuitId','constructorId']
    # Drop rows with missing values in selected features or target
    data = df.copy().dropna(subset=feats + ['position'])
    X = data[feats+cat]; y = data['position']
    # Split the data into training and testing sets
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
    # Create a preprocessing pipeline: scale numeric features and one-hot encode categorical features
    pre = ColumnTransformer([('num',StandardScaler(),feats),('cat',OneHotEncoder(handle_unknown='ignore'),cat)])
    rf = Pipeline([('p', pre), ('m', RandomForestRegressor(n_estimators=100, random_state=42))])
    gb = Pipeline([('p', pre), ('m', GradientBoostingRegressor(n_estimators=100, random_state=42))])
    xgb = Pipeline([('p', pre), ('m', XGBRegressor(n_estimators=100, random_state=42, verbosity=0))])
    cb = Pipeline([('p', pre), ('m', CatBoostRegressor(iterations=100, random_seed=42, verbose=0))])
    gb = Pipeline([('p',pre),('m',GradientBoostingRegressor(n_estimators=100,random_state=42))])
    rf.fit(X_train, y_train)
    gb.fit(X_train, y_train)
    xgb.fit(X_train, y_train)
    cb.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    gb_pred = gb.predict(X_test)
    xgb_pred = xgb.predict(X_test)
    cb_pred = cb.predict(X_test)
    rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred)); rf_r2 = r2_score(y_test, rf_pred)
    gb_rmse = np.sqrt(mean_squared_error(y_test, gb_pred)); gb_r2 = r2_score(y_test, gb_pred)
    xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_pred)); xgb_r2 = r2_score(y_test, xgb_pred)
    cb_rmse = np.sqrt(mean_squared_error(y_test, cb_pred)); cb_r2 = r2_score(y_test, cb_pred)
    models_scores = {
        'Random Forest': (rf, rf_rmse, rf_r2),
        'Gradient Boosting': (gb, gb_rmse, gb_r2),
        'XGBoost': (xgb, xgb_rmse, xgb_r2),
        'CatBoost': (cb, cb_rmse, cb_r2)
    }
    best_model_name = max(models_scores, key=lambda k: models_scores[k][2])
    st.subheader("📊 Model Comparison (R² Score)")
    model_df = pd.DataFrame({
        'Model': list(models_scores.keys()),
        'R² Score': [models_scores[m][2] for m in models_scores],
        'RMSE': [models_scores[m][1] for m in models_scores]
    })
    fig_r2 = px.bar(model_df, x='Model', y='R² Score', title='Model Performance (R² Score)', text='R² Score')
    fig_r2.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    fig_r2.update_layout(yaxis=dict(range=[0, 1]))
    st.plotly_chart(fig_r2, use_container_width=True)

    fig_rmse = px.bar(model_df, x='Model', y='RMSE', title='Model Performance (RMSE)', text='RMSE')
    fig_rmse.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    st.plotly_chart(fig_rmse, use_container_width=True)
    
    # Retrieve best model and compute feature importances
    best_model, best_rmse, best_r2 = models_scores[best_model_name]

    feature_names = pre.transformers_[0][2] + list(pre.transformers_[1][1].get_feature_names_out(cat))
    #feature Elimination using SelectFromModel
    if use_feature_elimination:
        selector = SelectFromModel(best_model.named_steps['m'], threshold='median', prefit=True)
        X_selected = selector.transform(pre.transform(X))
        selected_features = np.array(feature_names)[selector.get_support()]
        st.write("✅ Selected Features (via SelectFromModel):", selected_features)
        st.write(f"🔢 Features Before Selection: {len(feature_names)}")
        st.write(f"🔻 Features After Selection: {len(selected_features)}")
        from sklearn.pipeline import make_pipeline
        final_model = make_pipeline(pre, selector, best_model.named_steps['m'])
        final_model.fit(X, y)
    else:
        final_model = best_model
        selector = None
    importances = best_model.named_steps['m'].feature_importances_
    if use_feature_elimination:
        feature_names = np.array(pre.transformers_[0][2] + list(pre.transformers_[1][1].get_feature_names_out(cat)))[selector.get_support()]
    else:
        feature_names = pre.transformers_[0][2] + list(pre.transformers_[1][1].get_feature_names_out(cat))
    importances_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values('Importance', ascending=False)
    # PCA analysis on the preprocessed feature space
    st.subheader("🧠 PCA - Principal Component Analysis")
    pca = PCA(n_components=2)
    X_transformed_for_pca = pre.transform(X)
    X_pca_input = X_transformed_for_pca
    X_pca = pca.fit_transform(X_pca_input)
    pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
    if len(y) != len(X_pca):
        y_trimmed = y.iloc[:len(X_pca)].reset_index(drop=True)
    else:
        y_trimmed = y.reset_index(drop=True)
    pca_df['Position'] = y_trimmed
    fig_pca = px.scatter(pca_df, x='PC1', y='PC2', color='Position', title='PCA Projection of Feature Space')
    st.plotly_chart(fig_pca, use_container_width=True)
    # Show PCA Loadings
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=['PC1', 'PC2'],
        index=feature_names
    )
    st.subheader("📌 PCA Loadings (Feature Contributions to PC1 & PC2)")
    st.dataframe(loadings)
    st.subheader("🔍 Feature Importances")
    st.dataframe(importances_df)
    

    return final_model, best_model_name, best_rmse, best_r2, selector

# Perform statistical analysis on correlation of features to final position and display results
def statistical_analysis(df):
    st.subheader("📉 Correlation Matrix")
    corr_df = df[['total_stops','avg_stop_duration','earliest_lap','latest_lap','stop_range','position_change','points','pit_timing_ratio','efficiency_score','lap_momentum','change_per_stop']].dropna()
    corr_matrix = corr_df.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', ax=ax)
    ax.set_title("Correlation Matrix of Engineered Features")
    st.pyplot(fig)
    st.subheader("📈 Statistical Correlation Analysis")
    pit = df.copy()
    pit['final_position'] = pd.to_numeric(pit['position'], errors='coerce')
    pit = pit.dropna(subset=['total_stops','avg_stop_duration','final_position','earliest_lap'])
    corr, p = pearsonr(pit['total_stops'], pit['final_position'])
    st.write(f"**Total Pit Stops vs Final Position:** r = {corr:.2f}, p = {p:.4f}")
    corr2, p2 = pearsonr(pit['earliest_lap'], pit['final_position'])
    st.write(f"**Earliest Pit Lap vs Final Position:** r = {corr2:.2f}, p = {p2:.4f}")

# Generate all EDA plots including pit stop timings, efficiency, and race outcomes
# Uses optional filters from user selection
def create_eda(df, races=None, pit_stops=None, circuits=None):
    st.subheader("📺 Animated Pit Stop Times by Circuit and Year")
    if races is not None and pit_stops is not None and circuits is not None:
        merged = (
            pit_stops
            .merge(races[['raceId', 'year', 'circuitId']], on='raceId')
            .merge(circuits[['circuitId', 'name']], on='circuitId')
            .rename(columns={'name': 'circuit_name'})
        )
        merged['duration_sec'] = pd.to_numeric(merged['duration'], errors='coerce')
        merged = merged.dropna(subset=['duration_sec'])
        avg_circuit_year = (
            merged[merged['year'] >= 2011]
            .groupby(['year', 'circuit_name'])['duration_sec']
            .mean()
            .reset_index(name='avg_stop_seconds')
        )
        fig = px.bar(
            avg_circuit_year,
            x='circuit_name',
            y='avg_stop_seconds',
            animation_frame='year',
            title='Average Pit Stop Time per Circuit by Year (2011+)',
            labels={'circuit_name': 'Circuit', 'avg_stop_seconds': 'Avg Pit Stop (s)'},
            text=avg_circuit_year['avg_stop_seconds'].round(2)
        )
        fig.update_layout(xaxis={'categoryorder': 'total descending'})
        fig.update_yaxes(range=[0, avg_circuit_year['avg_stop_seconds'].max() * 1.1])
        st.plotly_chart(fig, use_container_width=True)

        # Box plot by circuit
        st.subheader("📦 Pit Stop Duration by Circuit (2011+)")
        fig_box = px.box(merged[merged['year'] >= 2011], x='circuit_name', y='duration_sec', points='outliers', title='Pit Stop Duration by Circuit (2011+)')
        fig_box.update_layout(xaxis={'categoryorder': 'total descending'})
        st.plotly_chart(fig_box, use_container_width=True)

        # Line chart by year
        st.subheader("📈 Yearly Average Pit Stop Duration (2011+)")
        yearly_avg = (
            merged[merged['year'] >= 2011]
            .groupby('year')['duration_sec']
            .mean()
            .reset_index(name='avg_stop_seconds')
        )
        fig_line = px.line(yearly_avg, x='year', y='avg_stop_seconds', markers=True, title='Yearly Average Pit Stop Duration (2011+)')
        st.plotly_chart(fig_line, use_container_width=True)
    st.subheader("🔍 Filtered Position vs Pit Stop Count")
    selected_circuit = st.sidebar.selectbox("Filter EDA by Circuit", df['circuit_name'].dropna().unique(), index=0, key='eda_circuit')
    selected_team = st.sidebar.selectbox("Filter EDA by Team", df['constructor_name'].dropna().unique(), index=0, key='eda_team')
    selected_year = st.sidebar.selectbox("Filter EDA by Year", sorted(df['year'].dropna().unique()), index=0, key='eda_year')
    show_all = st.sidebar.checkbox("Show All Data (Ignore Filters)", value=False)
    filtered_eda = df[(df['circuit_name'] == selected_circuit) & (df['constructor_name'] == selected_team) & (df['year'] == selected_year)]
    eda_pit_plot = filtered_eda[['total_stops', 'position']].dropna()
    if not eda_pit_plot.empty:
        fig = px.scatter(eda_pit_plot, x='total_stops', y='position', trendline='ols', title=f'{selected_team} at {selected_circuit} ({selected_year}): Pit Stops vs Position')
        st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("📋 Sample Table of Engineered Features")
    feature_cols = ['surname','constructor_name','circuit_name','total_stops','avg_stop_duration','earliest_lap','latest_lap','stop_range','pit_timing_ratio','efficiency_score','lap_momentum','change_per_stop','points']
    eda_source = df if show_all else df[(df['circuit_name'] == selected_circuit) & (df['constructor_name'] == selected_team) & (df['year'] == selected_year)]
    sample_table = eda_source[feature_cols].dropna().head(15)
    st.dataframe(sample_table)
    st.subheader("📊 Exploratory Data Analysis")
    eda_source = df if show_all else df[(df['circuit_name'] == selected_circuit) & (df['constructor_name'] == selected_team) & (df['year'] == selected_year)]
    d = eda_source.dropna(subset=['earliest_lap','latest_lap','avg_stop_duration','position_change','total_stops','surname','constructor_name','circuit_name'])
    st.subheader("🕓 Pit Stop Timing Impact")
    fig_timing = px.scatter(d, x='earliest_lap', y='position', trendline='ols', title='Earliest Pit Stop Lap vs Final Position')
    st.plotly_chart(fig_timing, use_container_width=True)
    
    eda_source = df if show_all else df[(df['circuit_name'] == selected_circuit) & (df['constructor_name'] == selected_team) & (df['year'] == selected_year)]
    d = eda_source.dropna(subset=['earliest_lap','latest_lap','avg_stop_duration','position_change','total_stops','surname','constructor_name','circuit_name'])

    st.plotly_chart(px.histogram(d, x='earliest_lap', nbins=20, title='Earliest Pit Stop Lap'), use_container_width=True, key='plot_earliest_lap_eda_1')
    st.plotly_chart(px.histogram(d, x='latest_lap', nbins=20, title='Latest Pit Stop Lap'), use_container_width=True, key='plot_latest_lap')
    st.plotly_chart(px.box(d, x='early_stop_flag', y='points', title='Points vs Early Stop'), use_container_width=True, key='plot_early_stop')
    st.plotly_chart(px.box(d, x='late_stop_flag', y='points', title='Points vs Late Stop'), use_container_width=True, key='plot_late_stop')
    st.plotly_chart(px.scatter(d, x='stop_range', y='position_change', trendline='ols', title='Stop Range vs Position Change'), use_container_width=True, key='plot_range_vs_change')

    d['efficiency'] = d['points'] / d['total_stops']
    eff_df = d.groupby('constructor_name')['efficiency'].mean().reset_index().sort_values('efficiency', ascending=False).head(10)
    st.plotly_chart(px.bar(eff_df, x='constructor_name', y='efficiency', title='Top Teams by Pit Efficiency'), use_container_width=True, key='plot_team_efficiency')

# Main Streamlit app logic - loads data, runs predictions, and renders analysis
def main():
    races, results, pit_stops, drivers, constructors, circuits, status = load_data()
    df = preprocess_data(races, results, pit_stops, drivers, constructors, circuits, status)

    st.sidebar.header("Prediction Parameters")
    df['year'] = pd.to_numeric(df['year'], errors='coerce')
    circuits_list = sorted(df['circuit_name'].dropna().unique())
    teams_list = sorted(df['constructor_name'].dropna().unique())

    sel_circ = st.sidebar.selectbox("Circuit", circuits_list)
    sel_team = st.sidebar.selectbox("Team", teams_list)
    stops = st.sidebar.slider("Pit Stops", 0, 5, 2)
    grid = st.sidebar.slider("Grid Position", 1, 20, 5)
    miny, maxy = int(df['year'].min()), int(df['year'].max())
    yr = st.sidebar.slider("Year", miny, maxy, maxy)

    filtered = df[(df['circuit_name'] == sel_circ) & (df['constructor_name'] == sel_team) & (df['year'] == yr)]
    avg = filtered['avg_stop_duration'].mean()
    if np.isnan(avg) or avg > 50:
      avg = df['avg_stop_duration'].mean()

    use_feature_elimination = st.sidebar.checkbox("Use Feature Elimination (SelectFromModel)", value=False)

    if st.sidebar.button("Train & Predict"):
        model, name, rmse, r2, selector = train_models(df, use_feature_elimination=use_feature_elimination)
        st.sidebar.success(f"{name} RMSE={rmse:.2f} R²={r2:.2f}")
        cid = df[df['circuit_name'] == sel_circ]['circuitId'].iloc[0]
        tid = df[df['constructor_name'] == sel_team]['constructorId'].iloc[0]
        est_earliest = filtered['earliest_lap'].mean()
        if np.isnan(est_earliest) or est_earliest > 50:
            est_earliest = df['earliest_lap'].mean()

        position_change = grid - 5
        change_per_stop = position_change / max(1, stops)
        est_earliest = filtered['earliest_lap'].mean()
        if np.isnan(est_earliest) or est_earliest > 50:
            est_earliest = df['earliest_lap'].mean()

        pit_timing_ratio = est_earliest / (filtered['latest_lap'].mean() if not np.isnan(filtered['latest_lap'].mean()) else df['latest_lap'].mean())

        # Estimate expected pit stops for the selected team/year
        expected_stops = filtered['total_stops'].mean()
        if np.isnan(expected_stops):
            expected_stops = df['total_stops'].mean()

        stop_penalty = 1 if stops > expected_stops else 0

        # Adjust avg_stop_duration or input penalty if needed in the future

        # Construct the input DataFrame for prediction
        inp = pd.DataFrame({
            'grid': [grid],
            'total_stops': [stops],
            'avg_stop_duration': [avg],
            'earliest_lap': [est_earliest],
            'pit_timing_ratio': [pit_timing_ratio],
            'circuitId': [cid],
            'constructorId': [tid]
        })
        pts = model.predict(inp)[0]
        if stop_penalty:
            pts *= 0.6  # apply penalty for unrealistic stop count
        if stop_penalty:
            st.warning("⚠️ Strategy Penalty Applied: Pit stops exceed team/year average")  # Cap at max possible with fastest lap
        rp = int(round(pts))
        final_pos = int(round(pts))
        extra_stops = stops - expected_stops
        if extra_stops > 0:
            final_pos += int(extra_stops * 2.5)
            final_pos = min(final_pos, 20)
        elif extra_stops < 0 and stops > 0:
            final_pos -= int(abs(extra_stops) * 1.5)
            st.success("🟢 Strategy Boost Applied: Optimized Pit Stops")
        final_pos = max(1, min(final_pos, 20))
        points_map = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}
        pts = points_map.get(final_pos, 0)
        # Fastest lap bonus simulation
        fastest_lap = np.random.choice([0, 1]) if final_pos <= 10 else 0
        pts += fastest_lap
        pts = min(pts, 26)  # Ensure max with fastest lap is 26

        st.subheader("🏁 Prediction")
        
        col1, col2 = st.columns(2)
        
        
        if final_pos == 1:
            st.success("🥇 Predicted to Win the Race!")
        elif final_pos <= 3:
            st.info("🏆 Podium Finish Expected")
        elif final_pos > 10:
            st.warning("📉 Predicted Outside Points Zone")

        if fastest_lap:
            st.info("Fastest Lap Point Awarded 🟢 (1 extra point)")

        col1.metric("Predicted Points", f"{pts:.2f}")
        col2.metric("Estimated Position", str(final_pos))
        st.write("🔍 Prediction Input")
        st.dataframe(inp)

    statistical_analysis(df)
    create_eda(df, races=races, pit_stops=pit_stops, circuits=circuits)
    

if __name__ == '__main__':
    main()

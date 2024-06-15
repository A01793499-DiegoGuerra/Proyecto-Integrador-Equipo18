import pandas as pd
import numpy as np
import math
import streamlit as st

import joblib
import os

from datetime import datetime

from scipy.optimize import differential_evolution

def convert_to_unix_time(date):
    return int((date - datetime(1970, 1, 1)).total_seconds())

def feature_engineering(df_interest):
    df_interest['hour_sin'] = np.sin(2 * np.pi * df_interest['hour']/23.0)
    df_interest['hour_cos'] = np.cos(2 * np.pi * df_interest['hour']/23.0)
    df_interest['day_sin'] = np.sin(2 * np.pi * df_interest['dow_n']/6.0)
    df_interest['day_cos'] = np.cos(2 * np.pi * df_interest['dow_n']/6.0)
    df_interest['month_sin'] = np.sin(2 * np.pi * df_interest['month']/11.0)
    df_interest['month_cos'] = np.cos(2 * np.pi * df_interest['month']/11.0)
    df_interest['unix_time'] = df_interest['sale_timeStamp'].apply(convert_to_unix_time)
    df_interest.drop(['sale_timeStamp'], axis = 1, inplace = True)
    return df_interest


def load_elasticity_models(path, filename):
    full_path = os.path.join(path, filename)
    models = joblib.load(full_path)
    return models['model']

def load_demand_models(path, filename):
    full_path = os.path.join(path, filename)
    models = joblib.load(full_path)
    return models['model'], models['scaler']

def setup_loaded_model(input_data, features, path, preloaded_model):
    model, scaler = load_demand_models(path, preloaded_model)
    alt_df  = feature_engineering(input_data[features].copy())
    current_price = input_data['precio_neto'].mean()
    
    return model, alt_df, scaler, current_price, alt_df

def eval_gross_margin(individual, train_data, model, scaler, current_price, elasticity, feature_names, type = 'max', delta = 0.05, fn_verbose = False):
    try:
        price = individual[0]
        # Create a dataframe to simulate the scenario with the evaluated price
        mean_values = np.mean(train_data[:, 1:], axis=0) 
        data_with_price = np.insert(mean_values, 0, price)
        temp_features = pd.DataFrame([data_with_price], columns = feature_names)
        
        # Calculate the scaled features for the model
        scaled_features = scaler.transform(temp_features)
        predicted_demand = model.predict(scaled_features)[0]
        
        # Extract cost of goods sold from the temporary features, ensuring it corresponds to the simulated scenario
        cost_of_goods_sold = temp_features['costo_neto'].iloc[0]
        
        gross_margin = (price - cost_of_goods_sold) * predicted_demand

        epsilon = 1e-6
        penalty_factor = 100
        price_deviation = abs(price - current_price) / current_price
        penalty = penalty_factor * max(epsilon, price_deviation - delta) ** (-elasticity)
                                                                                    
        gross_margin -= penalty
        
        if type == 'max':
            result = gross_margin
        if type == 'min':
            result = -gross_margin

        if fn_verbose:
            print(f"Evaluating price: {price}, Gross Margin: {gross_margin}, Result: {result}")
            
        return (result,)

    except Exception as e:
        print("Error in evalGrossMargin:", e)
        return (0,)  # Return a default or zero fitness in case of error


def optimize_price_diff(input_data, features, target, elasticity, lower_bound_ratio, upper_bound_ratio, technique = None, delta = 0.05, in_plot = False, fn_verbose = False, save_model = False, preloaded_model = None, path = None):
    if preloaded_model is not None:
        print(f'Executing from loaded model: {preloaded_model}')
        model, alt_df, scaler, current_price, features_refined = setup_loaded_model(input_data, features, path, preloaded_model)
        X_train = alt_df.values
    else:
        model, X_train, scaler, current_price, features_refined = technique(input_data, features, target, in_plot, save_model, path)

    lower_bound = current_price * lower_bound_ratio
    upper_bound = current_price * upper_bound_ratio
    
    bounds = [(lower_bound, upper_bound)]
    print('...............................................................')
    print('Optimizer Sequence: Differential Evolution')
    def negative_eval_gross_margin(individual, *args):
        return -eval_gross_margin(individual, *args)[0]

    result = differential_evolution(negative_eval_gross_margin, bounds,
                                    args = (X_train, model, scaler, current_price, elasticity, features_refined.columns, 'max', delta, fn_verbose),
                                    strategy = 'currenttobest1bin', maxiter = 1000, popsize = 50, tol = 0.01, mutation = (0.5, 1), recombination = 0.7, disp = True)

    optimal_price = result.x[0]
    max_gross_margin = -result.fun

    print(f'Optimal price: ${optimal_price:.2f}')
    print(f'Maximum Gross Margin: ${max_gross_margin:.2f}')

    return optimal_price, max_gross_margin


def interactive_view():

    st.title('Price Optimization with Differential Evolution')

    combustible = st.radio('What fuel do you want to analyze', ['Magna', 'Premium', 'Diesel'], index = 0,)

    path = 'models/'
    elasticity_filename = 'model_elasticity_' + combustible.lower() + '.pkl'
   

    if combustible.lower()  == 'magna':
        suffix = 'lgb'
    elif combustible.lower()  == 'premium':
        suffix = 'rf'
    elif combustible.lower()  == 'diesel':
        suffix = 'xgb'

    demand_filename = 'model_demand_' + combustible.lower() + '_' + suffix + '.pkl'

    try:
        price_elasticity = load_elasticity_models(path, elasticity_filename).coef_[0]
        model, scaler = load_demand_models(path, demand_filename)
        preloaded = True
    except FileNotFoundError:
        st.warning("Pretrained models not found")
        preloaded = False


    model_features = ['precio_neto', 'costo_neto', 'precio_brent', 'hour', 'month', 'dow_n', 'sale_timeStamp']
    model_target = ['volumen_despachado']


    current_datetime = datetime.now()

    precio_neto = st.number_input('Precio Neto', min_value = 0.0, max_value = 50.0, value = 25.0)
    costo_neto = st.number_input('Costo Neto', min_value = 0.0, max_value = 50.0, value = 25.0)
    precio_brent = st.number_input('Precio Brent', min_value = 0.0, max_value = 200.0, value = 100.0)
    hour = st.slider('Hour', 0, 23, 12)
    month = st.slider('Month', 1, 12, 6)
    dow_n = st.slider('Day of Week (numeric)', 0, 6, 3)
    sale_date = st.date_input('Sale Date', pd.to_datetime(current_datetime))
    st.write('Enter the required parameters for prediction and optimization.')
    sale_timeStamp = pd.to_datetime(f"{sale_date} {hour}:00:00")

    lower_bound_ratio = st.slider('Lower Bound Ratio', 0.5, 1.0, 0.7)
    upper_bound_ratio = st.slider('Upper Bound Ratio', 1.0, 2.0, 1.3)
    delta = st.slider('Delta', 0.01, 0.1, 0.05)

    input_data = pd.DataFrame({
        'precio_neto': [precio_neto],
        'costo_neto': [costo_neto],
        'precio_brent': [precio_brent],
        'hour': [hour],
        'month': [month],
        'dow_n': [dow_n],
        'sale_timeStamp': [sale_timeStamp]
    })


    input_data_demand = feature_engineering(input_data.copy())

    if preloaded:
        scaled_input = scaler.transform(input_data_demand)
        predicted_demand = model.predict(scaled_input)[0]
        st.write(f'Predicted Demand wo Price Optimization: {predicted_demand}')
    else:
        st.warning("Pretrained models not found")


    if st.button('Optimize Price'):
        if preloaded: 
            price, gross_margin  = optimize_price_diff(input_data, model_features, model_target, price_elasticity, lower_bound_ratio, upper_bound_ratio, 
                                                    delta = delta, preloaded_model = demand_filename, path = path)
            st.write(f'Optimal price: ${price:.2f}')
            st.write(f'Maximum Gross Margin: ${gross_margin:.2f}')

        else:
            st.warning("Pretrained models not found")



if __name__ == '__main__':
    interactive_view()

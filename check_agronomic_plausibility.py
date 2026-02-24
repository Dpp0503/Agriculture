import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def check_agronomic_plausibility():
    print("Checking Agronomic Plausibility...")
    df = pd.read_csv('d:/Rice/advanced_features.csv')
    
    # 1. Correlations with Yield
    target = 'Yield (Ton./Ha.)'
    features_to_check = [
        'Vegetative_Heat_Stress_Weeks',
        'Vegetative_Heat_x_Humidity',
        'Ripening_Rainfall_Total',
        'Ripening_Rainy_Weeks',
        'Reproductive_Mean_Solar',
        'Germination_Mean_Temp'
    ]
    
    with open('d:/Rice/agronomic_report.txt', 'w') as f:
        f.write(f"Correlations with {target}:\n")
        for feat in features_to_check:
            if feat in df.columns:
                corr = df[feat].corr(df[target])
                msg = f"{feat}: {corr:.4f}"
                print(msg)
                f.write(msg + "\n")
            else:
                msg = f"{feat}: Not found in dataframe"
                print(msg)
                f.write(msg + "\n")
            
    # 2. Plot Ripening Rain vs Yield
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x='Ripening_Rainfall_Total', y=target, alpha=0.5)
    plt.title('Yield vs Ripening Rainfall')
    plt.xlabel('Ripening Rainfall (mm)')
    plt.ylabel('Yield (Ton/Ha)')
    plt.savefig('d:/Rice/yield_vs_ripening_rain.png')
    plt.close()
    
    # 3. Plot Heat Stress vs Yield
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x='Vegetative_Heat_Stress_Weeks', y=target, alpha=0.5)
    plt.title('Yield vs Vegetative Heat Stress')
    plt.xlabel('Vegetative Heat Stress (Weeks > 35C)')
    plt.ylabel('Yield (Ton/Ha)')
    plt.savefig('d:/Rice/yield_vs_heat_stress.png')
    plt.close()

if __name__ == "__main__":
    check_agronomic_plausibility()


import data_service as ds
import pandas as pd
import numpy as np

def analyze_breadth_distribution():
    sectors = ['Technology Sector', 'Financial Services Sector', 'Energy Sector']
    
    print(f"{'Sector':<25} | {'Min':<6} | {'10%':<6} | {'20%':<6} | {'Avg':<6} | {'80%':<6} | {'90%':<6} | {'Max':<6}")
    print("-" * 85)
    
    for s_name in sectors:
        # Get EMA Trend Setup Count
        df_setup = ds.get_breadth_data(s_name, metric='ema_trend_setup', days=1825)
        # Get Active Count
        df_active = ds.get_breadth_data(s_name, metric='active_count', days=1825)
        
        if df_setup is not None and not df_active.empty:
            # Join and Calc %
            df = df_setup.join(df_active, lsuffix='_setup', rsuffix='_total', how='inner')
            df['pct'] = (df['Value_setup'] / df['Value_total']) * 100
            
            stats = df['pct'].describe(percentiles=[0.1, 0.2, 0.8, 0.9])
            
            print(f"{s_name:<25} | {stats['min']:<6.1f} | {stats['10%']:<6.1f} | {stats['20%']:<6.1f} | {stats['mean']:<6.1f} | {stats['80%']:<6.1f} | {stats['90%']:<6.1f} | {stats['max']:<6.1f}")

if __name__ == "__main__":
    analyze_breadth_distribution()

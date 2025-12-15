import streamlit as st
import plotly.express as px
import data_service as ds
import pandas as pd

st.set_page_config(page_title="US Market Sector Monitor", layout="wide")

st.title("US Market Sector Monitor")

# Sidebar
st.sidebar.header("Settings")
period_map = {
    "1mo": 30,
    "3mo": 90,
    "6mo": 180,
    "1y": 365,
    "ytd": 365, # Approx, handle better if needed
    "5y": 1825
}
selected_period_label = st.sidebar.selectbox("Select Time Period", list(period_map.keys()), index=3)
days = period_map[selected_period_label]

weight_type_label = st.sidebar.radio("ETF Weighting", ["Cap Weighted", "Equal Weighted"], index=0)
weight_type = "cap" if weight_type_label == "Cap Weighted" else "equal"

# Update Section
st.sidebar.markdown("---")
st.sidebar.subheader("Data Update")

# Show last update date
try:
    latest_date = ds.get_latest_data_date()
    st.sidebar.text(f"Last Data: {latest_date}")
except:
    st.sidebar.text("Last Data: Unknown")

# Progress Bar Container
progress_bar = st.sidebar.progress(0)
status_text = st.sidebar.empty()

def update_progress(msg, val):
    status_text.text(msg)
    progress_bar.progress(val)

if st.sidebar.button("⚡ Atualização Rápida (Gap)"):
    with st.spinner("Atualizando dados recentes..."):
        try:
            latest = ds.get_latest_data_date()
            if latest:
                ds.update_constituents_data(start_date=latest, progress_callback=update_progress)
                st.sidebar.success("Atualização concluída!")
                st.cache_data.clear()
            else:
                st.sidebar.warning("Base vazia. Rode 'Reset Completo' primeiro.")
        except Exception as e:
            st.sidebar.error(f"Erro: {e}")

if st.sidebar.button("🔄 Reset Completo (Lento)"):
    with st.spinner("Recarregando TODO histórico (Demora muito)..."):
        try:
            # Update ETFs
            update_progress("Updating ETFs...", 0.05)
            ds.update_sector_data(period="10y") 
            
            # Update Constituents
            ds.update_constituents_data(progress_callback=update_progress)
            
            st.sidebar.success("Database updated successfully!")
            st.cache_data.clear() 
        except Exception as e:
            st.sidebar.error(f"Update failed: {e}")

# Data Loading
@st.cache_data
def load_data(days, w_type):
    sectors = ds.get_sector_tickers(weight_type=w_type)
    df = ds.get_sector_data_from_db(period_days=days, weight_type=w_type)
    return df, sectors

@st.cache_data
def load_matrix(w_type):
    # Fetch returns for standard periods
    df_matrix = ds.get_sector_performance_matrix(weight_type=w_type)
    return df_matrix
def load_matrix(w_type):
    # Fetch returns for standard periods
    df_matrix = ds.get_sector_performance_matrix(weight_type=w_type)
    return df_matrix

# Navigation
page = st.sidebar.radio("View", ["Overview", "Performance Matrix", "Momentum Ranking", "Market Breadth"])

if page == "Overview":
    try:
        with st.spinner('Loading data from database...'):
            df_prices, sector_map = load_data(days, weight_type)
            
            if df_prices.empty:
                st.warning("No data found in database. Please click 'Update Database' in the sidebar.")
            else:
                # Performance over the selected period
                perf = (df_prices.iloc[-1] / df_prices.iloc[0] - 1) * 100
                perf = perf.sort_values(ascending=False)
                
                # Create a DataFrame for the chart
                perf_df = pd.DataFrame({'Ticker': perf.index, 'Return (%)': perf.values})
                
                # Map Ticker to Name
                ticker_to_name = {v: k for k, v in sector_map.items()}
                perf_df['Sector'] = perf_df['Ticker'].map(ticker_to_name)
                
                st.header(f"Sector Performance ({selected_period_label})")
                
                fig = px.bar(
                    perf_df, 
                    x='Return (%)', 
                    y='Sector', 
                    orientation='h', 
                    text='Return (%)',
                    color='Return (%)',
                    color_continuous_scale='RdYlGn',
                    title=f"Relative Performance (Last {days} days)"
                )
                fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
                fig.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig, use_container_width=True)

                st.divider()

                st.subheader("Sector Details")
                selected_sector_name = st.selectbox("Select Sector", list(sector_map.keys()))
                selected_ticker = sector_map[selected_sector_name]
                
                if selected_ticker in df_prices.columns:
                    st.line_chart(df_prices[selected_ticker])
                else:
                    st.info("No data for this sector in the selected range.")
            
    except Exception as e:
        st.error(f"Error processing data: {e}")

elif page == "Performance Matrix":
    st.header("Performance Matrix (Returns %)")
    st.write(f"Showing returns for **{weight_type_label}** ETFs.")
    
    try:
        df_matrix = load_matrix(weight_type)
        if df_matrix.empty:
             st.warning("Not enough data to calculate matrix. Try updating the database (needs > 252 days history).")
        else:
            # Add Sector Name column for clarity
            sectors = ds.get_sector_tickers(weight_type=weight_type)
            ticker_to_name = {v: k for k, v in sectors.items()}
            
            df_matrix['Sector'] = df_matrix.index.map(ticker_to_name)
            
            # Reorder columns: Sector first, then periods
            cols = ['Sector'] + [c for c in df_matrix.columns if c != 'Sector']
            df_display = df_matrix[cols]
            
            # Apply styling
            st.dataframe(
                df_display.style.background_gradient(cmap='RdYlGn', subset=['5d', '10d', '20d', '40d', '252d'])
                                .format("{:.2f}%", subset=['5d', '10d', '20d', '40d', '252d']),
                use_container_width=True,
                height=500
            )
    except Exception as e:
        st.error(f"Error calculating matrix: {e}")

elif page == "Momentum Ranking":
    st.header("Momentum Ranking")
    st.markdown("""
    **Formula:**
    `Score = 0.3 * Return(5d-1d) + 0.3 * Return(10d-5d) + 0.2 * Return(20d-10d) + 0.2 * Return(40d-20d)`
    """)
    st.write(f"Ranking for **{weight_type_label}** ETFs.")
    
    try:
        df_mom = ds.get_momentum_ranking(weight_type=weight_type)
        if df_mom.empty:
            st.warning("Not enough data to calculate momentum. Try updating the database.")
        else:
            # Add Sector Name
            sectors = ds.get_sector_tickers(weight_type=weight_type)
            ticker_to_name = {v: k for k, v in sectors.items()}
            
            df_mom['Sector'] = df_mom.index.map(ticker_to_name)
            
            # Reorder
            cols = ['Sector', 'Score', 'R(5-1)', 'R(10-5)', 'R(20-10)', 'R(40-20)']
            df_display = df_mom[cols]
            
            st.dataframe(
                df_display.style.background_gradient(cmap='RdYlGn', subset=['Score'])
                                .format("{:.2f}", subset=['Score', 'R(5-1)', 'R(10-5)', 'R(20-10)', 'R(40-20)']),
                use_container_width=True,
                height=600
            )
    except Exception as e:
        st.error(f"Error calculating momentum: {e}")

elif page == "Market Breadth":
    st.header("Market Breadth Analysis")
    st.write("Percentage of stocks in sector trading above Moving Averages.")
    
    # Sector selection
    sector_map = ds.get_sector_tickers(weight_type='cap') # Just to get names
    selected_sector_breadth = st.selectbox("Select Sector", list(sector_map.keys()), key='breadth_sector')
    
    # Days selection
    days_history = st.slider("History (Days)", 30, 1825, 365)
    
    # Benchmark Selection
    st.subheader("Benchmark Settings")
    
    # Get ETF Tickers for the selected sector
    tickers = ds.SECTORS_CONFIG[selected_sector_breadth]
    cap_ticker = tickers['cap']
    eq_ticker = tickers['equal']
    
    # Options for radio
    benchmark_options = {
        f"Equal Weight ({eq_ticker})": "equal",
        f"Cap Weighted ({cap_ticker})": "cap"
    }
    
    selected_benchmark_label = st.radio(
        "Compare with ETF:", 
        list(benchmark_options.keys()), 
        index=0, 
        horizontal=True
    )
    selected_benchmark_type = benchmark_options[selected_benchmark_label]
    benchmark_ticker = eq_ticker if selected_benchmark_type == 'equal' else cap_ticker
    
    # Visualization Logic
    metrics_config = [
        {'col': 'pct_above_ma5', 'label': '% > MA5', 'color': '#8884d8'},   # Purple
        {'col': 'pct_above_ma10', 'label': '% > MA10', 'color': '#82ca9d'}, # Greenish
        {'col': 'pct_above_ma20', 'label': '% > MA20', 'color': '#ffc658'}, # Yellow/Orange
        {'col': 'pct_above_ma50', 'label': '% > MA50', 'color': '#ff7300'}, # Orange
        {'col': 'pct_above_ma200', 'label': '% > MA200', 'color': '#d32f2f'} # Red
    ]

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # Fetch ETF Data based on selection
    df_etf = ds.get_etf_price_history(selected_sector_breadth, days=days_history, weight_type=selected_benchmark_type)
    
    # Loop through metrics and create charts
    for m in metrics_config:
        metric_key = m['col']
        label = m['label']
        color = m['color']
        
        # Fetch Breadth Data
        df_breadth = ds.get_breadth_data(selected_sector_breadth, metric=metric_key, days=days_history)
        
        if df_breadth is not None and not df_breadth.empty:
            # Create Dual Axis Figure
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            # Trace 1: Breadth (Left Axis)
            fig.add_trace(
                go.Scatter(
                    x=df_breadth.index, 
                    y=df_breadth['Value'], 
                    name=label,
                    line=dict(color=color, width=2),
                    mode='lines' # No markers
                ),
                secondary_y=False
            )
            
            # Trace 2: ETF Price (Right Axis)
            if not df_etf.empty:
                fig.add_trace(
                    go.Scatter(
                        x=df_etf.index,
                        y=df_etf['Close'],
                        name=f"{benchmark_ticker} ({selected_benchmark_type.title()})",
                        line=dict(color='gray', width=1, dash='dot'),
                        mode='lines',
                        opacity=0.5
                    ),
                    secondary_y=True
                )
            
            # Layout
            fig.update_layout(
                title=f"{selected_sector_breadth} - {label} vs {benchmark_ticker}",
                height=350,
                margin=dict(l=20, r=20, t=40, b=20),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            # Y-Axis 1 (Breadth)
            fig.update_yaxes(
                title_text="Breadth (%)", 
                range=[0, 100], 
                secondary_y=False,
                showgrid=True,
                gridcolor='rgba(128,128,128,0.2)'
            )
            
            # Y-Axis 2 (Price)
            fig.update_yaxes(
                title_text=f"{benchmark_ticker} Price", 
                secondary_y=True,
                showgrid=False
            )
            
            # Add 50% threshold line
            fig.add_shape(
                type="line",
                x0=df_breadth.index.min(),
                y0=50,
                x1=df_breadth.index.max(),
                y1=50,
                line=dict(color="gray", width=1, dash="dash"),
                xref="x",
                yref="y" # Left axis
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        else:
            st.warning(f"No data for {label}. Try updating the database.")

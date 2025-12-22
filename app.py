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
                # Update ETFs first (short period to cover gaps)
                ds.update_sector_data(period="1mo")
                # Update Constituents with a 7-day buffer to catch any lagging sectors
                safe_start = latest - pd.Timedelta(days=7)
                ds.update_constituents_data(start_date=safe_start, progress_callback=update_progress)
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
page = st.sidebar.radio("View", ["Overview", "Performance Matrix", "Momentum Ranking", "Momentum Change", "Market Breadth", "Sector Dashboard", "New Highs / Lows", "Sector Charts", "Sector Stocks"])

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
            
            # Reorder columns: Sector first, then Last Price, Date, then periods
            cols = ['Sector', 'Last Price', 'Date'] + [c for c in df_matrix.columns if c not in ['Sector', 'Last Price', 'Date']]
            df_display = df_matrix[cols]
            
            # Apply styling
            st.dataframe(
                df_display.style.background_gradient(cmap='RdYlGn', subset=['5d', '10d', '20d', '40d', '252d'])
                                .format("{:.2f}%", subset=['5d', '10d', '20d', '40d', '252d'])
                                .format("{:.2f}", subset=['Last Price']),
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
            
            # Define formatters
            fmt_score = "{:.2f}"
            fmt_pct = "{:.2f}%"

            # Reorder
            cols = ['Sector', 'Last Price', 'Date', 'Score', 'Score -5d', 'Score -20d', 'Score -50d', 'Score Chg (5d)', 'R(5-1)', 'R(10-5)', 'R(20-10)', 'R(40-20)']
            df_display = df_mom[cols]
            
            st.dataframe(
                df_display.style
                .format({
                    'Score': fmt_score,
                    'Score Chg (5d)': fmt_score,
                    'Score -5d': fmt_score,
                    'Score -20d': fmt_score,
                    'Score -50d': fmt_score,
                    'R(5-1)': fmt_pct,
                    'R(10-5)': fmt_pct,
                    'R(20-10)': fmt_pct,
                    'R(40-20)': fmt_pct,
                    'Last Price': "{:.2f}"
                })
                .background_gradient(cmap='RdYlGn', subset=['Score', 'Score -5d', 'Score -20d', 'Score -50d']),
                use_container_width=True,
                height=600
            )

            # --- Momentum History Chart ---
            st.divider()
            st.subheader("Momentum History Chart")
            
            col_ctrl_1, col_ctrl_2 = st.columns(2)
            with col_ctrl_1:
                 # Slider for view range
                history_days = st.slider("History Length (Days)", min_value=30, max_value=750, value=252, step=10)
            with col_ctrl_2:
                # Comparison Price Overlay
                # Get all options for dropdown
                all_opts = ds.get_all_sector_options() # [{'name':'...', 'ticker':'...'}, ...]
                price_overlay_opts = ["None"] + [o['name'] for o in all_opts]
                price_overlay_sel = st.selectbox("Overlay Price (Right Axis):", price_overlay_opts)

            st.write("**Select Sectors to Compare (Momentum Score):**")
            
            # Use ALL sector options for checkboxes
            # Create a map for name -> ticker
            name_to_ticker_all = {o['name']: o['ticker'] for o in all_opts}
            
            # Organize options by type
            cap_opts = sorted([o for o in all_opts if o['type'] == 'cap'], key=lambda x: x['sector'])
            eq_opts = sorted([o for o in all_opts if o['type'] == 'equal'], key=lambda x: x['sector'])
            
            # Default selection logic needs to match new name format "Energy (XLE)"
            # Current view "Energy". We need to find the option that matches this sector and current weight type.
            # actually we can just look up by ticker if we have it.
            # Let's find the tickers of the current view's top 3.
            # But df_display only has Sector Name "Energy". 
            # We know the current mode (weight_type).
            current_top_sectors = df_display['Sector'].head(3).tolist() # ["Energy", "Tech"]
            
            # Build list of default names to check
            default_chk_names = []
            for s_name in current_top_sectors:
                # Find matching option in all_opts
                match = next((o for o in all_opts if o['sector'] == s_name and o['type'] == weight_type), None)
                if match:
                    default_chk_names.append(match['name'])
            
            selected_sectors_chart = []
            
            col_sel_1, col_sel_2 = st.columns(2)
            
            with col_sel_1:
                st.caption("Cap Weighted")
                for opt in cap_opts:
                    is_checked = opt['name'] in default_chk_names
                    if st.checkbox(opt['name'], value=is_checked, key=f"chk_mom_{opt['ticker']}"):
                        selected_sectors_chart.append(opt['name'])
                        
            with col_sel_2:
                st.caption("Equal Weighted")
                for opt in eq_opts:
                    is_checked = opt['name'] in default_chk_names
                    if st.checkbox(opt['name'], value=is_checked, key=f"chk_mom_{opt['ticker']}"):
                        selected_sectors_chart.append(opt['name'])
            
            if selected_sectors_chart or price_overlay_sel != "None":
                import plotly.graph_objects as go
                from plotly.subplots import make_subplots
                
                # Create figure with secondary y-axis
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                
                # 1. Momentum Scores (Left Axis)
                min_score, max_score = 0, 0
                has_mom_data = False
                
                # Collect dates for gap removal
                all_dates = pd.Index([])

                for s_name in selected_sectors_chart:
                    ticker = name_to_ticker_all[s_name]
                    # Fetch requested history length
                    hist = ds.get_momentum_history(ticker, period_days=history_days)
                    if not hist.empty:
                        has_mom_data = True
                        if all_dates.empty: all_dates = hist.index
                        else: all_dates = all_dates.union(hist.index)
                        
                        min_score = min(min_score, hist['Score'].min())
                        max_score = max(max_score, hist['Score'].max())
                        
                        fig.add_trace(
                            go.Scatter(x=hist.index, y=hist['Score'], name=s_name, mode='lines'),
                            secondary_y=False
                        )

                # 2. Price Compare (Right Axis)
                if price_overlay_sel != "None":
                    ticker_p = name_to_ticker_all[price_overlay_sel]
                    price_hist = ds.get_price_history(ticker_p, period_days=history_days)
                    
                    if not price_hist.empty:
                        if all_dates.empty: all_dates = price_hist.index
                        else: all_dates = all_dates.union(price_hist.index)
                        
                        fig.add_trace(
                            go.Scatter(
                                x=price_hist.index, 
                                y=price_hist, 
                                name=f"{price_overlay_sel} Price", 
                                mode='lines',
                                line=dict(dash='dot', width=1)
                            ),
                            secondary_y=True
                        )

                # Background Colors (Green/Red)
                # Only apply if we have momentum data to make sense of the scale
                if has_mom_data:
                    # Margins for background
                    y_top = max(max_score, 1.1) * 1.05
                    y_bottom = min(min_score, 0.9)
                    if y_bottom < 0: y_bottom *= 1.05
                    else: y_bottom *= 0.95
                    
                    # Green Zone (> 100) - Note: Score is percentage now, so 1%?
                    # Wait, previous user request: "score multipled by 100".
                    # So 1.0 score becomes 100.0?
                    # Previous implementation: df['Score'] = df['Score'] * 100
                    # Original Score around 1.0? 
                    # Let's check verify output: "Score 0.011109" (raw). * 100 = 1.11.
                    # User request "light green above 1 and light red below 1".
                    # If score is 1.11, it is > 1.
                    # So Threshold is 1.0 (if stored as percentage 1%) or 100?
                    # Raw score formula: 0.3 * Ret. Returns are like 0.05 (5%).
                    # Score is approx weighted avg of returns.
                    # If returns are ~+5%, Score ~ 0.05. * 100 = 5.0.
                    # Wait, looking at previously verified data (Step 163):
                    # Score 1.82 means 1.82%.
                    # User said "green above 1". This implies 1%.
                    # OK, threshold is 1.0.

                    fig.add_hrect(
                        y0=1, y1=y_top,
                        fillcolor="rgba(0, 255, 0, 0.05)",
                        line_width=0,
                        layer="below",
                        secondary_y=False
                    )
                    fig.add_hrect(
                        y0=y_bottom, y1=1,
                        fillcolor="rgba(255, 0, 0, 0.05)",
                        line_width=0,
                        layer="below",
                        secondary_y=False
                    )

                # Remove Gaps
                if not all_dates.empty:
                    dt_all = pd.date_range(start=all_dates.min(), end=all_dates.max())
                    dt_breaks = dt_all.difference(all_dates)
                    fig.update_xaxes(rangebreaks=[dict(values=dt_breaks)])

                fig.update_yaxes(title_text="Momentum Score (%)", secondary_y=False)
                fig.update_yaxes(title_text="Price ($)", secondary_y=True, showgrid=False)
                fig.update_layout(title="Momentum Score & Price Evolution", hovermode="x unified")
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Select sectors to view chart.")
    except Exception as e:
        st.error(f"Error calculating momentum: {e}")

elif page == "Momentum Change":
    st.header("Momentum Change Analysis")
    st.markdown("""
    **Indicator:** Daily variation of the Momentum Score (`Score Today - Score Yesterday`).
    - **Green Bars (+):** Momentum is improving.
    - **Red Bars (-):** Momentum is deteriorating.
    - **Blue Line:** 5-Day Moving Average of the change (Trend).
    
    Use this to identify sectors with **negative momentum** (low score) but **positive change** (recovering).
    """)
    
    # Settings
    col_mom_1, col_mom_2 = st.columns(2)
    with col_mom_1:
        history_days_mom = st.slider("History (Days)", 30, 750, 120, step=10, key='mom_change_days')
    
    # Get all sectors for current weight type
    all_opts = ds.get_all_sector_options()
    current_opts = [o for o in all_opts if o['type'] == weight_type]
    # Sort alphabetically
    current_opts.sort(key=lambda x: x['name'])
    
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    # Name to ticker map
    name_to_ticker = {o['name']: o['ticker'] for o in all_opts}
    
    # Display in Single Column
    for idx, opt in enumerate(current_opts):
        s_name = opt['name']
        ticker = opt['ticker']
        
        # Container for each chart
        with st.container():
            # Fetch History
            df_hist = ds.get_momentum_history(ticker, period_days=history_days_mom + 10) 
            
            if not df_hist.empty:
                # Calculate Indicators
                df_hist['Diff'] = df_hist['Score'].diff()
                df_hist['MA5_Diff'] = df_hist['Diff'].rolling(window=5).mean()
                
                # Filter to requested days
                df_plot = df_hist.tail(history_days_mom)
                
                if df_plot.empty:
                    st.warning(f"No recent data for {s_name}")
                    continue

                # Current Score
                curr_score = df_plot['Score'].iloc[-1]
                
                # Chart
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                
                # 1. Daily Change (Bars)
                colors = ['green' if x >= 0 else 'red' for x in df_plot['Diff']]
                
                fig.add_trace(
                    go.Bar(
                        x=df_plot.index, 
                        y=df_plot['Diff'], 
                        name='Daily Change',
                        marker_color=colors,
                        opacity=0.6,
                        showlegend=False
                    ),
                    secondary_y=False
                )
                
                # 2. MA5 of Change (Line)
                fig.add_trace(
                    go.Scatter(
                        x=df_plot.index,
                        y=df_plot['MA5_Diff'],
                        name='5d MA',
                        line=dict(color='blue', width=2),
                        showlegend=True
                    ),
                    secondary_y=False
                )
                
                # 3. Momentum Score (Context - Right Axis)
                fig.add_trace(
                    go.Scatter(
                        x=df_plot.index,
                        y=df_plot['Score'],
                        name='Mom Score',
                        line=dict(color='gray', width=1, dash='dot'),
                        opacity=0.5,
                        showlegend=True
                    ),
                    secondary_y=True
                )
                
                # Layout
                fig.update_layout(
                    title=f"<b>{s_name}</b> (Score: {curr_score:.2f})",
                    height=350,
                    margin=dict(l=20, r=20, t=40, b=20),
                    hovermode="x unified",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                
                fig.update_yaxes(title_text="Change", secondary_y=False)
                fig.update_yaxes(secondary_y=True, showgrid=False)
                
                # Remove gaps
                dt_all = pd.date_range(start=df_plot.index.min(), end=df_plot.index.max())
                dt_breaks = dt_all.difference(df_plot.index)
                fig.update_xaxes(rangebreaks=[dict(values=dt_breaks)])
                
                st.plotly_chart(fig, use_container_width=True)
                st.divider()
            else:
                st.warning(f"No data for {s_name}")

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
            
            # Remove Gaps (Weekends/Holidays)
            # Combine dates from both dataframes to be safe
            all_dates = df_breadth.index
            if not df_etf.empty:
                all_dates = all_dates.union(df_etf.index)
            
            if not all_dates.empty:
                dt_all = pd.date_range(start=all_dates.min(), end=all_dates.max())
                dt_breaks = dt_all.difference(all_dates)
                fig.update_xaxes(rangebreaks=[dict(values=dt_breaks)])
            
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

elif page == "Sector Dashboard":
    st.header("Sector Dashboard")
    st.markdown("Consolidated view of **Momentum Scores** and **Market Breadth** (% Stocks > MA).")
    
    # Weight Type Selection
    col_dash_1, col_dash_2 = st.columns([1, 4])
    with col_dash_1:
         weight_type_dash = st.radio("Weight Type:", ['Cap Weighted', 'Equal Weighted'], index=0, key='dash_weight')
    
    w_type = 'cap' if weight_type_dash == 'Cap Weighted' else 'equal'
    
    # 1. Fetch Consolidated Data
    df_dash = ds.get_dashboard_data(weight_type=w_type)
    
    if not df_dash.empty:
        # Prep DataFrame for Display
        # Columns in df_dash: [Score, R(5-1)..., Last Price, Date, Score -5d, -20d, -50d, Score Chg (5d), pct_above_ma5...]
        
        # Select Cols
        cols_to_show = [
            'Sector', 'Date',
            'Score', 'Score -5d', 'Score -20d', 'Score -50d', 
            'pct_above_ma5', 'pct_above_ma20', 'pct_above_ma20_5d', 'pct_above_ma50', 'pct_above_ma200'
        ]
        
        # Filter existing cols
        display_cols = [c for c in cols_to_show if c in df_dash.columns]
        df_view = df_dash[display_cols].copy()
        
        # Rename Breadth Cols for compact view
        rename_map = {
            'pct_above_ma5': '% > MA5',
            'pct_above_ma20': '% > MA20',
            'pct_above_ma20_5d': '% > MA20 -5d',
            'pct_above_ma50': '% > MA50',
            'pct_above_ma200': '% > MA200'
        }
        df_view.rename(columns=rename_map, inplace=True)
        
        # Custom formatters to handle None/NaN safely
        def fmt_pct(x):
            return "{:.1f}%".format(x) if pd.notnull(x) else "-"
            
        def fmt_score(x):
            return "{:.2f}".format(x) if pd.notnull(x) else "-"

        # Styling
        import matplotlib.colors as mcolors
        import matplotlib.cm as cm
        import numpy as np

        def color_map(val, vmin, vmax, cmap_name='RdYlGn'):
            if pd.isna(val):
                return "" # No style for NaNs
            
            # Normalize
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax, clip=True)
            cmap = cm.get_cmap(cmap_name)
            rgba = cmap(norm(val))
            color = mcolors.to_hex(rgba)
            return f'background-color: {color}; color: black;' # Force black text for contrast if needed, or smart detection

        # Apply specific styling
        # Since style.map/applymap is slow or complex to pass args per column group easily in one go if ranges differ,
        # we can use simple lambdas.
        
        st.dataframe(
            df_view.style
            .format({
                'Score': fmt_score,
                'Score -5d': fmt_score,
                'Score -20d': fmt_score,
                'Score -50d': fmt_score,
                '% > MA5': fmt_pct,
                '% > MA20': fmt_pct,
                '% > MA20 -5d': fmt_pct,
                '% > MA50': fmt_pct,
                '% > MA200': fmt_pct,
            })
            # Gradient for Scores (-10 to 10)
            .map(lambda x: color_map(x, -3, 3, 'RdYlGn'), subset=['Score', 'Score -5d', 'Score -20d', 'Score -50d'])
            # Gradient for Percentages (0 to 100)
            .map(lambda x: color_map(x, 0, 100, 'RdYlGn'), subset=['% > MA5', '% > MA20', '% > MA20 -5d', '% > MA50', '% > MA200']),
            height=600,
            use_container_width=True,
            column_config={
                "Sector": st.column_config.TextColumn("Sector", width="medium"),
            }
        )
    else:
        st.info("No data available.")

elif page == "New Highs / Lows":
    st.header("New Highs / New Lows (252 Days)")
    st.markdown("Number of stocks making new 52-week Highs and Lows per sector.")
    
    # Sector Selection
    # Get sector names
    sector_map = ds.get_sector_tickers(weight_type='cap')
    sector_names = list(sector_map.keys())
    
    selected_sector_hl = st.selectbox("Select Sector", sector_names, key='hl_sector')
    
    # Days History
    days_hl = st.slider("History (Days)", 30, 1825, 365, key='hl_days')
    
    # Fetch Data
    df_hl = ds.get_sector_high_low_data(selected_sector_hl, days=days_hl)
    
    # Get total constituents
    total_constituents = ds.get_sector_constituent_count(selected_sector_hl)
    
    if not df_hl.empty:
        # Display Total Count Above Chart (Boxed)
        with st.container(border=True):
            st.metric("Total Stocks in Sector", total_constituents)

        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        # ... (Chart code remains mostly same, just ensuring indent/context) ...
        # Calculate Percentages
        if total_constituents > 0:
            df_hl['pct_high'] = (df_hl['new_highs_252'] / total_constituents)
            df_hl['pct_low'] = (df_hl['new_lows_252'] / total_constituents)
            df_hl['pct_net'] = (df_hl['Net'] / total_constituents)
        else:
             df_hl['pct_high'] = 0; df_hl['pct_low'] = 0; df_hl['pct_net'] = 0

        # Create Subplots: 3 Rows with Secondary Y Axis
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, 
                            vertical_spacing=0.15, 
                            subplot_titles=("New Highs", "New Lows", "Net (Highs - Lows)"),
                            row_heights=[0.28, 0.28, 0.44], 
                            specs=[[{"secondary_y": True}], [{"secondary_y": True}], [{"secondary_y": True}]])
                            
        # 1. New Highs (Row 1)
        max_h = df_hl['new_highs_252'].max()
        if max_h == 0: max_h = 10 # Default buffer
        r_h = [0, max_h * 1.1] # 10% buffer
        r_h_pct = [0, r_h[1] / total_constituents] if total_constituents else [0, 1]

        fig.add_trace(go.Bar(x=df_hl.index, y=df_hl['new_highs_252'], name='New Highs', marker_color='green'), row=1, col=1, secondary_y=False)
        fig.add_trace(go.Scatter(x=df_hl.index, y=df_hl['pct_high'], name='% Highs', mode='lines', line=dict(width=0), opacity=0), row=1, col=1, secondary_y=True)
        
        fig.update_yaxes(range=r_h, row=1, col=1, secondary_y=False)
        fig.update_yaxes(range=r_h_pct, row=1, col=1, secondary_y=True)

        # 2. New Lows (Row 2)
        max_l = df_hl['new_lows_252'].max()
        if max_l == 0: max_l = 10
        r_l = [0, max_l * 1.1]
        r_l_pct = [0, r_l[1] / total_constituents] if total_constituents else [0, 1]

        fig.add_trace(go.Bar(x=df_hl.index, y=df_hl['new_lows_252'], name='New Lows', marker_color='red'), row=2, col=1, secondary_y=False)
        fig.add_trace(go.Scatter(x=df_hl.index, y=df_hl['pct_low'], name='% Lows', mode='lines', line=dict(width=0), opacity=0), row=2, col=1, secondary_y=True)
        
        fig.update_yaxes(range=r_l, row=2, col=1, secondary_y=False)
        fig.update_yaxes(range=r_l_pct, row=2, col=1, secondary_y=True)
        
        # 3. Net Chart (Row 3)
        min_n, max_n = df_hl['Net'].min(), df_hl['Net'].max()
        if min_n == 0 and max_n == 0: min_n, max_n = -10, 10
        
        # Add buffer
        span = max_n - min_n
        if span == 0: span = 10
        r_n = [min_n - (span*0.1), max_n + (span*0.1)]
        # Force 0 to be included if desired, or just let it float? User wants 0 aligned.
        # If we scale strictly by / Total, 0 aligns with 0 automatically.
        r_n_pct = [r_n[0] / total_constituents, r_n[1] / total_constituents] if total_constituents else [-1, 1]
        
        net_colors = ['green' if x >= 0 else 'red' for x in df_hl['Net']]
        fig.add_trace(go.Bar(x=df_hl.index, y=df_hl['Net'], name='Net', marker_color=net_colors), row=3, col=1, secondary_y=False)
        fig.add_trace(go.Scatter(x=df_hl.index, y=df_hl['pct_net'], name='% Net', mode='lines', line=dict(width=0), opacity=0), row=3, col=1, secondary_y=True)
        
        fig.update_yaxes(range=r_n, row=3, col=1, secondary_y=False)
        fig.update_yaxes(range=r_n_pct, row=3, col=1, secondary_y=True)
        
        # Configure Axes
        # Remove Gaps
        dt_all = pd.date_range(start=df_hl.index.min(), end=df_hl.index.max())
        dt_breaks = dt_all.difference(df_hl.index)
        fig.update_xaxes(rangebreaks=[dict(values=dt_breaks)])
        
        # Format Right Axis as Percentage and Remove Grid
        fig.update_yaxes(tickformat=".1%", showgrid=False, secondary_y=True)
        
        fig.update_layout(height=900, hovermode="x unified", barmode='group', showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
        # Display Stats (Boxed)
        with st.container(border=True):
            st.write("###### Latest Readings")
            latest = df_hl.iloc[-1]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("New Highs", int(latest['new_highs_252']))
            c2.metric("New Lows", int(latest['new_lows_252']))
            c3.metric("Net", int(latest['Net']), delta=int(latest['Net']))
            c4.metric("Total Stocks", total_constituents)
        
    else:
        st.info("No New Highs/Lows data available. Please ensure backfill is complete.")

elif page == "Sector Charts":
    st.header("Sector Charts (Last 60 Days)")
    
    # Get all sectors
    sector_opts = ds.get_sector_tickers(weight_type='cap') # {Name: Ticker}
    
    # Iterate over all sectors
    for s_name in sorted(sector_opts.keys()):
        s_ticker = sector_opts[s_name]
        
        st.subheader(f"{s_name} ({s_ticker})")
        
        # Row 1: Breadth & High/Low (3 Cols)
        c1, c2, c3 = st.columns(3)
        
        # 1. Net New Highs/Lows (60 Days)
        df_net = ds.get_sector_high_low_data(s_name, days=60)
        
        with c1:
            if not df_net.empty:
                import plotly.graph_objects as go
                fig1 = go.Figure()
                colors = ['green' if x >= 0 else 'red' for x in df_net['Net']]
                fig1.add_trace(go.Bar(x=df_net.index, y=df_net['Net'], marker_color=colors))
                fig1.update_layout(title="Net New Highs/Lows", height=300, margin=dict(l=20, r=20, t=40, b=20), showlegend=False)
                # Remove gaps
                dt_all = pd.date_range(start=df_net.index.min(), end=df_net.index.max())
                dt_breaks = dt_all.difference(df_net.index)
                fig1.update_xaxes(rangebreaks=[dict(values=dt_breaks)])
                st.plotly_chart(fig1, use_container_width=True)
            else:
                st.caption("No High/Low Data")
                
        # 2. % > MA20 (60 Days)
        df_ma20 = ds.get_breadth_history(s_name, 'pct_above_ma20', days=60)
        
        with c2:
            if not df_ma20.empty:
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(x=df_ma20.index, y=df_ma20['Value'], mode='lines', line=dict(color='blue')))
                fig2.update_layout(title="% > MA20", height=300, yaxis=dict(range=[0, 100]), margin=dict(l=20, r=20, t=40, b=20), showlegend=False)
                # Remove gaps
                dt_all = pd.date_range(start=df_ma20.index.min(), end=df_ma20.index.max())
                dt_breaks = dt_all.difference(df_ma20.index)
                fig2.update_xaxes(rangebreaks=[dict(values=dt_breaks)])
                
                fig2.add_hline(y=50, line_dash="dot", line_color="gray")
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.caption("No Breadth Data")

        # 3. % > MA50 (60 Days)
        df_ma50 = ds.get_breadth_history(s_name, 'pct_above_ma50', days=60)
        
        with c3:
            if not df_ma50.empty:
                fig3 = go.Figure()
                fig3.add_trace(go.Scatter(x=df_ma50.index, y=df_ma50['Value'], mode='lines', line=dict(color='purple')))
                fig3.update_layout(title="% > MA50", height=300, yaxis=dict(range=[0, 100]), margin=dict(l=20, r=20, t=40, b=20), showlegend=False)
                # Remove gaps
                dt_all = pd.date_range(start=df_ma50.index.min(), end=df_ma50.index.max())
                dt_breaks = dt_all.difference(df_ma50.index)
                fig3.update_xaxes(rangebreaks=[dict(values=dt_breaks)])
                
                fig3.add_hline(y=50, line_dash="dot", line_color="gray")
                st.plotly_chart(fig3, use_container_width=True)
            else:
                st.caption("No Breadth Data")

        # Row 2: Momentum (2 Cols)
        c4, c5 = st.columns(2)

        # 4. Momentum Score (60 Days)
        # Get slightly more history to calculate change comfortably
        df_mom = ds.get_momentum_history(s_ticker, period_days=70)
        
        if not df_mom.empty:
            # Prepare data
            df_mom['Diff'] = df_mom['Score'].diff()
            df_mom['MA5_Diff'] = df_mom['Diff'].rolling(window=5).mean()
            
            # Slice to last 60 days for display
            df_mom_disp = df_mom.tail(60)
            
            # Chart 4: Absolute Score
            with c4:
                fig4 = go.Figure()
                # Colored background zones
                min_y = min(df_mom_disp['Score'].min(), -2)
                max_y = max(df_mom_disp['Score'].max(), 2)
                
                # Green Zone (>1)
                fig4.add_shape(type="rect", x0=df_mom_disp.index.min(), x1=df_mom_disp.index.max(), y0=1, y1=max_y, fillcolor="green", opacity=0.1, layer="below", line_width=0)
                # Red Zone (<1)
                fig4.add_shape(type="rect", x0=df_mom_disp.index.min(), x1=df_mom_disp.index.max(), y0=min_y, y1=1, fillcolor="red", opacity=0.1, layer="below", line_width=0)
                
                fig4.add_trace(go.Scatter(x=df_mom_disp.index, y=df_mom_disp['Score'], mode='lines', line=dict(color='black')))
                fig4.update_layout(title="Momentum Score", height=300, margin=dict(l=20, r=20, t=40, b=20), showlegend=False)
                # Remove gaps
                dt_all = pd.date_range(start=df_mom_disp.index.min(), end=df_mom_disp.index.max())
                dt_breaks = dt_all.difference(df_mom_disp.index)
                fig4.update_xaxes(rangebreaks=[dict(values=dt_breaks)])
                st.plotly_chart(fig4, use_container_width=True)
                
            # Chart 5: Momentum Change
            with c5:
                fig5 = go.Figure()
                colors_diff = ['green' if x >= 0 else 'red' for x in df_mom_disp['Diff']]
                
                # Bar
                fig5.add_trace(go.Bar(x=df_mom_disp.index, y=df_mom_disp['Diff'], name='Change', marker_color=colors_diff, opacity=0.6))
                # Line
                fig5.add_trace(go.Scatter(x=df_mom_disp.index, y=df_mom_disp['MA5_Diff'], name='MA5', line=dict(color='blue', width=2)))
                
                fig5.update_layout(title="Momentum Change", height=300, margin=dict(l=20, r=20, t=40, b=20), showlegend=True,
                                   legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                # Remove gaps
                fig5.update_xaxes(rangebreaks=[dict(values=dt_breaks)])
                st.plotly_chart(fig5, use_container_width=True)
        else:
            with c4: st.caption("No Momentum Data")
            with c5: st.caption("No Momentum Data")
        
        st.divider()

elif page == "Sector Stocks":
    st.header("Sector Constituents")
    st.markdown("View the list of stocks (tickers) belonging to each sector.")
    
    # Sector Selection
    sector_opts = ds.get_sector_tickers(weight_type='cap')
    sector_names = sorted(list(sector_opts.keys()))
    
    selected_sector_stocks = st.selectbox("Select Sector", sector_names, key='stocks_sector')
    
    # Get constituents
    tickers = ds.get_sector_constituents(selected_sector_stocks)
    count = len(tickers)
    
    st.subheader(f"{selected_sector_stocks}")
    st.markdown(f"**Total Stocks:** {count}")
    
    if count > 0:
        # Display as a clean list or dataframe?
        with st.expander("View Ticker List", expanded=True):
            st.code(", ".join(tickers), language=None)
            
        # Also a table for easy reading
        import pandas as pd
        df_tickers = pd.DataFrame(tickers, columns=["Ticker"])
        st.dataframe(df_tickers, height=400, use_container_width=False)
        
    else:
        st.info("No constituents found for this sector.")

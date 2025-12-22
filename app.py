import streamlit as st
import plotly.express as px
import data_service as ds
import pandas as pd

st.set_page_config(page_title="US Market Sector Monitor", layout="wide")

st.title("US Market Sector Monitor")

# Sidebar




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
# Navigation
if "current_view" not in st.session_state:
    st.session_state.current_view = "Overview"

opts_etf = ["Overview", "Performance Matrix", "Momentum Ranking", "Momentum Score charts"]
opts_stock = ["Sector Charts", "Market Breadth", "New Highs / Lows", "Sector Stocks"]

# Determine indices based on current state
idx_etf = opts_etf.index(st.session_state.current_view) if st.session_state.current_view in opts_etf else None
idx_stock = opts_stock.index(st.session_state.current_view) if st.session_state.current_view in opts_stock else None

st.sidebar.subheader("ETF Analysis")
sel_etf = st.sidebar.radio("ETF Analysis", opts_etf, index=idx_etf, key="radio_etf", label_visibility="collapsed")

st.sidebar.markdown("---")
st.sidebar.subheader("Stock Analysis")
sel_stock = st.sidebar.radio("Stock Analysis", opts_stock, index=idx_stock, key="radio_stock", label_visibility="collapsed")

# Logic to update state
new_view = st.session_state.current_view
# Check if ETF radio triggered a change (and is a valid selection)
if sel_etf and sel_etf != st.session_state.current_view and sel_etf in opts_etf:
    new_view = sel_etf

# Check if Stock radio triggered a change
if sel_stock and sel_stock != st.session_state.current_view and sel_stock in opts_stock:
    new_view = sel_stock

if new_view != st.session_state.current_view:
    st.session_state.current_view = new_view
    st.rerun()

page = st.session_state.current_view

if page == "Overview":
    try:
        with st.spinner('Loading data from database...'):
            # Period Selector
            st.write("### Settings")
            period_options = {
                "30 Days": 30,
                "60 Days": 60,
                "120 Days": 120,
                "240 Days": 240,
                "365 Days (1Y)": 365
            }
            selected_period_key = st.radio(
                "Select Period:", 
                options=list(period_options.keys()), 
                index=0, # Default to 30 Days
                horizontal=True,
                key="overview_period_selector"
            )
            days = period_options[selected_period_key]
            selected_period_label = selected_period_key # For chart titles

            # Define types to show
            overview_types = [("Cap Weighted", "cap"), ("Equal Weighted", "equal")]
            
            # Create columns for side-by-side display
            col1, col2 = st.columns(2)
            cols = [col1, col2]
            
            for idx, (label, w_t) in enumerate(overview_types):
                with cols[idx]:
                    df_prices, sector_map = load_data(days, w_t)
                    
                    if df_prices.empty:
                        st.warning(f"No data found for {label}. Please click 'Update Database' in the sidebar.")
                    else:
                        # Performance over the selected period
                        perf = (df_prices.iloc[-1] / df_prices.iloc[0] - 1) * 100
                        perf = perf.sort_values(ascending=False)
                        
                        # Create a DataFrame for the chart
                        perf_df = pd.DataFrame({'Ticker': perf.index, 'Return (%)': perf.values})
                        
                        # Map Ticker to Name
                        ticker_to_name = {v: k for k, v in sector_map.items()}
                        perf_df['Sector'] = perf_df['Ticker'].map(ticker_to_name)
                        
                        st.subheader(f"{label}")
                        
                        fig = px.bar(
                            perf_df, 
                            x='Sector', 
                            y='Return (%)', 
                            text='Return (%)',
                            color='Return (%)',
                            color_continuous_scale='RdYlGn',
                            title=f"Relative Performance (Last {days} days)"
                        )
                        fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
                        # For vertical bars, we want the highest return on the left typically, or just sorted.
                        # Since we sorted `perf` descending, the DF is sorted. 
                        # Plotly defaults to plotting in order of data or categorical. 
                        # 'total descending' ensures visual sort if index didn't match.
                        fig.update_layout(xaxis={'categoryorder':'total descending'}, height=500)
                        st.plotly_chart(fig, use_container_width=True)

            st.divider()
            
    except Exception as e:
        st.error(f"Error processing data: {e}")

elif page == "Performance Matrix":
    st.header("Performance Matrix (Returns %)")
    
    types_to_show = [("Cap Weighted", "cap"), ("Equal Weighted", "equal")]
    
    for label, w_t in types_to_show:
        st.subheader(f"{label}")
        try:
            df_matrix = load_matrix(w_t)
            if df_matrix.empty:
                st.warning(f"Not enough data to calculate matrix for {label}.")
            else:
                # Add Sector Name column for clarity
                sectors = ds.get_sector_tickers(weight_type=w_t)
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
                    height=400,
                    key=f"perf_matrix_{w_t}"
                )
        except Exception as e:
            st.error(f"Error calculating matrix for {label}: {e}")
        
        st.divider()

elif page == "Momentum Ranking":
    st.header("Momentum Ranking")
    st.markdown("""
    **Formula:**
    `Score = 0.25 * Return(5d-1d) + 0.25 * Return(10d-5d) + 0.25 * Return(20d-10d) + 0.25 * Return(40d-20d)`
    """)
    
    types_to_show = [("Cap Weighted", "cap"), ("Equal Weighted", "equal")]
    
    for label, w_t in types_to_show:
        st.subheader(f"{label}")
        try:
            df_mom = ds.get_momentum_ranking(weight_type=w_t)
            if df_mom.empty:
                st.warning(f"Not enough data to calculate momentum for {label}.")
            else:
                # Add Sector Name
                sectors = ds.get_sector_tickers(weight_type=w_t)
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
                    height=500,
                    key=f"mom_rank_{w_t}"
                )
        except Exception as e:
            st.error(f"Error calculating momentum for {label}: {e}")
        st.divider()

    try:
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
            # Build list of default names to check
            default_chk_names = []
            
            # Since we maintain consistency with previous logic, let's just pick one type for defaulting
            # or maybe default to 'cap' for auto-selection logic's sake if needed. 
            # But the user might want empty or persistent.
            # Let's just assume we want Cap Weighted leaders by default if we have to pick.
            chk_default_type = 'cap' 
            try:
                # Recalculate ranking for default check just in case
                df_mom_def = ds.get_momentum_ranking(weight_type=chk_default_type)
                if not df_mom_def.empty:
                    # Get top sector name
                    sectors_def = ds.get_sector_tickers(weight_type=chk_default_type)
                    t2n_def = {v: k for k, v in sectors_def.items()}
                    df_mom_def['Sector'] = df_mom_def.index.map(t2n_def)
                    current_top_sectors = df_mom_def['Sector'].head(3).tolist()
                    
                    for s_name in current_top_sectors:
                        match = next((o for o in all_opts if o['sector'] == s_name and o['type'] == chk_default_type), None)
                        if match:
                            default_chk_names.append(match['name'])
            except:
                pass # Fallback to empty if error

            
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




elif page == "Momentum Score charts":
    st.header("Momentum Score History (Last 60 Days)")
    st.write("Comparing Cap Weighted (Left) vs Equal Weighted (Right) Momentum Scores.")
    
    # Imports
    import plotly.express as px
    
    # Get all sectors
    # We want to iterate through unique sectors and show both types
    # ds.get_sector_tickers('cap') returns {Name: Ticker}
    cap_sectors = ds.get_sector_tickers('cap')
    equal_sectors = ds.get_sector_tickers('equal')
    
    # Sorted list of sector names based on Momentum Ranking (Cap Weighted)
    # Using Cap Weighted as the primary sort key
    df_rank = ds.get_momentum_ranking('cap')
    
    if not df_rank.empty:
        # Create map Ticker -> Name
        ticker_to_name = {v: k for k, v in cap_sectors.items()}
        
        # Get sorted names from ranking
        sorted_names = [ticker_to_name.get(t) for t in df_rank.index if t in ticker_to_name]
        
        # Add any missing sectors (e.g. no data for ranking but exists in config)
        remaining = [s for s in cap_sectors.keys() if s not in sorted_names]
        sector_names = sorted_names + sorted(remaining)
    else:
        # Fallback to alpha if no ranking data
        sector_names = sorted(list(cap_sectors.keys()))
    
    for s_name in sector_names:
        st.subheader(s_name)
        
        # Fetch data for both first to calculate scale
        ticker_cap = cap_sectors.get(s_name)
        df_cap = pd.DataFrame()
        if ticker_cap:
            df_cap = ds.get_momentum_history(ticker_cap, period_days=60)
            
        ticker_equal = equal_sectors.get(s_name)
        df_equal = pd.DataFrame()
        if ticker_equal:
            df_equal = ds.get_momentum_history(ticker_equal, period_days=60)
            
        # Calculate common y-axis range
        y_min = 0
        y_max = 0
        
        all_scores = []
        if not df_cap.empty:
            all_scores.extend(df_cap['Score'].tolist())
        if not df_equal.empty:
            all_scores.extend(df_equal['Score'].tolist())
            
        if all_scores:
            val_min = min(all_scores)
            val_max = max(all_scores)
            
            # Ensure 1 is in range for context
            val_min = min(val_min, 0.95)
            val_max = max(val_max, 1.05)
            
            # Add 5% buffer
            padding = (val_max - val_min) * 0.05
            y_min = val_min - padding
            y_max = val_max + padding
        
        col1, col2 = st.columns(2)
        
        def add_background_regions(figure, y_upper, y_lower):
            # Green (Above 1)
            figure.add_hrect(
                y0=1, y1=y_upper,
                fillcolor="rgba(0, 255, 0, 0.05)",
                line_width=0,
                layer="below"
            )
            # Red (Below 1)
            figure.add_hrect(
                y0=y_lower, y1=1,
                fillcolor="rgba(255, 0, 0, 0.05)",
                line_width=0,
                layer="below"
            )

        # 1. Cap Weighted (Left)
        with col1:
            st.markdown("**Cap Weighted**")
            if ticker_cap:
                if not df_cap.empty:
                    # Plot
                    fig = px.line(df_cap, x=df_cap.index, y='Score', title=f"{ticker_cap} Score")
                    fig.update_layout(height=300, margin=dict(l=20, r=20, t=30, b=20))
                    if all_scores:
                        fig.update_yaxes(range=[y_min, y_max])
                        add_background_regions(fig, y_max, y_min)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info(f"No data for {ticker_cap}")
            else:
                st.warning("Ticker not found")
                
        # 2. Equal Weighted (Right)
        with col2:
            st.markdown("**Equal Weighted**")
            if ticker_equal:
                if not df_equal.empty:
                    # Plot
                    fig = px.line(df_equal, x=df_equal.index, y='Score', title=f"{ticker_equal} Score")
                    fig.update_layout(height=300, margin=dict(l=20, r=20, t=30, b=20))
                    if all_scores:
                        fig.update_yaxes(range=[y_min, y_max])
                        add_background_regions(fig, y_max, y_min)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info(f"No data for {ticker_equal}")
            else:
                st.warning("Ticker not found")
        
        st.divider()

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

        # Get Tickers for both weights
        tickers = ds.SECTORS_CONFIG[s_name]
        s_ticker_cap = tickers['cap']
        s_ticker_equal = tickers['equal']
        
        # Helper to plot momentum
        def plot_momentum_chart(col_obj, ticker, title_suffix):
             df_mom = ds.get_momentum_history(ticker, period_days=60)
             
             with col_obj:
                 if not df_mom.empty:
                    fig = go.Figure()
                    
                    # Colored background zones
                    min_y = min(df_mom['Score'].min(), -2)
                    max_y = max(df_mom['Score'].max(), 2)
                    
                    # Ensure 1 is in range
                    min_y = min(min_y, 0.95)
                    max_y = max(max_y, 1.05)
                    
                    # Green Zone (>1)
                    fig.add_shape(type="rect", x0=df_mom.index.min(), x1=df_mom.index.max(), y0=1, y1=max_y, fillcolor="green", opacity=0.1, layer="below", line_width=0)
                    # Red Zone (<1)
                    fig.add_shape(type="rect", x0=df_mom.index.min(), x1=df_mom.index.max(), y0=min_y, y1=1, fillcolor="red", opacity=0.1, layer="below", line_width=0)
                    
                    fig.add_trace(go.Scatter(x=df_mom.index, y=df_mom['Score'], mode='lines', line=dict(color='black')))
                    fig.update_layout(
                        title=f"Momentum Score {title_suffix}", 
                        height=300, 
                        margin=dict(l=20, r=20, t=40, b=20), 
                        showlegend=False
                    )
                    
                    # Remove gaps
                    dt_all = pd.date_range(start=df_mom.index.min(), end=df_mom.index.max())
                    dt_breaks = dt_all.difference(df_mom.index)
                    fig.update_xaxes(rangebreaks=[dict(values=dt_breaks)])
                    
                    st.plotly_chart(fig, use_container_width=True)
                 else:
                    st.caption(f"No Momentum Data for {ticker}")

        # 4. Momentum Score (Cap Weighted)
        plot_momentum_chart(c4, s_ticker_cap, f"({s_ticker_cap})")

        # 5. Momentum Score (Equal Weighted)
        plot_momentum_chart(c5, s_ticker_equal, f"({s_ticker_equal})")

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

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import data_service as ds
import pandas as pd

st.set_page_config(page_title="US Market Sector Monitor", layout="wide")

st.title("US Market Sector Monitor")

# Sidebar




# Update Section moved to 'Data Management' page under Administrator.

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
opts_breadth = ["Sector Charts", "Market Breadth", "New Highs / Lows", "ATR Strength", "EMA Trend Setup", "Stocks > 25% (84d)", "Análise de Setores"]
opts_individual = ["Stocks Counting"]
opts_admin = ["Data Management"]

# Determine indices based on current state
idx_etf = opts_etf.index(st.session_state.current_view) if st.session_state.current_view in opts_etf else None
idx_breadth = opts_breadth.index(st.session_state.current_view) if st.session_state.current_view in opts_breadth else None
idx_individual = opts_individual.index(st.session_state.current_view) if st.session_state.current_view in opts_individual else None
idx_admin = opts_admin.index(st.session_state.current_view) if st.session_state.current_view in opts_admin else None

st.sidebar.subheader("Sector's ETF's")
sel_etf = st.sidebar.radio("Sector's ETF's", opts_etf, index=idx_etf, key="radio_etf", label_visibility="collapsed")

st.sidebar.markdown("---")
st.sidebar.subheader("Market Breadth")
sel_breadth = st.sidebar.radio("Market Breadth", opts_breadth, index=idx_breadth, key="radio_breadth", label_visibility="collapsed")

st.sidebar.markdown("---")
st.sidebar.subheader("Individual Stocks")
sel_individual = st.sidebar.radio("Individual Stocks", opts_individual, index=idx_individual, key="radio_individual", label_visibility="collapsed")

st.sidebar.markdown("---")
st.sidebar.subheader("Administrator")
sel_admin = st.sidebar.radio("Administrator", opts_admin, index=idx_admin, key="radio_admin", label_visibility="collapsed")

# Logic to update state
new_view = st.session_state.current_view
# Check if ETF radio triggered a change
if sel_etf and sel_etf != st.session_state.current_view and sel_etf in opts_etf:
    new_view = sel_etf

# Check if Breadth radio triggered a change
if sel_breadth and sel_breadth != st.session_state.current_view and sel_breadth in opts_breadth:
    new_view = sel_breadth

# Check if Individual radio triggered a change
if sel_individual and sel_individual != st.session_state.current_view and sel_individual in opts_individual:
    new_view = sel_individual

# Check if Admin radio triggered a change
if sel_admin and sel_admin != st.session_state.current_view and sel_admin in opts_admin:
    new_view = sel_admin

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
                "5 Days": 5,
                "10 Days": 10,
                "20 Days": 20,
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
    `Score = 0.25 * Return(5d-0d) + 0.25 * Return(10d-5d) + 0.25 * Return(20d-10d) + 0.25 * Return(40d-20d)`
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
                cols = ['Sector', 'Last Price', 'Date', 'Score', 'Score -5d', 'Score -20d', 'Score -50d', 'Score Chg (5d)', 'R(5-0)', 'R(10-5)', 'R(20-10)', 'R(40-20)']
                df_display = df_mom[cols]
                
                st.dataframe(
                    df_display.style
                    .format({
                        'Score': fmt_score,
                        'Score Chg (5d)': fmt_score,
                        'Score -5d': fmt_score,
                        'Score -20d': fmt_score,
                        'Score -50d': fmt_score,
                        'R(5-0)': fmt_pct,
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
            if 'df_mom' in locals() and not df_mom.empty:
                 st.write(f"Available columns: {df_mom.columns.tolist()}")
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

    # --- New Chart: % > MA20 vs % > MA50 ---
    st.subheader("Medium vs Long Term Trend")
    st.caption("Comparison of % Stocks > MA20 and % Stocks > MA50")
    
    df_ma20 = ds.get_breadth_data(selected_sector_breadth, metric='pct_above_ma20', days=days_history)
    df_ma50 = ds.get_breadth_data(selected_sector_breadth, metric='pct_above_ma50', days=days_history)
    df_ma5 = ds.get_breadth_data(selected_sector_breadth, metric='pct_above_ma5', days=days_history)
    df_ma10 = ds.get_breadth_data(selected_sector_breadth, metric='pct_above_ma10', days=days_history)
    
    if (df_ma20 is not None and not df_ma20.empty) and (df_ma50 is not None and not df_ma50.empty):
        # Create Subplots with Dual Axis for all charts
        fig_combined = make_subplots(specs=[[{"secondary_y": True}]])
        
        # MA20
        fig_combined.add_trace(go.Scatter(
            x=df_ma20.index, y=df_ma20['Value'],
            mode='lines', name='% > MA20',
            line=dict(color='#ffc658', width=2) # Yellow/Orange
        ), secondary_y=False)
        
        # MA50
        fig_combined.add_trace(go.Scatter(
            x=df_ma50.index, y=df_ma50['Value'],
            mode='lines', name='% > MA50',
            line=dict(color='#ff7300', width=2) # Orange/Red
        ), secondary_y=False)

        # Plot ETF
        if not df_etf.empty:
            fig_combined.add_trace(go.Scatter(
                x=df_etf.index, y=df_etf['Close'],
                name=f"{benchmark_ticker}",
                line=dict(color='gray', width=1, dash='dot'),
                mode='lines', opacity=0.5
            ), secondary_y=True)
            
            fig_combined.update_yaxes(title_text=f"{benchmark_ticker} Price", secondary_y=True, showgrid=False)
        
        fig_combined.update_layout(
            title="Breadth Momentum: MA20 vs MA50",
            height=350,
            margin=dict(l=20, r=20, t=40, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            yaxis=dict(range=[0, 100], title="Percentage (%)")
        )
         # Remove Gaps
        all_dates_c = df_ma20.index.union(df_ma50.index)
        if not df_etf.empty: all_dates_c = all_dates_c.union(df_etf.index)
        dt_all_c = pd.date_range(start=all_dates_c.min(), end=all_dates_c.max())
        dt_breaks_c = dt_all_c.difference(all_dates_c)
        fig_combined.update_xaxes(rangebreaks=[dict(values=dt_breaks_c)])
        
        # 50% Line
        fig_combined.add_hline(y=50, line_dash="dash", line_color="gray", secondary_y=False)
        
        st.plotly_chart(fig_combined, use_container_width=True)
        
        # --- Spread Chart (MA20 - MA50) ---
        # Align indexes
        df_spread = df_ma20.join(df_ma50, lsuffix='_20', rsuffix='_50', how='inner')
        df_spread['diff'] = df_spread['Value_20'] - df_spread['Value_50']
        
        if not df_spread.empty:
            fig_spread = make_subplots(specs=[[{"secondary_y": True}]])
            colors = ['green' if x >= 0 else 'red' for x in df_spread['diff']]
            
            fig_spread.add_trace(go.Bar(
                x=df_spread.index, 
                y=df_spread['diff'],
                marker_color=colors,
                name='Spread (MA20 - MA50)'
            ), secondary_y=False)

            # Plot ETF
            if not df_etf.empty:
                fig_spread.add_trace(go.Scatter(
                    x=df_etf.index, y=df_etf['Close'],
                    name=f"{benchmark_ticker}",
                    line=dict(color='gray', width=1, dash='dot'),
                    mode='lines', opacity=0.5
                ), secondary_y=True)
                fig_spread.update_yaxes(title_text=f"{benchmark_ticker} Price", secondary_y=True, showgrid=False)
            
            fig_spread.update_layout(
                title="Spread: (% > MA20) - (% > MA50)",
                height=250, # Smaller height
                margin=dict(l=20, r=20, t=30, b=20),
                yaxis=dict(title="Delta (%)"),
                showlegend=False
            )
            # Remove Gaps matching the chart above
            fig_spread.update_xaxes(rangebreaks=[dict(values=dt_breaks_c)])
            
            st.plotly_chart(fig_spread, use_container_width=True)
            
        # --- Spread Chart (MA10 - MA20) ---
        if (df_ma10 is not None and not df_ma10.empty) and (df_ma20 is not None and not df_ma20.empty):
            df_spread_10_20 = df_ma10.join(df_ma20, lsuffix='_10', rsuffix='_20', how='inner')
            df_spread_10_20['diff'] = df_spread_10_20['Value_10'] - df_spread_10_20['Value_20']
            
            fig_spread_10_20 = make_subplots(specs=[[{"secondary_y": True}]])
            colors_10_20 = ['green' if x >= 0 else 'red' for x in df_spread_10_20['diff']]
            
            fig_spread_10_20.add_trace(go.Bar(
                x=df_spread_10_20.index, 
                y=df_spread_10_20['diff'],
                marker_color=colors_10_20,
                name='Spread (MA10 - MA20)'
            ), secondary_y=False)

            # Plot ETF
            if not df_etf.empty:
                fig_spread_10_20.add_trace(go.Scatter(
                    x=df_etf.index, y=df_etf['Close'],
                    name=f"{benchmark_ticker}",
                    line=dict(color='gray', width=1, dash='dot'),
                    mode='lines', opacity=0.5
                ), secondary_y=True)
                fig_spread_10_20.update_yaxes(title_text=f"{benchmark_ticker} Price", secondary_y=True, showgrid=False)
            
            fig_spread_10_20.update_layout(
                title="Spread: (% > MA10) - (% > MA20)",
                height=250,
                margin=dict(l=20, r=20, t=30, b=20),
                yaxis=dict(title="Delta (%)"),
                showlegend=False
            )
            # Remove Gaps
            all_dates_10_20 = df_ma10.index.union(df_ma20.index)
            if not df_etf.empty: all_dates_10_20 = all_dates_10_20.union(df_etf.index)
            dt_all_10_20 = pd.date_range(start=all_dates_10_20.min(), end=all_dates_10_20.max())
            dt_breaks_10_20 = dt_all_10_20.difference(all_dates_10_20)
            fig_spread_10_20.update_xaxes(rangebreaks=[dict(values=dt_breaks_10_20)])
            
            st.plotly_chart(fig_spread_10_20, use_container_width=True)

        # --- Spread Chart (MA5 - MA10) ---
        if (df_ma5 is not None and not df_ma5.empty) and (df_ma10 is not None and not df_ma10.empty):
            df_spread_5_10 = df_ma5.join(df_ma10, lsuffix='_5', rsuffix='_10', how='inner')
            df_spread_5_10['diff'] = df_spread_5_10['Value_5'] - df_spread_5_10['Value_10']
            
            fig_spread_5_10 = make_subplots(specs=[[{"secondary_y": True}]])
            colors_5_10 = ['green' if x >= 0 else 'red' for x in df_spread_5_10['diff']]
            
            fig_spread_5_10.add_trace(go.Bar(
                x=df_spread_5_10.index, 
                y=df_spread_5_10['diff'],
                marker_color=colors_5_10,
                name='Spread (MA5 - MA10)'
            ), secondary_y=False)

            # Plot ETF
            if not df_etf.empty:
                fig_spread_5_10.add_trace(go.Scatter(
                    x=df_etf.index, y=df_etf['Close'],
                    name=f"{benchmark_ticker}",
                    line=dict(color='gray', width=1, dash='dot'),
                    mode='lines', opacity=0.5
                ), secondary_y=True)
                fig_spread_5_10.update_yaxes(title_text=f"{benchmark_ticker} Price", secondary_y=True, showgrid=False)
            
            fig_spread_5_10.update_layout(
                title="Spread: (% > MA5) - (% > MA10)",
                height=250,
                margin=dict(l=20, r=20, t=30, b=20),
                yaxis=dict(title="Delta (%)"),
                showlegend=False
            )
            # Remove Gaps - reuse dates
            all_dates_5_10 = df_ma5.index.union(df_ma10.index)
            if not df_etf.empty: all_dates_5_10 = all_dates_5_10.union(df_etf.index)
            dt_all_5_10 = pd.date_range(start=all_dates_5_10.min(), end=all_dates_5_10.max())
            dt_breaks_5_10 = dt_all_5_10.difference(all_dates_5_10)
            fig_spread_5_10.update_xaxes(rangebreaks=[dict(values=dt_breaks_5_10)])
            
            st.plotly_chart(fig_spread_5_10, use_container_width=True)

        st.divider()

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

elif page == "Stocks Counting":
    st.header("Sector Constituents")
    st.markdown("View the list of stocks (tickers) belonging to each sector.")
    
    # Get all sector names
    sector_opts = ds.get_sector_tickers(weight_type='cap')
    sector_names = sorted(list(sector_opts.keys()))
    
    # Iterate through all sectors
    for s_name in sector_names:
        # Get constituents
        tickers = ds.get_sector_constituents(s_name)
        count = len(tickers)
        
        st.subheader(f"{s_name} (Current: {count})")
        
        # Plot Active Count History
        df_active = ds.get_active_constituent_history(s_name)
        if not df_active.empty:
            import plotly.express as px
            fig = px.area(df_active, x=df_active.index, y='Count', title="Active Stocks Over Time")
            fig.update_layout(height=250, margin=dict(l=20, r=20, t=30, b=20), showlegend=False)
            dt_all = pd.date_range(start=df_active.index.min(), end=df_active.index.max())
            dt_breaks = dt_all.difference(df_active.index)
            fig.update_xaxes(rangebreaks=[dict(values=dt_breaks)])
            st.plotly_chart(fig, use_container_width=True)
            
        if count > 0:
            st.code(", ".join(tickers), language=None)
        else:
            st.info("No constituents found.")
        
        st.divider()

elif page == "Stocks > 25% (84d)":
    st.header("Stocks > 25% Increase (Last 84 Days)")
    st.markdown("Number of stocks in each sector that have risen by 25% or more over a rolling 84-day period.")
    
    # History Slider
    history_days = st.slider("History (Days)", min_value=90, max_value=3650, value=365, step=30, key='up25_slider')
    
    # Get all sectors
    sector_opts = ds.get_sector_tickers(weight_type='cap')
    sector_names = sorted(list(sector_opts.keys()))
    
    import plotly.express as px
    
    with st.spinner("Calculating historical data..."):
        for s_name in sector_names:
            st.markdown(f"### {s_name}")
            
            # Metric Definitions: (Threshold Label, Metric Name, Plot Color Logic)
            # Logic: Red if <= 10th percentile, else Blue
            metrics = [
                ("> 25%", "pct_up_25_84d"),
                ("> 50%", "pct_up_50_84d"),
                ("> 100%", "pct_up_100_84d")
            ]
            
            for label, metric_key in metrics:
                # 3. Get Data (now cached)
                # Using updated signature: get_stocks_up_history(sector_name, metric_name, days_history)
                df_up = ds.get_stocks_up_history(s_name, metric_name=metric_key, days_history=history_days)
                
                if not df_up.empty:
                    # Calculate 10th percentile
                    quantile_10 = df_up['Percent'].quantile(0.10)
                    
                    # Assign colors
                    df_up['Condition'] = df_up['Percent'].apply(lambda x: 'Low (<=10%)' if x <= quantile_10 else 'Normal')
                    
                    # Plot
                    fig = px.bar(
                        df_up, 
                        x=df_up.index, 
                        y='Percent', 
                        title=f"{s_name} - Stocks {label}",
                        color='Condition',
                        color_discrete_map={'Low (<=10%)': 'red', 'Normal': '#1f77b4'}
                    )
                    fig.update_layout(height=250, margin=dict(l=20, r=20, t=30, b=20), yaxis_title="% Stocks", showlegend=True)
                    
                    # Remove gaps
                    dt_all = pd.date_range(start=df_up.index.min(), end=df_up.index.max())
                    dt_breaks = dt_all.difference(df_up.index)
                    fig.update_xaxes(rangebreaks=[dict(values=dt_breaks)])
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info(f"No data available for {s_name} ({label})")
            
            st.divider()

elif page == "Análise de Setores":
    st.header("Análise de Setores")
    st.write("Identifique a distribuição setorial da sua lista de ações.")

    # Input Area
    raw_input = st.text_area("Insira os Tickers (separados por vírgula, espaço ou nova linha):", height=150, placeholder="AAPL, MSFT\nGOOG XLE")
    
    if st.button("Analisar Carteira"):
        if not raw_input.strip():
            st.warning("Por favor, insira pelo menos um ticker.")
        else:
            # Parse using Regex to handle multiple separators
            import re
            # Split by comma, newline, or whitespace
            tickers = re.split(r'[,\s\n]+', raw_input)
            # Clean and filter empty strings
            tickers = [t.strip().upper() for t in tickers if t.strip()]
            
            if not tickers:
                st.warning("Nenhum ticker válido encontrado.")
            else:
                with st.spinner("Analisando setores..."):
                    # Backend Calls
                    sector_map = ds.get_sectors_for_tickers(tickers)
                    sector_totals = ds.get_sector_counts()
                    

                    # Identify unknowns initially
                    unknowns = [t for t in tickers if t not in sector_map]
                    
                    if unknowns:
                        st.info(f"Buscando {len(unknowns)} tickers desconhecidos no Yahoo Finance...")
                        yahoo_map = ds.fetch_sector_from_yahoo(unknowns)
                        
                        # Merge results
                        if yahoo_map:
                            sector_map.update(yahoo_map)
                            st.success(f"Encontrados {len(yahoo_map)} setores via Yahoo Finance.")
                        
                        # Re-evaluate unknowns
                        unknowns = [t for t in tickers if t not in sector_map]

                    # 1. Process User Data
                    user_counts = {}
                    for t in tickers:
                        if t in sector_map:
                            s = sector_map[t]
                            user_counts[s] = user_counts.get(s, 0) + 1
                            
                    if unknowns:
                        st.warning(f"Tickers não encontrados na base ou Yahoo: {', '.join(unknowns)}")
                    
                    if not user_counts:
                         st.error("Nenhum setor identificado para os tickers fornecidos.")
                    else:
                        # 2. Calculate Ratios
                        data = []
                        for sector, u_count in user_counts.items():
                            app_count = sector_totals.get(sector, 0)
                            
                            ratio = 0
                            if app_count > 0:
                                ratio = u_count / app_count
                                
                            data.append({
                                'Setor': sector,
                                'Seu Contagem': u_count,
                                'Total no App': app_count,
                                'Razão': ratio,
                                'Razão (%)': ratio * 100
                            })
                            
                        # Create DF
                        df = pd.DataFrame(data)
                        df = df.sort_values('Razão', ascending=False)
                        
                        st.divider()
                        st.subheader("Representatividade Setorial")
                        
                        # Chart
                        fig = px.bar(
                            df, 
                            x='Setor', 
                            y='Razão (%)',
                            text='Razão (%)',
                            title="Densidade da Carteira por Setor (Sua Qtde / Total Monitorado)",
                            labels={'Razão (%)': 'Razão (%)'},
                            color='Razão (%)',
                            color_continuous_scale='Viridis'
                        )
                        fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
                        fig.update_layout(height=500)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Dataframe
                        st.markdown("### Detalhes")
                        st.dataframe(df, use_container_width=True)

elif page == "ATR Strength":
    st.header("ATR Strength Panel (Bullish)")
    st.markdown("---")
    
    # Date Selection
    latest_date = ds.get_latest_data_date()
    # Ideally allow date selection, but sticking to latest for now
    st.info(f"Showing data for: **{latest_date}**")
    
    # Fetch Data
    with st.spinner("Calculating ATR Stats..."):
        df_atr = ds.get_atr_variation_stats(target_date=latest_date)
        
    if df_atr.empty:
        st.warning("No data found for the selected date.")
    else:
        # Filter for logic
        # Count total stocks per sector
        total_counts_sector = df_atr.groupby('Sector')['ticker'].count().reset_index(name='TotalCount')
        total_counts_industry = df_atr.groupby(['Sector', 'Industry'])['ticker'].count().reset_index(name='TotalCount')
        
        # Filter: Stocks > 1 ATR (Implicitly Positive based on new logic)
        df_qualified = df_atr[df_atr['is_above_atr']].copy()
        
        # --- Sector Aggregation ---
        # Count qualified
        qual_counts_sector = df_qualified.groupby('Sector')['ticker'].count().reset_index(name='QualifiedCount')
        # Mean Signal Strength
        qual_strength_sector = df_qualified.groupby('Sector')['signal_strength'].mean().reset_index(name='MeanStrength')
        
        # Merge Sector
        df_sector_agg = pd.merge(total_counts_sector, qual_counts_sector, on='Sector', how='left').fillna(0)
        df_sector_agg = pd.merge(df_sector_agg, qual_strength_sector, on='Sector', how='left') # MeanStrength can be NaN if 0 qualified
        
        df_sector_agg['PctQualified'] = (df_sector_agg['QualifiedCount'] / df_sector_agg['TotalCount']) * 100
        # Fill NaN Strength with 0 or Neutral? Let's say 0 but handle color scale
        df_sector_agg['MeanStrength'] = df_sector_agg['MeanStrength'].fillna(0)
        
        # --- Industry Aggregation ---
        qual_counts_ind = df_qualified.groupby(['Sector', 'Industry'])['ticker'].count().reset_index(name='QualifiedCount')
        qual_strength_ind = df_qualified.groupby(['Sector', 'Industry'])['signal_strength'].mean().reset_index(name='MeanStrength')
        
        df_ind_agg = pd.merge(total_counts_industry, qual_counts_ind, on=['Sector', 'Industry'], how='left').fillna(0)
        df_ind_agg = pd.merge(df_ind_agg, qual_strength_ind, on=['Sector', 'Industry'], how='left')
        
        df_ind_agg['PctQualified'] = (df_ind_agg['QualifiedCount'] / df_ind_agg['TotalCount']) * 100
        df_ind_agg['MeanStrength'] = df_ind_agg['MeanStrength'].fillna(0)
        
        # --- Visualization 1: Sector Treemap ---
        st.subheader("Sector Bullish Strength")
        st.caption("Size = % Stocks > 0.7 ATR (Positive) | Color = Mean Strength (Gain / ATR)")
        
        fig_sec = px.treemap(
            df_sector_agg,
            path=['Sector'],
            values='PctQualified', # Size
            color='MeanStrength',  # Color
            # User custom scale: Blue (Low) -> Green (High)
            color_continuous_scale=[
                '#0000ff', '#0040ff', '#0080ff', '#00c0ff', '#00ffff', 
                '#00ffc0', '#00ff80', '#00ff40', '#00ff00'
            ],
            title='Sectors by Volatility Intensity',
            hover_data=['TotalCount', 'QualifiedCount', 'MeanStrength'],
            custom_data=['QualifiedCount', 'PctQualified']
        )
        fig_sec.update_traces(texttemplate='%{label}<br>(%{customdata[0]}) %{customdata[1]:.1f}%') # Add Count and Pct to Label

        # Fix: If PctQualified is 0, Treemap might hide it.
        # But we want to show it. Treemap values must be positive.
        # If 0, maybe set small epsilon? Or user accepts they disappear?
        # User requested "Size proportional to percentage". If 0%, size 0, disappears. Correct.
        
        st.plotly_chart(fig_sec, use_container_width=True)
        
        st.divider()
        
        # --- Visualization 2: Industry Treemap ---
        st.divider()
        
        # --- Visualization 2: Industry Treemaps per Sector ---
        st.subheader("Industry Strength by Sector")
        st.caption("Detailed view per Sector. Color scale is relative to the specific Sector.")
        
        # Get unique sectors
        sectors = df_ind_agg['Sector'].unique()
        sectors.sort()
        
        # Create grid layout (e.g., 2 columns)
        cols = st.columns(2)
        
        for i, sector in enumerate(sectors):
            df_sec_ind = df_ind_agg[df_ind_agg['Sector'] == sector].copy()
            
            # Skip if no data
            if df_sec_ind.empty or df_sec_ind['PctQualified'].sum() == 0:
                continue
                
            fig = px.treemap(
                df_sec_ind,
                path=['Industry'],
                values='PctQualified',
                color='MeanStrength',
                # User custom scale: Blue (Low) -> Green (High)
                color_continuous_scale=[
                    '#0000ff', '#0040ff', '#0080ff', '#00c0ff', '#00ffff', 
                    '#00ffc0', '#00ff80', '#00ff40', '#00ff00'
                ],
                title=f"{sector}",
                hover_data=['TotalCount', 'QualifiedCount', 'MeanStrength'],
                custom_data=['QualifiedCount', 'PctQualified']
            )
            fig.update_traces(texttemplate='%{label}<br>(%{customdata[0]}) %{customdata[1]:.1f}%') # Add Count and Pct to Label
            

            with cols[i % 2]:
                st.plotly_chart(fig, use_container_width=True)


elif page == "EMA Trend Setup":

    st.header("Sector Trends: EMA Bullish Setup")
    st.caption("Stocks meeting criteria: EMA8 > EMA20 > EMA50 and Close > EMA20")

    # History Slider
    days_history = st.slider("History (Days)", 30, 1825, 365, key="ema_setup_slider")
    
    # Grid Layout - REMOVED for Full Width
    # cols = st.columns(3) 
    
    # Get all sectors
    sector_opts = ds.get_sector_tickers(weight_type='cap')
    
    for idx, (s_name, s_ticker) in enumerate(sorted(sector_opts.items())):
        # col = cols[idx % 3]
        
        # with col:
        st.subheader(f"{s_name} ({s_ticker})")
        
        # 1. Broad Breadth Metric (Numerator)
        df_setup = ds.get_breadth_data(s_name, metric='ema_trend_setup', days=days_history)
        
        # 2. Active Count (Denominator)
        df_active = ds.get_breadth_data(s_name, metric='active_count', days=days_history)
        
        # Fetch ETF for overlay
        df_etf = ds.get_etf_price_history(s_name, days=days_history, weight_type='cap')
        
        if df_setup is not None and not df_setup.empty and df_active is not None and not df_active.empty:
            # Align and Calculate Percentage
            df_chart = df_setup.join(df_active, lsuffix='_setup', rsuffix='_total', how='inner')
            df_chart['pct'] = (df_chart['Value_setup'] / df_chart['Value_total']) * 100
            
            # Create chart
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            # Line chart for Setup Percentage
            fig.add_trace(go.Scatter(
                x=df_chart.index, 
                y=df_chart['pct'], 
                name='% Stocks',
                mode='lines',
                line=dict(width=2, color='#00C49F') # Teal/Greenish
            ), secondary_y=False)
            
            # ETF Price Overlay
            if not df_etf.empty:
                fig.add_trace(go.Scatter(
                    x=df_etf.index,
                    y=df_etf['Close'],
                    name=f"{s_ticker}",
                    line=dict(color='gray', width=1, dash='dot'),
                    mode='lines', opacity=0.5
                ), secondary_y=True)
            
            fig.update_layout(
                title=dict(text=f"{s_name}", font=dict(size=14)),
                height=400, 
                margin=dict(l=40, r=40, t=40, b=40),
                showlegend=True,
                xaxis=dict(showticklabels=True),
                yaxis=dict(title="% Stocks", tickformat=".1f") # Format as percentage
            )
            
            # Remove Gaps
            all_dates = df_chart.index
            if not df_etf.empty: all_dates = all_dates.union(df_etf.index)
            dt_all = pd.date_range(start=all_dates.min(), end=all_dates.max())
            dt_breaks = dt_all.difference(all_dates)
            fig.update_xaxes(rangebreaks=[dict(values=dt_breaks)])
            
            fig.update_yaxes(showgrid=True, secondary_y=False) # Show grid for percentage
            fig.update_yaxes(showgrid=False, showticklabels=False, secondary_y=True) 
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning(f"No data for {s_name}")


elif page == "Data Management":
    st.header("Data Management & Updates")
    st.markdown("---")
    
    col_info, col_status = st.columns(2)
    
    with col_info:
        st.subheader("System Status")
        # Show last update date
        try:
            latest_date = ds.get_latest_data_date()
            st.info(f"**Last Data Available:** {latest_date}")
        except:
            st.warning("Last Data: Unknown")
            
    # Progress Bar Container in main area
    st.subheader("Update Progress")
    progress_bar = st.progress(0)
    status_text = st.empty()

    def update_progress(msg, val):
        status_text.text(msg)
        progress_bar.progress(val)

    st.divider()

    # Sector Selection
    st.subheader("Update Configuration")
    sector_options = list(ds.get_sector_tickers('cap').keys())
    
    # Use session state to persist selection if needed, but standard key works
    selected_sectors_update = st.multiselect(
        "Select Sectors to Update:", 
        options=sector_options, 
        default=sector_options,
        key="admin_sector_select"
    )

    st.markdown("### Actions")
    
    tab1, tab2 = st.tabs(["⚡ Quick Update", "🔄 Full Reset"])
    
    with tab1:
        st.markdown("**Quick Check (Gap Fill):** Updates ETFs (1mo) and Constituents (last 7 days). Use this for daily updates.")
        if st.button("🚀 Start Quick Update", type="primary"):
            with st.spinner("Updating recent data..."):
                try:
                    latest = ds.get_latest_data_date()
                    if latest:
                        # Update ETFs first
                        ds.update_sector_data(period="1mo")
                        # Update Constituents
                        safe_start = latest - pd.Timedelta(days=7)
                        
                        if not selected_sectors_update:
                            st.warning("Please select at least one sector.")
                        else:
                            ds.update_constituents_data(
                                sector_name=selected_sectors_update, 
                                start_date=safe_start, 
                                progress_callback=update_progress
                            )
                            st.success("Quick update completed successfully!")
                            st.cache_data.clear()
                    else:
                        st.warning("Database seems empty. Please run 'Full Reset' first.")
                except Exception as e:
                    st.error(f"Error during update: {e}")

    with tab2:
        st.markdown("**Full Reset (Slow):** Re-downloads 10 years of data for ETFs and full history for chosen sectors. **This can take a long time.**")
        if st.button("⚠️ Start Full Reset", type="secondary"):
            with st.spinner("Reloading ALL history (Get some coffee)..."):
                try:
                    if not selected_sectors_update:
                        st.warning("Please select at least one sector.")
                    else:
                        # Update ETFs
                        update_progress("Updating ETFs...", 0.05)
                        ds.update_sector_data(period="10y") 
                        
                        # Update Constituents
                        ds.update_constituents_data(
                            sector_name=selected_sectors_update, 
                            progress_callback=update_progress
                        )
                        
                        st.success("Database fully reset and updated!")
                        st.cache_data.clear() 
                except Exception as e:
                    st.error(f"Full reset failed: {e}")

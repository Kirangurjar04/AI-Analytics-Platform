import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle

import plotly.express as px
import plotly.graph_objects as go



# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI-Powered E-Commerce Analytics Platform",
    page_icon="🛒",
    layout="wide"
)


# ============================================================
# PROJECT PATH
# ============================================================

BASE_PATH = r"C:\Users\Dell\Desktop\Analytics Platform"

DATA_PATH = os.path.join(
    BASE_PATH,
    "ecommerce_features.csv"
)


# ============================================================
# LOAD MAIN DATASET
# ============================================================

try:

    df = pd.read_csv(DATA_PATH)

except Exception as e:

    st.error(f"Unable to load ecommerce_features.csv: {e}")
    st.stop()


# ============================================================
# CLEAN COLUMN NAMES
# ============================================================

df.columns = (
    df.columns
    .astype(str)
    .str.strip()
)


# ============================================================
# DATE COLUMN
# ============================================================

if "Order Date" not in df.columns:

    st.error(
        "The column 'Order Date' was not found in ecommerce_features.csv."
    )

    st.write("Available columns:", list(df.columns))

    st.stop()


df["Order Date"] = pd.to_datetime(
    df["Order Date"],
    errors="coerce"
)


# Remove rows where date is missing

df = df.dropna(
    subset=["Order Date"]
).copy()


# ============================================================
# REQUIRED NUMERIC COLUMNS
# ============================================================

if "Sales/Revenue (INR)" in df.columns:

    df["Sales/Revenue (INR)"] = pd.to_numeric(
        df["Sales/Revenue (INR)"],
        errors="coerce"
    )


# ============================================================
# TITLE IN SIDEBAR
# ============================================================

st.sidebar.title("Dashboard Controls")


# ============================================================
# CATEGORY FILTER
# ============================================================

if "Product Category" in df.columns:

    categories = sorted(
        df["Product Category"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

else:

    categories = []


selected_categories = st.sidebar.multiselect(
    "Select Category",
    categories,
    default=categories
)


filtered_df = df.copy()


if selected_categories:

    filtered_df = filtered_df[
        filtered_df["Product Category"].isin(
            selected_categories
        )
    ].copy()


# ============================================================
# DATE FILTER
# ============================================================

if not filtered_df.empty:

    min_date = filtered_df["Order Date"].min().date()
    max_date = filtered_df["Order Date"].max().date()

else:

    min_date = df["Order Date"].min().date()
    max_date = df["Order Date"].max().date()


selected_dates = st.sidebar.date_input(
    "Select Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)


# Date range is selected only when two dates exist

if isinstance(selected_dates, tuple) and len(selected_dates) == 2:

    start_date, end_date = selected_dates

    filtered_df = filtered_df[
        (filtered_df["Order Date"].dt.date >= start_date)
        &
        (filtered_df["Order Date"].dt.date <= end_date)
    ].copy()


# ============================================================
# BASIC SAFETY CHECK
# ============================================================

if filtered_df.empty:

    st.warning(
        "No data is available for the selected filters."
    )

    st.stop()

# ============================================================
# KPI CALCULATIONS
# ============================================================

if "Sales/Revenue (INR)" in filtered_df.columns:

    total_revenue = filtered_df[
        "Sales/Revenue (INR)"
    ].sum()

else:

    total_revenue = 0


if "Order ID" in filtered_df.columns:

    total_orders = filtered_df[
        "Order ID"
    ].nunique()

else:

    total_orders = len(filtered_df)


if "Customer ID" in filtered_df.columns:

    total_customers = filtered_df[
        "Customer ID"
    ].nunique()

else:

    total_customers = 0


if "Product ID" in filtered_df.columns:

    total_products = filtered_df[
        "Product ID"
    ].nunique()

else:

    total_products = 0


# ============================================================
# KPI DISPLAY
# ============================================================

st.title("🛒 AI-Powered E-Commerce Analytics Platform")

st.markdown(
    "### Business Intelligence Dashboard"
)


col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "Total Revenue",
        f"₹{total_revenue:,.0f}"
    )


with col2:

    st.metric(
        "Total Orders",
        f"{total_orders:,}"
    )


with col3:

    st.metric(
        "Total Customers",
        f"{total_customers:,}"
    )


with col4:

    st.metric(
        "Total Products",
        f"{total_products:,}"
    )


# ============================================================
# DAILY SALES
# ============================================================

daily_sales = (
    filtered_df
    .groupby("Order Date")[
        "Sales/Revenue (INR)"
    ]
    .sum()
    .reset_index()
)


# ============================================================
# MONTHLY SALES
# ============================================================

monthly_sales = (
    filtered_df
    .assign(
        Month=filtered_df[
            "Order Date"
        ].dt.to_period("M").astype(str)
    )
    .groupby("Month")[
        "Sales/Revenue (INR)"
    ]
    .sum()
    .reset_index()
)


# ============================================================
# CATEGORY SALES
# ============================================================

if "Product Category" in filtered_df.columns:

    category_sales = (
        filtered_df
        .groupby("Product Category")[
            "Sales/Revenue (INR)"
        ]
        .sum()
        .reset_index()
        .sort_values(
            "Sales/Revenue (INR)",
            ascending=False
        )
    )

else:

    category_sales = pd.DataFrame()


# ============================================================
# TOP PRODUCTS
# ============================================================

if "Product Name" in filtered_df.columns:

    top_products = (
        filtered_df
        .groupby("Product Name")[
            "Sales/Revenue (INR)"
        ]
        .sum()
        .reset_index()
        .sort_values(
            "Sales/Revenue (INR)",
            ascending=False
        )
        .head(10)
    )

else:

    top_products = pd.DataFrame()


# ============================================================
# CUSTOMER SPENDING
# ============================================================

if "Customer ID" in filtered_df.columns:

    customer_spending = (
        filtered_df
        .groupby("Customer ID")[
            "Sales/Revenue (INR)"
        ]
        .sum()
        .reset_index()
        .sort_values(
            "Sales/Revenue (INR)",
            ascending=False
        )
    )

else:

    customer_spending = pd.DataFrame()


# ============================================================
# SENTIMENT ANALYSIS
# ============================================================

sentiment_file = os.path.join(
    BASE_PATH,
    "product_sentiment.csv"
)

sentiment_df = pd.DataFrame()

if os.path.exists(sentiment_file):

    try:

        sentiment_df = pd.read_csv(
            sentiment_file
        )

        sentiment_df.columns = (
            sentiment_df.columns
            .astype(str)
            .str.strip()
        )

    except Exception:

        sentiment_df = pd.DataFrame()


# ============================================================
# PRODUCT SENTIMENT DATA
# ============================================================

if (
    not sentiment_df.empty
    and "Product Name" in sentiment_df.columns
    and all(
        col in sentiment_df.columns
        for col in [
            "Negative",
            "Neutral",
            "Positive"
        ]
    )
):

    # Make sure percentage columns are numeric
    for col in [
        "Negative",
        "Neutral",
        "Positive"
    ]:

        sentiment_df[col] = pd.to_numeric(
            sentiment_df[col],
            errors="coerce"
        )

    # Remove invalid rows
    sentiment_df = sentiment_df.dropna(
        subset=[
            "Product Name",
            "Negative",
            "Neutral",
            "Positive"
        ]
    )

    # Convert product-level percentages
    # into dashboard-friendly sentiment summary

    sentiment_counts = pd.DataFrame({
        "Sentiment": [
            "Negative",
            "Neutral",
            "Positive"
        ],

        "Reviews": [
            sentiment_df["Negative"].mean(),
            sentiment_df["Neutral"].mean(),
            sentiment_df["Positive"].mean()
        ]
    })

    # Round percentages
    sentiment_counts["Reviews"] = (
        sentiment_counts["Reviews"]
        .round(2)
    )

    # Product sentiment table
    sentiment_filtered = sentiment_df.copy()

    # ========================================================
    # SENTIMENT BY CATEGORY
    # ========================================================
    
    if (
        "Product Category" in sentiment_filtered.columns
    ):

        sentiment_category = pd.crosstab(
            sentiment_filtered["Product Category"],
            columns="count"
        ).reset_index()

    else:

        sentiment_category = pd.DataFrame()

else:

    sentiment_counts = pd.DataFrame(
        columns=[
            "Sentiment",
            "Reviews"
        ]
    )

    sentiment_filtered = pd.DataFrame()

    sentiment_category = pd.DataFrame()


# ============================================================
# PRODUCT RECOMMENDATIONS
# ============================================================

recommendations = pd.DataFrame()


recommendation_file_candidates = [

    os.path.join(
        BASE_PATH,
        "product_recommendations.csv"
    ),

    os.path.join(
        BASE_PATH,
        "product_recommendation.csv"
    )
]


# ============================================================
# FIND RECOMMENDATION CSV
# ============================================================

recommendation_file = None


for file_path in recommendation_file_candidates:

    if os.path.exists(file_path):

        recommendation_file = file_path

        break


# ============================================================
# LOAD RECOMMENDATIONS
# ============================================================

if recommendation_file is not None:

    try:

        recommendations = pd.read_csv(
            recommendation_file
        )

        recommendations.columns = (
            recommendations.columns
            .astype(str)
            .str.strip()
        )

    except Exception:

        recommendations = pd.DataFrame()


# ============================================================
# RECOMMENDATION SECTION
# ============================================================

if (
    not recommendations.empty
    and "Customer ID" in recommendations.columns
):

    customer_list = sorted(
        recommendations["Customer ID"]
        .dropna()
        .unique()
        .tolist()
    )

elif "Customer ID" in filtered_df.columns:

    customer_list = sorted(
        filtered_df["Customer ID"]
        .dropna()
        .unique()
        .tolist()
    )

else:

    customer_list = []


# ============================================================
# CUSTOMER SELECTION
# ============================================================

if customer_list:

    selected_customer = st.selectbox(
        "Select Customer",
        customer_list
    )

else:

    selected_customer = None


# ============================================================
# GET CUSTOMER RECOMMENDATIONS
# ============================================================

if (
    selected_customer is not None
    and not recommendations.empty
    and "Customer ID" in recommendations.columns
):

    customer_recommendations = (
        recommendations[
            recommendations["Customer ID"]
            == selected_customer
        ]
        .copy()
    )

    if "Recommendation Score" in customer_recommendations.columns:

        customer_recommendations = (
            customer_recommendations
            .sort_values(
                "Recommendation Score",
                ascending=False
            )
            .head(5)
        )

else:

    customer_recommendations = pd.DataFrame()


# ============================================================
# DISPLAY RECOMMENDATIONS
# ============================================================

st.subheader("Recommended Products")


if customer_recommendations.empty:

    st.info(
        "No recommendation data is available for this customer."
    )

else:

    display_columns = []

    for col in [
        "Product ID",
        "Product Name",
        "Recommendation Score"
    ]:

        if col in customer_recommendations.columns:

            display_columns.append(col)


    if display_columns:

        st.dataframe(
            customer_recommendations[
                display_columns
            ],
            use_container_width=True
        )

    else:

        st.dataframe(
            customer_recommendations,
            use_container_width=True
        )

# ============================================================
# DASHBOARD SECTION SELECTOR
# ============================================================

section = st.sidebar.selectbox(
    "Select Section",
    [
        "Overview",
        "Sales Analytics",
        "Product Analytics",
        "Customer Analytics",
        "AI Insights",
        "Recommendations"
    ]
)


# ============================================================
# OVERVIEW
# ============================================================

if section == "Overview":

    st.title("📊 Overview")

    st.subheader("Revenue Overview")

    st.plotly_chart(
        px.line(
            monthly_sales,
            x="Month",
            y="Sales/Revenue (INR)",
            markers=True,
            title="Monthly Revenue Trend"
        ),
        use_container_width=True
    )


    if not category_sales.empty:

        st.subheader("Sales by Category")

        st.plotly_chart(
            px.bar(
                category_sales,
                x="Product Category",
                y="Sales/Revenue (INR)",
                title="Category-wise Revenue"
            ),
            use_container_width=True
        )


# ============================================================
# SALES ANALYTICS
# ============================================================

elif section == "Sales Analytics":

    st.title("📈 Sales Analytics")

    st.subheader("Daily Sales Trend")

    st.plotly_chart(
        px.line(
            daily_sales,
            x="Order Date",
            y="Sales/Revenue (INR)",
            title="Daily Revenue Trend"
        ),
        use_container_width=True
    )


    st.subheader("Monthly Revenue Trend")

    st.plotly_chart(
        px.line(
            monthly_sales,
            x="Month",
            y="Sales/Revenue (INR)",
            markers=True,
            title="Monthly Revenue"
        ),
        use_container_width=True
    )


    if not category_sales.empty:

        st.subheader("Category Revenue")

        st.plotly_chart(
            px.bar(
                category_sales,
                x="Product Category",
                y="Sales/Revenue (INR)",
                title="Revenue by Product Category"
            ),
            use_container_width=True
        )


# ============================================================
# PRODUCT ANALYTICS
# ============================================================

elif section == "Product Analytics":

    st.title("📦 Product Analytics")


    if not top_products.empty:

        st.subheader("Top 10 Products")

        st.plotly_chart(
            px.bar(
                top_products,
                x="Sales/Revenue (INR)",
                y="Product Name",
                orientation="h",
                title="Top 10 Products by Revenue"
            ),
            use_container_width=True
        )

        st.dataframe(
            top_products,
            use_container_width=True
        )


    if not category_sales.empty:

        st.subheader("Category Performance")

        st.plotly_chart(
            px.pie(
                category_sales,
                names="Product Category",
                values="Sales/Revenue (INR)",
                title="Revenue Distribution by Category"
            ),
            use_container_width=True
        )


# ============================================================
# CUSTOMER ANALYTICS
# ============================================================

elif section == "Customer Analytics":

    st.title("👥 Customer Analytics")


    if not customer_spending.empty:

        st.subheader("Top Customers by Spending")

        top_customers = customer_spending.head(10)

        st.plotly_chart(
            px.bar(
                top_customers,
                x="Customer ID",
                y="Sales/Revenue (INR)",
                title="Top Customers by Revenue"
            ),
            use_container_width=True
        )

        st.dataframe(
            top_customers,
            use_container_width=True
        )


# # ============================================================
# # AI INSIGHTS
# # ============================================================

elif section == "AI Insights":

    st.header("AI Insights")

# ============================================================
# CELL 1 - CUSTOMER SENTIMENT ANALYSIS
# ============================================================

    import os
    import pickle
    import numpy as np
    import pandas as pd
    import streamlit as st
    import plotly.express as px
    import plotly.graph_objects as go

    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ------------------------------------------------------------
# PROJECT PATH
# ------------------------------------------------------------

    BASE_PATH = r"C:\Users\Dell\Desktop\Analytics Platform"

    REVIEW_SENTIMENT_FILE = os.path.join(
        BASE_PATH,
        "review_sentiment.csv"
    )


# ------------------------------------------------------------
# USE THE DATA ALREADY FILTERED BY DASHBOARD
# ------------------------------------------------------------

    if "filtered_df" in globals() and isinstance(filtered_df, pd.DataFrame):

        ai_df = filtered_df.copy()

    else:

        data_file = os.path.join(
            BASE_PATH,
            "ecommerce_features.csv"
        )

        ai_df = pd.read_csv(data_file)


# ------------------------------------------------------------
# BASIC CLEANING
# ------------------------------------------------------------

    ai_df.columns = (
        ai_df.columns
        .astype(str)
        .str.strip()
    )

    if "Order Date" in ai_df.columns:

        ai_df["Order Date"] = pd.to_datetime(
            ai_df["Order Date"],
            errors="coerce"
        )


    # ------------------------------------------------------------
    # LOAD CUSTOMER REVIEW SENTIMENT
    # ------------------------------------------------------------

    if os.path.exists(REVIEW_SENTIMENT_FILE):

        review_sentiment_df = pd.read_csv(
            REVIEW_SENTIMENT_FILE
        )

        review_sentiment_df.columns = (
            review_sentiment_df.columns
            .astype(str)
            .str.strip()
        )

    else:

        review_sentiment_df = pd.DataFrame()


    # ------------------------------------------------------------
    # FIND SENTIMENT COLUMN
    # ------------------------------------------------------------

    sentiment_column = None

    for column_name in [
        "Predicted_Sentiment",
        "Sentiment",
        "sentiment",
        "Sentiment_Label"
    ]:

        if column_name in review_sentiment_df.columns:

            sentiment_column = column_name
            break


    # ------------------------------------------------------------
    # FILTER SENTIMENT DATA TO CURRENT DASHBOARD FILTER
    # ------------------------------------------------------------

    customer_sentiment_df = review_sentiment_df.copy()

    if (
        not customer_sentiment_df.empty
        and sentiment_column is not None
    ):

        # Match using Order ID
        if (
            "Order ID" in ai_df.columns
            and "Order ID" in customer_sentiment_df.columns
        ):

            valid_orders = set(
                ai_df["Order ID"]
                .dropna()
                .astype(str)
            )

            customer_sentiment_df = (
                customer_sentiment_df[
                    customer_sentiment_df["Order ID"]
                    .astype(str)
                    .isin(valid_orders)
                ]
                .copy()
            )

        # Otherwise match using Customer ID
        elif (
            "Customer ID" in ai_df.columns
            and "Customer ID" in customer_sentiment_df.columns
        ):

            valid_customers = set(
                ai_df["Customer ID"]
                .dropna()
                .astype(str)
            )

            customer_sentiment_df = (
                customer_sentiment_df[
                    customer_sentiment_df["Customer ID"]
                    .astype(str)
                    .isin(valid_customers)
                ]
                .copy()
            )


    # ------------------------------------------------------------
    # CREATE CLEAN SENTIMENT COLUMN
    # ------------------------------------------------------------

    if (
        not customer_sentiment_df.empty
        and sentiment_column is not None
    ):

        customer_sentiment_df["Sentiment"] = (
            customer_sentiment_df[sentiment_column]
            .astype(str)
            .str.strip()
            .str.title()
        )

        customer_sentiment_df = (
            customer_sentiment_df[
                customer_sentiment_df["Sentiment"]
                .isin(
                    [
                        "Positive",
                        "Neutral",
                        "Negative"
                    ]
                )
            ]
            .copy()
        )


    # ------------------------------------------------------------
    # DISPLAY
    # ------------------------------------------------------------

    st.header("🤖 Customer Sentiment Analysis")

    if customer_sentiment_df.empty:

        st.warning(
            "No customer sentiment data is available for the selected filters."
        )

    else:

        sentiment_counts = (
            customer_sentiment_df["Sentiment"]
            .value_counts()
            .reindex(
                [
                    "Positive",
                    "Neutral",
                    "Negative"
                ],
                fill_value=0
            )
            .reset_index()
        )

        sentiment_counts.columns = [
            "Sentiment",
            "Reviews"
        ]


        total_reviews = sentiment_counts["Reviews"].sum()


        positive_count = int(
            sentiment_counts.loc[
                sentiment_counts["Sentiment"] == "Positive",
                "Reviews"
            ].iloc[0]
        )

        neutral_count = int(
            sentiment_counts.loc[
                sentiment_counts["Sentiment"] == "Neutral",
                "Reviews"
            ].iloc[0]
        )

        negative_count = int(
            sentiment_counts.loc[
                sentiment_counts["Sentiment"] == "Negative",
                "Reviews"
            ].iloc[0]
        )


        positive_percentage = (
            positive_count / total_reviews * 100
            if total_reviews > 0
            else 0
        )


        neutral_percentage = (
            neutral_count / total_reviews * 100
            if total_reviews > 0
            else 0
        )


        negative_percentage = (
            negative_count / total_reviews * 100
            if total_reviews > 0
            else 0
        )


        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Positive Reviews",
                f"{positive_percentage:.1f}%"
            )

        with col2:

            st.metric(
                "Neutral Reviews",
                f"{neutral_percentage:.1f}%"
            )

        with col3:

            st.metric(
                "Negative Reviews",
                f"{negative_percentage:.1f}%"
            )


        chart_col1, chart_col2 = st.columns(2)


        with chart_col1:

            fig_sentiment_pie = px.pie(
                sentiment_counts,
                names="Sentiment",
                values="Reviews",
                title="Customer Sentiment Distribution"
            )

            st.plotly_chart(
                fig_sentiment_pie,
                use_container_width=True
            )


        with chart_col2:

            fig_sentiment_bar = px.bar(
                sentiment_counts,
                x="Sentiment",
                y="Reviews",
                title="Customer Sentiment Count"
            )

            st.plotly_chart(
                fig_sentiment_bar,
                use_container_width=True
            )


        # --------------------------------------------------------
        # SENTIMENT BY CATEGORY
        # --------------------------------------------------------

        if (
            "Product Category" in customer_sentiment_df.columns
        ):

            category_sentiment = pd.crosstab(
                customer_sentiment_df["Product Category"],
                customer_sentiment_df["Sentiment"]
            ).reset_index()

            if not category_sentiment.empty:

                st.subheader(
                    "Customer Sentiment by Product Category"
                )

                fig_category_sentiment = px.bar(
                    category_sentiment,
                    x="Product Category",
                    y=[
                        column
                        for column in [
                            "Positive",
                            "Neutral",
                            "Negative"
                        ]
                        if column in category_sentiment.columns
                    ],
                    barmode="group",
                    title="Sentiment by Product Category"
                )

                st.plotly_chart(
                    fig_category_sentiment,
                    use_container_width=True
                )


        # --------------------------------------------------------
        # AI INSIGHT
        # --------------------------------------------------------

        if positive_percentage >= 50:

            sentiment_message = (
                f"💡 AI Insight: Customer sentiment is predominantly positive "
                f"with {positive_percentage:.1f}% positive reviews."
            )

        elif negative_percentage > positive_percentage:

            sentiment_message = (
                f"⚠️ AI Insight: Negative sentiment ({negative_percentage:.1f}%) "
                f"is higher than positive sentiment. Customer experience "
                f"should be investigated."
            )

        else:

            sentiment_message = (
                f"💡 AI Insight: Customer sentiment is mixed. "
                f"Positive reviews account for {positive_percentage:.1f}% "
                f"of the available reviews."
            )


        st.info(sentiment_message)

        # ============================================================
    # CELL 2 - SALES TREND INSIGHTS
    # ============================================================

    st.header("📈 Sales Trend Insights")


    sales_df = ai_df.copy()


    if "Order Date" in sales_df.columns:

        sales_df["Order Date"] = pd.to_datetime(
            sales_df["Order Date"],
            errors="coerce"
        )


    if "Sales/Revenue (INR)" in sales_df.columns:

        sales_df["Sales/Revenue (INR)"] = pd.to_numeric(
            sales_df["Sales/Revenue (INR)"],
            errors="coerce"
        )

    elif "Calculated Revenue (INR)" in sales_df.columns:

        sales_df["Sales/Revenue (INR)"] = pd.to_numeric(
            sales_df["Calculated Revenue (INR)"],
            errors="coerce"
        )

    elif "Total Order Value (INR)" in sales_df.columns:

        sales_df["Sales/Revenue (INR)"] = pd.to_numeric(
            sales_df["Total Order Value (INR)"],
            errors="coerce"
        )


    sales_df = sales_df.dropna(
        subset=[
            "Order Date",
            "Sales/Revenue (INR)"
        ]
    )


    if sales_df.empty:

        st.warning(
            "Sales data is not available."
        )

    else:

        total_revenue = sales_df[
            "Sales/Revenue (INR)"
        ].sum()

        total_orders = (
            sales_df["Order ID"].nunique()
            if "Order ID" in sales_df.columns
            else len(sales_df)
        )

        average_order_value = (
            total_revenue / total_orders
            if total_orders > 0
            else 0
        )


        col1, col2, col3 = st.columns(3)


        with col1:

            st.metric(
                "Total Revenue",
                f"₹{total_revenue:,.0f}"
            )


        with col2:

            st.metric(
                "Total Orders",
                f"{total_orders:,}"
            )


        with col3:

            st.metric(
                "Average Order Value",
                f"₹{average_order_value:,.0f}"
            )


        # --------------------------------------------------------
        # MONTHLY SALES
        # --------------------------------------------------------

        sales_df["Month"] = (
            sales_df["Order Date"]
            .dt.to_period("M")
            .astype(str)
        )


        monthly_sales = (
            sales_df
            .groupby("Month")["Sales/Revenue (INR)"]
            .sum()
            .reset_index()
        )


        monthly_sales["Growth %"] = (
            monthly_sales["Sales/Revenue (INR)"]
            .pct_change()
            .replace(
                [np.inf, -np.inf],
                np.nan
            )
            * 100
        )


        fig_monthly_sales = px.line(
            monthly_sales,
            x="Month",
            y="Sales/Revenue (INR)",
            markers=True,
            title="Monthly Revenue Trend"
        )


        st.plotly_chart(
            fig_monthly_sales,
            use_container_width=True
        )


        # --------------------------------------------------------
        # CATEGORY REVENUE
        # --------------------------------------------------------

        if "Product Category" in sales_df.columns:

            category_sales = (
                sales_df
                .groupby("Product Category")[
                    "Sales/Revenue (INR)"
                ]
                .sum()
                .reset_index()
                .sort_values(
                    "Sales/Revenue (INR)",
                    ascending=False
                )
            )


            fig_category_sales = px.bar(
                category_sales,
                x="Product Category",
                y="Sales/Revenue (INR)",
                title="Revenue by Product Category"
            )


            st.plotly_chart(
                fig_category_sales,
                use_container_width=True
            )


        # --------------------------------------------------------
        # AI SALES INSIGHT
        # --------------------------------------------------------

        if len(monthly_sales) >= 2:

            previous_month = monthly_sales.iloc[-2][
                "Sales/Revenue (INR)"
            ]

            latest_month = monthly_sales.iloc[-1][
                "Sales/Revenue (INR)"
            ]


            if previous_month != 0:

                growth = (
                    (latest_month - previous_month)
                    / previous_month
                    * 100
                )

            else:

                growth = 0


            if growth >= 0:

                st.success(
                    f"💡 AI Insight: Revenue increased by "
                    f"{growth:.2f}% compared with the previous month."
                )

            else:

                st.warning(
                    f"💡 AI Insight: Revenue decreased by "
                    f"{abs(growth):.2f}% compared with the previous month."
                )


        if (
            "Product Category" in sales_df.columns
            and not category_sales.empty
        ):

            top_category = category_sales.iloc[0][
                "Product Category"
            ]

            st.info(
                f"🏆 Highest revenue category: {top_category}"
            )

        # ============================================================
    # CELL 3 - SALES PREDICTION
    # ============================================================

    st.header("🔮 Sales Prediction")


    prediction_df = ai_df.copy()


    if "Order Date" in prediction_df.columns:

        prediction_df["Order Date"] = pd.to_datetime(
            prediction_df["Order Date"],
            errors="coerce"
        )


    if "Sales/Revenue (INR)" in prediction_df.columns:

        prediction_df["Sales/Revenue (INR)"] = pd.to_numeric(
            prediction_df["Sales/Revenue (INR)"],
            errors="coerce"
        )

    elif "Calculated Revenue (INR)" in prediction_df.columns:

        prediction_df["Sales/Revenue (INR)"] = pd.to_numeric(
            prediction_df["Calculated Revenue (INR)"],
            errors="coerce"
        )

    elif "Total Order Value (INR)" in prediction_df.columns:

        prediction_df["Sales/Revenue (INR)"] = pd.to_numeric(
            prediction_df["Total Order Value (INR)"],
            errors="coerce"
        )


    prediction_df = prediction_df.dropna(
        subset=[
            "Order Date",
            "Sales/Revenue (INR)"
        ]
    )


    if prediction_df["Order Date"].nunique() < 35:

        st.warning(
            "Not enough historical dates for reliable sales prediction."
        )

    else:

        # --------------------------------------------------------
        # DAILY REVENUE
        # --------------------------------------------------------

        daily_sales = (
            prediction_df
            .groupby("Order Date")[
                "Sales/Revenue (INR)"
            ]
            .sum()
            .sort_index()
        )


        # Convert to a continuous daily time series.
        # Missing sales days are treated as zero.
        full_dates = pd.date_range(
            start=daily_sales.index.min(),
            end=daily_sales.index.max(),
            freq="D"
        )


        daily_sales = (
            daily_sales
            .reindex(full_dates, fill_value=0)
            .rename("Revenue")
            .to_frame()
        )


        daily_sales.index.name = "Date"


        # --------------------------------------------------------
        # FEATURES
        # --------------------------------------------------------

        daily_sales["Day"] = daily_sales.index.day

        daily_sales["Month"] = daily_sales.index.month

        daily_sales["Year"] = daily_sales.index.year

        daily_sales["DayOfWeek"] = (
            daily_sales.index.dayofweek
        )

        daily_sales["DayOfYear"] = (
            daily_sales.index.dayofyear
        )


        daily_sales["Previous_Day_Revenue"] = (
            daily_sales["Revenue"]
            .shift(1)
        )


        daily_sales["Previous_7_Days_Revenue"] = (
            daily_sales["Revenue"]
            .shift(7)
        )


        daily_sales["Previous_30_Days_Revenue"] = (
            daily_sales["Revenue"]
            .shift(30)
        )


        model_df = daily_sales.dropna().copy()


        features = [
            "Day",
            "Month",
            "Year",
            "DayOfWeek",
            "DayOfYear",
            "Previous_Day_Revenue",
            "Previous_7_Days_Revenue",
            "Previous_30_Days_Revenue"
        ]


        X = model_df[features]

        y = model_df["Revenue"]


        # --------------------------------------------------------
        # CHRONOLOGICAL TRAIN/TEST SPLIT
        # --------------------------------------------------------

        split_index = int(
            len(model_df) * 0.80
        )


        X_train = X.iloc[:split_index]

        X_test = X.iloc[split_index:]


        y_train = y.iloc[:split_index]

        y_test = y.iloc[split_index:]


        # --------------------------------------------------------
        # RANDOM FOREST MODEL
        # --------------------------------------------------------

        sales_model = RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            n_jobs=-1
        )


        sales_model.fit(
            X_train,
            y_train
        )


        y_pred = sales_model.predict(
            X_test
        )


        mae = mean_absolute_error(
            y_test,
            y_pred
        )


        rmse = np.sqrt(
            mean_squared_error(
                y_test,
                y_pred
            )
        )


        r2 = r2_score(
            y_test,
            y_pred
        )


        # --------------------------------------------------------
        # MODEL METRICS
        # --------------------------------------------------------

        col1, col2, col3 = st.columns(3)


        with col1:

            st.metric(
                "MAE",
                f"₹{mae:,.0f}"
            )


        with col2:

            st.metric(
                "RMSE",
                f"₹{rmse:,.0f}"
            )


        with col3:

            st.metric(
                "R² Score",
                f"{r2:.3f}"
            )


        # --------------------------------------------------------
        # ACTUAL VS PREDICTED
        # --------------------------------------------------------

        prediction_result = pd.DataFrame({

            "Date": model_df.index[split_index:],

            "Actual Revenue": y_test.values,

            "Predicted Revenue": y_pred

        })


        fig_prediction = go.Figure()


        fig_prediction.add_trace(
            go.Scatter(
                x=prediction_result["Date"],
                y=prediction_result["Actual Revenue"],
                mode="lines",
                name="Actual Revenue"
            )
        )


        fig_prediction.add_trace(
            go.Scatter(
                x=prediction_result["Date"],
                y=prediction_result["Predicted Revenue"],
                mode="lines",
                name="Predicted Revenue"
            )
        )


        fig_prediction.update_layout(
            title="Actual vs Predicted Revenue",
            xaxis_title="Date",
            yaxis_title="Revenue (INR)"
        )


        st.plotly_chart(
            fig_prediction,
            use_container_width=True
        )


        # --------------------------------------------------------
        # 7-DAY FUTURE FORECAST
        # --------------------------------------------------------

        forecast_history = daily_sales.copy()


        forecast_values = []


        for i in range(7):

            next_date = (
                forecast_history.index.max()
                + pd.Timedelta(days=1)
            )


            previous_day = (
                forecast_history["Revenue"]
                .iloc[-1]
            )


            if len(forecast_history) >= 7:

                previous_7_days = (
                    forecast_history["Revenue"]
                    .iloc[-7]
                )

            else:

                previous_7_days = previous_day


            if len(forecast_history) >= 30:

                previous_30_days = (
                    forecast_history["Revenue"]
                    .iloc[-30]
                )

            else:

                previous_30_days = previous_day


            future_features = pd.DataFrame({

                "Day": [next_date.day],

                "Month": [next_date.month],

                "Year": [next_date.year],

                "DayOfWeek": [next_date.dayofweek],

                "DayOfYear": [next_date.dayofyear],

                "Previous_Day_Revenue": [
                    previous_day
                ],

                "Previous_7_Days_Revenue": [
                    previous_7_days
                ],

                "Previous_30_Days_Revenue": [
                    previous_30_days
                ]

            })


            predicted_revenue = float(
                sales_model.predict(
                    future_features[features]
                )[0]
            )


            predicted_revenue = max(
                predicted_revenue,
                0
            )


            forecast_values.append({

                "Date": next_date,

                "Predicted Revenue": predicted_revenue

            })


            forecast_history.loc[
                next_date,
                "Revenue"
            ] = predicted_revenue


        forecast_df = pd.DataFrame(
            forecast_values
        )


        st.subheader(
            "Next 7 Days Revenue Forecast"
        )


        fig_forecast = px.line(
            forecast_df,
            x="Date",
            y="Predicted Revenue",
            markers=True,
            title="Predicted Revenue for Next 7 Days"
        )


        st.plotly_chart(
            fig_forecast,
            use_container_width=True
        )


        total_forecast = forecast_df[
            "Predicted Revenue"
        ].sum()


        st.success(
            f"🔮 AI Prediction: Expected revenue for "
            f"the next 7 days is approximately "
            f"₹{total_forecast:,.0f}."
        )

        # ============================================================
    # CELL 4 - CUSTOMER SEGMENTATION
    # ============================================================

    st.header("👥 Customer Segmentation")


    customer_df = ai_df.copy()


    if "Customer ID" not in customer_df.columns:

        st.warning(
            "Customer ID is not available for customer segmentation."
        )

    else:

        # --------------------------------------------------------
        # FIND REVENUE COLUMN
        # --------------------------------------------------------

        if "Sales/Revenue (INR)" in customer_df.columns:

            customer_df["Revenue"] = pd.to_numeric(
                customer_df["Sales/Revenue (INR)"],
                errors="coerce"
            )

        elif "Calculated Revenue (INR)" in customer_df.columns:

            customer_df["Revenue"] = pd.to_numeric(
                customer_df["Calculated Revenue (INR)"],
                errors="coerce"
            )

        elif "Total Order Value (INR)" in customer_df.columns:

            customer_df["Revenue"] = pd.to_numeric(
                customer_df["Total Order Value (INR)"],
                errors="coerce"
            )

        else:

            customer_df["Revenue"] = 0


        if "Quantity" in customer_df.columns:

            customer_df["Quantity"] = pd.to_numeric(
                customer_df["Quantity"],
                errors="coerce"
            )

        else:

            customer_df["Quantity"] = 0


        if "Rating" in customer_df.columns:

            customer_df["Rating"] = pd.to_numeric(
                customer_df["Rating"],
                errors="coerce"
            )

        else:

            customer_df["Rating"] = np.nan


        customer_summary = (
            customer_df
            .groupby("Customer ID")
            .agg(
                Number_of_Orders=(
                    "Order ID",
                    "nunique"
                )
                if "Order ID" in customer_df.columns
                else (
                    "Customer ID",
                    "size"
                ),

                Total_Quantity=(
                    "Quantity",
                    "sum"
                ),

                Total_Spending=(
                    "Revenue",
                    "sum"
                ),

                Average_Rating=(
                    "Rating",
                    "mean"
                )
            )
            .reset_index()
        )


        customer_summary["Average_Order_Value"] = (
            customer_summary["Total_Spending"]
            /
            customer_summary["Number_of_Orders"].replace(
                0,
                np.nan
            )
        )


        customer_summary = (
            customer_summary
            .replace(
                [np.inf, -np.inf],
                np.nan
            )
            .fillna(0)
        )


        # --------------------------------------------------------
        # SELECT ML FEATURES
        # --------------------------------------------------------

        segmentation_features = [
            "Number_of_Orders",
            "Total_Quantity",
            "Total_Spending",
            "Average_Rating",
            "Average_Order_Value"
        ]


        if len(customer_summary) >= 6:

            scaler = StandardScaler()


            scaled_data = scaler.fit_transform(
                customer_summary[
                    segmentation_features
                ]
            )


            kmeans = KMeans(
                n_clusters=3,
                random_state=42,
                n_init=10
            )


            customer_summary["Cluster"] = (
                kmeans.fit_predict(
                    scaled_data
                )
            )


            # ----------------------------------------------------
            # GIVE BUSINESS-FRIENDLY NAMES TO CLUSTERS
            # ----------------------------------------------------

            cluster_spending = (
                customer_summary
                .groupby("Cluster")[
                    "Total_Spending"
                ]
                .mean()
                .sort_values()
            )


            cluster_names = {}


            ordered_clusters = list(
                cluster_spending.index
            )


            cluster_names[
                ordered_clusters[0]
            ] = "Low Value"


            cluster_names[
                ordered_clusters[1]
            ] = "Medium Value"


            cluster_names[
                ordered_clusters[2]
            ] = "High Value"


            customer_summary["Customer Segment"] = (
                customer_summary["Cluster"]
                .map(cluster_names)
            )


        else:

            # Fallback for very small filtered datasets
            customer_summary["Customer Segment"] = "Customer"


        # --------------------------------------------------------
        # SEGMENT COUNTS
        # --------------------------------------------------------

        segment_counts = (
            customer_summary[
                "Customer Segment"
            ]
            .value_counts()
            .reset_index()
        )


        segment_counts.columns = [
            "Customer Segment",
            "Customers"
        ]


        col1, col2, col3 = st.columns(3)


        for column, segment_name in zip(
            [col1, col2, col3],
            [
                "High Value",
                "Medium Value",
                "Low Value"
            ]
        ):

            count = int(
                segment_counts.loc[
                    segment_counts[
                        "Customer Segment"
                    ] == segment_name,
                    "Customers"
                ].sum()
            )


            with column:

                st.metric(
                    segment_name + " Customers",
                    count
                )


        # --------------------------------------------------------
        # CHART
        # --------------------------------------------------------

        fig_segments = px.bar(
            segment_counts,
            x="Customer Segment",
            y="Customers",
            title="Customer Segment Distribution"
        )


        st.plotly_chart(
            fig_segments,
            use_container_width=True
        )


        # --------------------------------------------------------
        # SPENDING BY SEGMENT
        # --------------------------------------------------------

        spending_by_segment = (
            customer_summary
            .groupby("Customer Segment")[
                "Total_Spending"
            ]
            .sum()
            .reset_index()
        )


        fig_segment_spending = px.bar(
            spending_by_segment,
            x="Customer Segment",
            y="Total_Spending",
            title="Customer Spending by Segment"
        )


        st.plotly_chart(
            fig_segment_spending,
            use_container_width=True
        )


        # --------------------------------------------------------
        # AI INSIGHT
        # --------------------------------------------------------

        high_value_spending = (
            customer_summary.loc[
                customer_summary[
                    "Customer Segment"
                ] == "High Value",
                "Total_Spending"
            ].sum()
        )


        total_customer_spending = (
            customer_summary[
                "Total_Spending"
            ].sum()
        )


        if total_customer_spending > 0:

            high_value_percentage = (
                high_value_spending
                / total_customer_spending
                * 100
            )

        else:

            high_value_percentage = 0


        st.info(
            f"💡 AI Insight: High-value customers "
            f"contribute approximately "
            f"{high_value_percentage:.1f}% of the spending "
            f"in the selected data."
        )
# ============================================================
# CELL 5 - RATING & REVIEW INSIGHTS
# ============================================================

    st.header("⭐ Rating & Review Insights")


    rating_df = ai_df.copy()


    if "Rating" in rating_df.columns:

        rating_df["Rating"] = pd.to_numeric(
            rating_df["Rating"],
            errors="coerce"
        )

        rating_df = rating_df.dropna(
            subset=["Rating"]
        )


    if rating_df.empty:

        st.warning(
            "Rating data is not available."
        )

    else:

        average_rating = rating_df[
            "Rating"
        ].mean()


        total_ratings = len(
            rating_df
        )


        low_rating_count = len(
            rating_df[
                rating_df["Rating"] <= 2
            ]
        )


        high_rating_count = len(
            rating_df[
                rating_df["Rating"] >= 4
            ]
        )


        col1, col2, col3 = st.columns(3)


        with col1:

            st.metric(
                "Average Rating",
                f"{average_rating:.2f} / 5"
            )


        with col2:

            st.metric(
                "Highly Rated Reviews",
                high_rating_count
            )


        with col3:

            st.metric(
                "Low Rated Reviews",
                low_rating_count
            )


        # --------------------------------------------------------
        # RATING DISTRIBUTION
        # --------------------------------------------------------

        rating_distribution = (
            rating_df[
                "Rating"
            ]
            .value_counts()
            .sort_index()
            .reset_index()
        )


        rating_distribution.columns = [
            "Rating",
            "Reviews"
        ]


        fig_rating = px.bar(
            rating_distribution,
            x="Rating",
            y="Reviews",
            title="Rating Distribution"
        )


        st.plotly_chart(
            fig_rating,
            use_container_width=True
        )


        # --------------------------------------------------------
        # RATING BY CATEGORY
        # --------------------------------------------------------

        if "Product Category" in rating_df.columns:

            category_rating = (
                rating_df
                .groupby(
                    "Product Category"
                )["Rating"]
                .mean()
                .reset_index()
                .sort_values(
                    "Rating",
                    ascending=False
                )
            )


            fig_category_rating = px.bar(
                category_rating,
                x="Product Category",
                y="Rating",
                title="Average Rating by Product Category"
            )


            fig_category_rating.update_yaxes(
                range=[0, 5]
            )


            st.plotly_chart(
                fig_category_rating,
                use_container_width=True
            )


        # --------------------------------------------------------
        # RATING AND SENTIMENT RELATIONSHIP
        # --------------------------------------------------------

        if (
            not customer_sentiment_df.empty
            and "Rating" in customer_sentiment_df.columns
        ):

            sentiment_rating_df = (
                customer_sentiment_df.copy()
            )


            sentiment_rating_df["Rating"] = pd.to_numeric(
                sentiment_rating_df["Rating"],
                errors="coerce"
            )


            sentiment_rating_df = (
                sentiment_rating_df
                .dropna(
                    subset=["Rating"]
                )
            )


            if not sentiment_rating_df.empty:

                rating_sentiment = (
                    sentiment_rating_df
                    .groupby(
                        [
                            "Rating",
                            "Sentiment"
                        ]
                    )
                    .size()
                    .reset_index(
                        name="Reviews"
                    )
                )


                fig_rating_sentiment = px.bar(
                    rating_sentiment,
                    x="Rating",
                    y="Reviews",
                    color="Sentiment",
                    barmode="group",
                    title="Rating vs Customer Sentiment"
                )


                st.plotly_chart(
                    fig_rating_sentiment,
                    use_container_width=True
                )


        # --------------------------------------------------------
        # AI INSIGHT
        # --------------------------------------------------------

        if average_rating >= 4:

            st.success(
                f"💡 AI Insight: Overall customer rating is strong "
                f"at {average_rating:.2f}/5."
            )

        elif average_rating >= 3:

            st.info(
                f"💡 AI Insight: Customer rating is moderate "
                f"at {average_rating:.2f}/5. There is room for improvement."
            )

        else:

            st.warning(
                f"⚠️ AI Insight: Average rating is low at "
                f"{average_rating:.2f}/5. Customer experience "
                f"requires attention."
            )

        # ============================================================
    # CELL 6 - PRODUCT & CATEGORY INSIGHTS
    # ============================================================

    st.header("🛍️ Product & Category Insights")


    product_df = ai_df.copy()


    # ------------------------------------------------------------
    # REVENUE COLUMN
    # ------------------------------------------------------------

    if "Sales/Revenue (INR)" in product_df.columns:

        product_df["Revenue"] = pd.to_numeric(
            product_df["Sales/Revenue (INR)"],
            errors="coerce"
        )

    elif "Calculated Revenue (INR)" in product_df.columns:

        product_df["Revenue"] = pd.to_numeric(
            product_df["Calculated Revenue (INR)"],
            errors="coerce"
        )

    elif "Total Order Value (INR)" in product_df.columns:

        product_df["Revenue"] = pd.to_numeric(
            product_df["Total Order Value (INR)"],
            errors="coerce"
        )

    else:

        product_df["Revenue"] = 0


    if "Quantity" in product_df.columns:

        product_df["Quantity"] = pd.to_numeric(
            product_df["Quantity"],
            errors="coerce"
        )

    else:

        product_df["Quantity"] = 0


    product_df["Revenue"] = (
        product_df["Revenue"]
        .fillna(0)
    )


    product_df["Quantity"] = (
        product_df["Quantity"]
        .fillna(0)
    )


    # ------------------------------------------------------------
    # CATEGORY ANALYSIS
    # ------------------------------------------------------------

    if "Product Category" in product_df.columns:

        category_performance = (
            product_df
            .groupby("Product Category")
            .agg(
                Revenue=(
                    "Revenue",
                    "sum"
                ),
                Quantity_Sold=(
                    "Quantity",
                    "sum"
                ),
                Orders=(
                    "Product Category",
                    "size"
                )
            )
            .reset_index()
            .sort_values(
                "Revenue",
                ascending=False
            )
        )


        if not category_performance.empty:

            top_category = (
                category_performance.iloc[0][
                    "Product Category"
                ]
            )


            top_category_revenue = (
                category_performance.iloc[0][
                    "Revenue"
                ]
            )


            col1, col2 = st.columns(2)


            with col1:

                st.metric(
                    "Top Revenue Category",
                    top_category
                )


            with col2:

                st.metric(
                    "Category Revenue",
                    f"₹{top_category_revenue:,.0f}"
                )


            fig_category_performance = px.bar(
                category_performance,
                x="Product Category",
                y="Revenue",
                title="Category Revenue Performance"
            )


            st.plotly_chart(
                fig_category_performance,
                use_container_width=True
            )


    # ------------------------------------------------------------
    # PRODUCT ANALYSIS
    # ------------------------------------------------------------

    if "Product Name" in product_df.columns:

        product_performance = (
            product_df
            .groupby("Product Name")
            .agg(
                Revenue=(
                    "Revenue",
                    "sum"
                ),
                Quantity_Sold=(
                    "Quantity",
                    "sum"
                )
            )
            .reset_index()
            .sort_values(
                "Revenue",
                ascending=False
            )
        )


        if not product_performance.empty:

            top_products = (
                product_performance
                .head(10)
            )


            st.subheader(
                "Top 10 Products by Revenue"
            )


            fig_top_products = px.bar(
                top_products,
                x="Revenue",
                y="Product Name",
                orientation="h",
                title="Top Products by Revenue"
            )


            st.plotly_chart(
                fig_top_products,
                use_container_width=True
            )


            top_product = (
                product_performance.iloc[0][
                    "Product Name"
                ]
            )


            top_product_revenue = (
                product_performance.iloc[0][
                    "Revenue"
                ]
            )


            st.info(
                f"🏆 Top product by revenue: "
                f"{top_product} "
                f"(₹{top_product_revenue:,.0f})"
            )


    # ------------------------------------------------------------
    # CATEGORY + SENTIMENT COMBINATION
    # ------------------------------------------------------------

    if (
        not customer_sentiment_df.empty
        and "Product Category" in customer_sentiment_df.columns
    ):

        category_sentiment_summary = (
            customer_sentiment_df
            .groupby(
                [
                    "Product Category",
                    "Sentiment"
                ]
            )
            .size()
            .reset_index(
                name="Reviews"
            )
        )


        if not category_sentiment_summary.empty:

            st.subheader(
                "Category Performance with Customer Sentiment"
            )


            fig_category_sentiment_final = px.bar(
                category_sentiment_summary,
                x="Product Category",
                y="Reviews",
                color="Sentiment",
                barmode="group",
                title="Customer Sentiment Across Categories"
            )


            st.plotly_chart(
                fig_category_sentiment_final,
                use_container_width=True
            )


    # ------------------------------------------------------------
    # FINAL AI BUSINESS INSIGHT
    # ------------------------------------------------------------

    if (
        "Product Category" in product_df.columns
        and "Product Name" in product_df.columns
    ):

        if (
            not category_performance.empty
            and not product_performance.empty
        ):

            st.success(
                f"💡 AI Insight: "
                f"{top_category} is currently the strongest "
                f"revenue-generating category, while "
                f"{top_product} is the highest-revenue product. "
                f"These products/categories should be monitored "
                f"for continued demand and customer satisfaction."
            )
# ============================================================
# RECOMMENDATIONS
# ============================================================

elif section == "Recommendations":

    st.title("🎯 Product Recommendations")

    st.write(
        "Personalized product recommendations "
        "for selected customers."
    )


    if not customer_recommendations.empty:

        st.dataframe(
            customer_recommendations,
            use_container_width=True
        )

    else:

        st.info(
            "No recommendation data is available."
        )

# ============================================================
# FOOTER / PROJECT INFORMATION
# ============================================================

st.sidebar.markdown("---")

st.sidebar.markdown(
    """
    **AI-Powered E-Commerce Analytics Platform**

    Built with:
    - Python
    - Streamlit
    - Pandas
    - Plotly
    - Machine Learning
    """
)


# ============================================================
# DATA INFORMATION
# ============================================================

with st.expander("Dataset Information"):

    st.write(
        f"Main dataset: ecommerce_features.csv"
    )

    st.write(
        f"Rows available after filtering: "
        f"{len(filtered_df):,}"
    )

    st.write(
        f"Columns available: "
        f"{len(filtered_df.columns)}"
    )

    st.write(
        "Main dataset columns:"
    )

    st.write(
        list(filtered_df.columns)
    )


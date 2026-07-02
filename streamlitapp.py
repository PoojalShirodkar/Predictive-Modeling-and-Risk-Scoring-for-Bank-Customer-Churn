# Predictive Modeling & Risk Scoring for Bank Customer Churn
# streamlitapp.py
# Part 2.1

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import os


# Page Configuration

st.set_page_config(
    page_title="Bank Customer Churn Dashboard",
    page_icon="🏦",
    layout="wide"
)


# Custom CSS

st.markdown("""
<style>

.main{
    background-color:#F8F9FA;
}

.card{
    padding:20px;
    border-radius:12px;
    background:white;
    box-shadow:2px 2px 8px rgba(0,0,0,0.15);
    text-align:center;
}

.big-font{
    font-size:22px;
    font-weight:bold;
}

.small-font{
    font-size:15px;
}

</style>
""", unsafe_allow_html=True)


# Dashboard Title

st.markdown("""
<h1 style='text-align:center;color:#0B5394'>
🏦 Predictive Modeling & Risk Scoring for Bank Customer Churn
</h1>
""", unsafe_allow_html=True)

st.markdown("---")


# Load Dataset

try:
    df = pd.read_csv("data/European_Bank.csv")
except:
    df = pd.read_csv("European_Bank.csv")

# Remove unnecessary columns
for col in ["CustomerId", "Surname"]:
    if col in df.columns:
        df.drop(col, axis=1, inplace=True)


# Load Model

model = joblib.load("models/random_forest.pkl")
scaler = joblib.load("models/scaler.pkl")


# KPI Calculations

total_customers = len(df)

total_churn = int(df["Exited"].sum())

retention_rate = round(
    (1 - df["Exited"].mean()) * 100,
    2
)

average_balance = round(
    df["Balance"].mean(),
    2
)


# Additional KPI Metrics

avg_age = round(df["Age"].mean(), 1)

avg_credit = round(df["CreditScore"].mean(), 1)

avg_salary = round(df["EstimatedSalary"].mean(), 2)

avg_products = round(df["NumOfProducts"].mean(), 2)

avg_tenure = round(df["Tenure"].mean(), 1)

active_members = int(df["IsActiveMember"].sum())

credit_card_holders = int(df["HasCrCard"].sum())


# KPI Card Function

def card(title, value, color, icon):

    st.markdown(
        f"""
<div style="
background:{color};
border-radius:14px;
padding:10px;
height:125px;
width:100%;
display:flex;
flex-direction:column;
justify-content:center;
align-items:center;
text-align:center;
color:white;
box-shadow:0 3px 10px rgba(0,0,0,0.18);
overflow:hidden;
">

<div style="
font-size:20px;
margin-bottom:6px;
line-height:1;
">
{icon}
</div>

<div style="
font-size:14px;
font-weight:600;
line-height:1.2;
height:34px;
display:flex;
align-items:center;
justify-content:center;
padding:0 4px;
overflow:hidden;
">
{title}
</div>

<div style="
font-size:18px;
font-weight:700;
line-height:1;
margin-top:8px;
white-space:nowrap;
overflow:hidden;
text-overflow:ellipsis;
max-width:100%;
">
{value}
</div>

</div>
""",
        unsafe_allow_html=True,
    )

# KPI Cards

# KPI CARDS - ROW 1

col1,col2,col3,col4,col5 = st.columns(
    [1,1,1,1,1],
    gap="small"
)

with col1:
    card("Total Customers", total_customers, "#2874A6", "👥")

with col2:
    card("Customers Churned", total_churn,"#F4D03F", "📉")

with col3:
    card("Retention Rate", f"{retention_rate}%","#5DADE2", "💙")

with col4:
    card("Average Balance", f"₹{average_balance/1000:.1f}K","#D4AC0D", "💰")

with col5:
    card("Average Age", avg_age,"#85C1E9", "🎂")

st.markdown(
    "<div style='margin-top:-10px'></div>",
    unsafe_allow_html=True
)

# KPI CARDS - ROW 2

col6,col7,col8,col9,col10,col11 = st.columns(
    [1,1,1,1,1,1],
    gap="small"
)

with col6:
    card("Credit Score",avg_credit,"#3498DB", "🏦")

with col7:
    card("Average Salary", f"₹{avg_salary/1000:.0f}K", "#F7DC6F", "💵")

with col8:
    card("Products",avg_products,"#5DADE2", "📦")

with col9:
    card("Active Members",active_members, "#AED6F1", "✅")

with col10:
    card("Card Holders",credit_card_holders,"#F9E79F", "💳")

with col11:
    card("Avg Tenure", f"{avg_tenure} Yrs","#2874A6", "📅")


st.markdown(
    "<div style='margin-bottom:8px'></div>",
    unsafe_allow_html=True
)


# Sidebar


st.sidebar.title("Customer Details")

credit_score = st.sidebar.slider(
    "Credit Score",
    300,
    900,
    650
)

age = st.sidebar.slider(
    "Age",
    18,
    90,
    35
)

tenure = st.sidebar.slider(
    "Tenure",
    0,
    10,
    5
)

balance = st.sidebar.number_input(
    "Balance",
    value=50000.0
)

num_products = st.sidebar.selectbox(
    "Number of Products",
    [1,2,3,4]
)

credit_card = st.sidebar.selectbox(
    "Has Credit Card",
    [0,1]
)

active_member = st.sidebar.selectbox(
    "Active Member",
    [0,1]
)

salary = st.sidebar.number_input(
    "Estimated Salary",
    value=50000.0
)

gender = st.sidebar.selectbox(
    "Gender",
    ["Male","Female"]
)

geography = st.sidebar.selectbox(
    "Geography",
    ["France","Germany","Spain"]
)

st.sidebar.markdown("---")

# Dashboard Filters

st.sidebar.subheader("📊 Dashboard Filters")

filter_gender = st.sidebar.selectbox(
    "Filter Gender",
    ["All"] + sorted(df["Gender"].unique().tolist())
)

filter_geography = st.sidebar.selectbox(
    "Filter Geography",
    ["All"] + sorted(df["Geography"].unique().tolist())
)

filter_age = st.sidebar.slider(
    "Filter Age",
    int(df["Age"].min()),
    int(df["Age"].max()),
    (
        int(df["Age"].min()),
        int(df["Age"].max())
    )
)

# Create Filtered DataFrame
filtered_df = df.copy()

if filter_gender != "All":
    filtered_df = filtered_df[
        filtered_df["Gender"] == filter_gender
    ]

if filter_geography != "All":
    filtered_df = filtered_df[
        filtered_df["Geography"] == filter_geography
    ]

filtered_df = filtered_df[
    (filtered_df["Age"] >= filter_age[0]) &
    (filtered_df["Age"] <= filter_age[1])
]

st.sidebar.markdown("---")

predict_button = st.sidebar.button("Predict Churn")


# Part 2.2 - Prediction Logic


# Feature Engineering Function

def prepare_input():

    balance_salary_ratio = balance / (salary + 1)
    product_density = num_products / (age + 1)
    engagement_product = active_member * num_products
    age_tenure = age * tenure

    gender_male = 1 if gender == "Male" else 0

    geo_germany = 1 if geography == "Germany" else 0
    geo_spain = 1 if geography == "Spain" else 0

    data = pd.DataFrame({

        "CreditScore":[credit_score],
        "Age":[age],
        "Tenure":[tenure],
        "Balance":[balance],
        "NumOfProducts":[num_products],
        "HasCrCard":[credit_card],
        "IsActiveMember":[active_member],
        "EstimatedSalary":[salary],

        "BalanceSalaryRatio":[balance_salary_ratio],
        "ProductDensity":[product_density],
        "EngagementProduct":[engagement_product],
        "AgeTenure":[age_tenure],

        "Gender_Male":[gender_male],
        "Geography_Germany":[geo_germany],
        "Geography_Spain":[geo_spain]

    })

    return data



# Prediction


if predict_button:

    input_df = prepare_input()

    # Scale input
    input_scaled = scaler.transform(input_df)

    # Logistic Regression uses scaled values.
    # Tree models ignore scaling, but using scaled input keeps consistency.
    try:
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1]
    except:
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

    st.markdown("---")

    col1, col2 = st.columns([2,1])

    with col1:

        st.subheader("Prediction Result")

        if prediction == 1:

            st.error("Customer is likely to CHURN")

        else:

            st.success("Customer is likely to STAY")

        st.write("### Churn Probability")

        st.progress(float(probability))

        st.write(f"**{probability*100:.2f}%**")

      
        # Risk Score
        

        if probability < 0.30:

            st.success("🟢 Low Risk")

        elif probability < 0.70:

            st.warning("🟡 Medium Risk")

        else:

            st.error("🔴 High Risk")

    with col2:

        gauge = go.Figure(go.Indicator(

            mode="gauge+number",

            value=probability*100,

            title={'text':"Risk Score"},

            gauge={

                'axis':{'range':[0,100]},

                'bar':{'color':'red'},

                'steps':[

                    {'range':[0,30],'color':'green'},

                    {'range':[30,70],'color':'yellow'},

                    {'range':[70,100],'color':'red'}

                ]

            }

        ))

        st.plotly_chart(
            gauge,
            use_container_width=True
        )


# Probability Pie Chart


    st.markdown("---")

    pie = px.pie(

        names=["Retain","Churn"],

        values=[
            1-probability,
            probability
        ],

        title="Probability Distribution"

    )

    st.plotly_chart(
        pie,
        use_container_width=True
    )

   
# Part 2.3 - Analytics Dashboard


st.markdown("---")

tab1, tab2, tab3 = st.tabs([
    "📊 Dashboard Analytics",
    "📈 Feature Importance",
    "🔄 What-if Analysis"
])


# TAB 1


with tab1:

    st.subheader("Customer Analytics")

    col1, col2, col3 = st.columns(3)

    with col1:

        fig = px.histogram(
            filtered_df,
            x="Age",
            color="Exited",
            title="Age Distribution",
            color_discrete_sequence=["#5DADE2", "#F4D03F"]
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:

        fig = px.histogram(
            filtered_df,
            x="Balance",
            color="Exited",
            title="Balance Distribution",
            color_discrete_sequence=["#85C1E9", "#F7DC6F"]
        )

        st.plotly_chart(fig, use_container_width=True)


    with col3:

        fig = px.bar(
            filtered_df.groupby("Gender")["Exited"].mean().reset_index(),
            x="Gender",
            y="Exited",
            title="Average Churn by Gender",
            color="Gender",
            color_discrete_sequence=["#3498DB", "#F4D03F"]
        )

        st.plotly_chart(fig, use_container_width=True)
    
    col4, col5, col6 = st.columns(3)

    with col4:

        fig = px.bar(
            filtered_df.groupby("Geography")["Exited"].mean().reset_index(),
            x="Geography",
            y="Exited",
            title="Average Churn by Geography",
            color="Geography",
            color_discrete_sequence=[
              "#2874A6",
              "#5DADE2",
              "#F4D03F"
            ]
        )

        st.plotly_chart(fig, use_container_width=True)
    

    with col5:

        fig = px.pie(
            filtered_df,
            names="Exited",
            title="Customer Churn Distribution",
            color_discrete_sequence=[
              "#3498DB",
              "#F4D03F"
            ]
        )

        st.plotly_chart(fig, use_container_width=True)

    with col6:

        fig = px.histogram(
            filtered_df,
            x="CreditScore",
            color="Exited",
            title="Credit Score Distribution",
            color_discrete_sequence=[
              "#AED6F1",
              "#F9E79F"
            ]
        )

        st.plotly_chart(fig, use_container_width=True)

    col7, col8, col9 = st.columns(3)

    with col7:

        fig = px.box(
            filtered_df,
            x="Exited",
            y="Tenure",
            title="Tenure vs Churn",
            color_discrete_sequence=["#5DADE2","#F4D03F"]
        )

        st.plotly_chart(fig, use_container_width=True)

    with col8:

        fig = px.histogram(
            filtered_df,
            x="NumOfProducts",
            color="Exited",
            title="Products Owned by Customers",
            color_discrete_sequence=["#AED6F1","#F7DC6F"]
        )

        st.plotly_chart(fig, use_container_width=True)


    with col9:

        fig = px.bar(
            filtered_df.groupby("IsActiveMember")["Exited"].mean().reset_index(),
            x="IsActiveMember",
            y="Exited",
            title="Active Members vs Churn",
            color_discrete_sequence=["#5DADE2"]
        )

        st.plotly_chart(fig, use_container_width=True)

    col10, col11, col12= st.columns(3)

    with col10:

        fig = px.bar(
            filtered_df.groupby("HasCrCard")["Exited"].mean().reset_index(),
            x="HasCrCard",
            y="Exited",
            title="Credit Card Holders vs Churn",
            color_discrete_sequence=["#F4D03F"]
        )

        st.plotly_chart(fig, use_container_width=True)

    with col11:

        fig = px.histogram(
            filtered_df,
            x="EstimatedSalary",
            color="Exited",
            title="Salary Distribution",
            color_discrete_sequence=["#85C1E9","#F7DC6F"]
        )

        st.plotly_chart(fig, use_container_width=True)
    
    with col12:

        age_group = pd.cut(
            filtered_df["Age"],
            bins=[18,30,40,50,60,100],
            labels=["18-30","31-40","41-50","51-60","60+"]
        )

        chart = (
            filtered_df.assign(AgeGroup=age_group)
            .groupby("AgeGroup")["Exited"]
            .mean()
            .reset_index()
        )

        fig = px.bar(
            chart,
            x="AgeGroup",
            y="Exited",
            title="Churn Rate by Age Group",
            color="Exited",
            color_continuous_scale=["#87CEFA","#FFD54F"]
        )

        st.plotly_chart(fig, use_container_width=True)
    
    col13, col14= st.columns(2)

    with col13:

        import plotly.express as px

        corr = filtered_df.corr(numeric_only=True)

        fig = px.imshow(
            corr,
            text_auto=False,
            color_continuous_scale="Viridis",
            aspect="auto",
            title="Correlation Heatmap"
        )

        fig.update_layout(
            height=700,
            xaxis_title="Features",
            yaxis_title="Features"
        )

        fig.update_xaxes(tickangle=-45)

        st.plotly_chart(
            fig,
            use_container_width=True
        )
    
    with col14:
        fig = px.scatter(
        filtered_df,
        x="EstimatedSalary",
        y="Balance",
        color="Exited",
        size="Age",
        hover_data=["CreditScore","Tenure"],
        title="Balance vs Salary",
        color_discrete_map={
            0:"#64B5F6",
            1:"#FFD54F"
        }
    )

    st.plotly_chart(fig, use_container_width=True)

# TAB 2


with tab2:

    st.subheader("Feature Importance")

    if os.path.exists("models/feature_importance.csv"):

        importance = pd.read_csv(
            "models/feature_importance.csv"
        )

        fig = px.bar(

            importance.head(15),

            x="Importance",

            y="Feature",

            orientation="h",

            title="Top Important Features"

        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.dataframe(importance)

    else:

        st.warning(
            "Run train_model.py first."
        )


# TAB 3


with tab3:

    st.subheader("What-if Scenario Simulator")

    st.write(
        "Change customer details below and observe how the churn risk changes."
    )

    sim_balance = st.slider(
        "Balance",
        0,
        250000,
        int(balance)
    )

    sim_products = st.slider(
        "Number of Products",
        1,
        4,
        int(num_products)
    )

    sim_active = st.selectbox(
        "Active Member",
        [0,1],
        key="sim_active"
    )

    sim_salary = st.slider(
        "Estimated Salary",
        1000,
        200000,
        int(salary)
    )

    if st.button("Run Simulation"):

        balance_salary_ratio = sim_balance / (sim_salary + 1)

        product_density = sim_products / (age + 1)

        engagement_product = sim_active * sim_products

        age_tenure = age * tenure

        gender_male = 1 if gender == "Male" else 0

        geo_germany = 1 if geography == "Germany" else 0

        geo_spain = 1 if geography == "Spain" else 0

        simulation = pd.DataFrame({

            "CreditScore":[credit_score],
            "Age":[age],
            "Tenure":[tenure],
            "Balance":[sim_balance],
            "NumOfProducts":[sim_products],
            "HasCrCard":[credit_card],
            "IsActiveMember":[sim_active],
            "EstimatedSalary":[sim_salary],

            "BalanceSalaryRatio":[balance_salary_ratio],
            "ProductDensity":[product_density],
            "EngagementProduct":[engagement_product],
            "AgeTenure":[age_tenure],

            "Gender_Male":[gender_male],
            "Geography_Germany":[geo_germany],
            "Geography_Spain":[geo_spain]

        })

        try:

            simulation_scaled = scaler.transform(simulation)

            sim_probability = model.predict_proba(
                simulation_scaled
            )[0][1]

        except:

            sim_probability = model.predict_proba(
                simulation
            )[0][1]

        st.metric(
            "Predicted Churn Probability",
            f"{sim_probability*100:.2f}%"
        )

        fig = go.Figure(go.Indicator(

            mode="gauge+number",

            value=sim_probability*100,

            title={'text':'Simulated Churn Risk'},

            gauge={

                'axis':{'range':[0,100]},

                'bar':{'color':'darkblue'}

            }

        ))

        st.plotly_chart(
            fig,
            use_container_width=True
        )

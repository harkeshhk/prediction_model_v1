import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="ABC Ltd. Predictive Analytics",
    page_icon="📊",
    layout="wide"
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"


# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_models():

    with open(MODEL_DIR / "logistic_model.sav", "rb") as file:
        logistic_model = pickle.load(file)

    with open(MODEL_DIR / "linear_model.sav", "rb") as file:
        linear_model = pickle.load(file)

    return logistic_model, linear_model


@st.cache_data
def load_json_files():

    with open(MODEL_DIR / "metrics.json", "r") as file:
        metrics = json.load(file)

    with open(MODEL_DIR / "model_config.json", "r") as file:
        config = json.load(file)

    return metrics, config


# ============================================================
# TRY TO LOAD MODELS
# ============================================================

try:

    logistic_model, linear_model = load_models()
    metrics, config = load_json_files()

except Exception as e:

    st.error("The application could not load the trained models.")

    st.code(str(e))

    st.info(
        "Please make sure the four files are present inside the "
        "'models' folder and that the scikit-learn version used "
        "to load the models matches the version used during training."
    )

    st.stop()


# ============================================================
# HEADER
# ============================================================

st.title("📊 ABC Ltd. Predictive Analytics Dashboard")

st.markdown(
    """
    ### Predictive Analytics & Managerial Decision Support

    This application uses two machine-learning models:

    - **Logistic Regression** → predicts employee attrition risk
    - **Linear Regression** → predicts estimated monthly income

    The predictions are intended as **decision-support information**
    and should be interpreted together with managerial judgement.
    """
)

st.divider()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("👤 Employee Profile")

st.sidebar.markdown(
    "Enter the employee information below to generate predictions."
)


# ------------------------------------------------------------
# Employee demographics
# ------------------------------------------------------------

st.sidebar.subheader("Demographics")

age = st.sidebar.slider(
    "Age",
    min_value=18,
    max_value=60,
    value=35
)

gender = st.sidebar.selectbox(
    "Gender",
    ["Male", "Female"]
)

marital_status = st.sidebar.selectbox(
    "Marital Status",
    ["Single", "Married", "Divorced"]
)

education = st.sidebar.selectbox(
    "Education Level",
    [1, 2, 3, 4, 5],
    index=1,
    help="1 = Below College, 2 = College, 3 = Bachelor, "
         "4 = Master, 5 = Doctor"
)

education_field = st.sidebar.selectbox(
    "Education Field",
    [
        "Life Sciences",
        "Medical",
        "Marketing",
        "Technical Degree",
        "Human Resources",
        "Other"
    ]
)


# ------------------------------------------------------------
# Job information
# ------------------------------------------------------------

st.sidebar.subheader("Job Information")

department = st.sidebar.selectbox(
    "Department",
    [
        "Sales",
        "Research & Development",
        "Human Resources"
    ]
)

job_role = st.sidebar.selectbox(
    "Job Role",
    [
        "Sales Executive",
        "Research Scientist",
        "Laboratory Technician",
        "Manufacturing Director",
        "Healthcare Representative",
        "Manager",
        "Sales Representative",
        "Research Director",
        "Human Resources"
    ]
)

job_level = st.sidebar.selectbox(
    "Job Level",
    [1, 2, 3, 4, 5],
    index=1
)

business_travel = st.sidebar.selectbox(
    "Business Travel",
    [
        "Travel_Rarely",
        "Travel_Frequently",
        "Non-Travel"
    ]
)

distance_from_home = st.sidebar.slider(
    "Distance From Home",
    min_value=1,
    max_value=30,
    value=10
)

num_companies_worked = st.sidebar.slider(
    "Number of Companies Worked",
    min_value=0,
    max_value=10,
    value=2
)


# ------------------------------------------------------------
# Job satisfaction and environment
# ------------------------------------------------------------

st.sidebar.subheader("Employee Experience")

environment_satisfaction = st.sidebar.selectbox(
    "Environment Satisfaction",
    [1, 2, 3, 4],
    index=2
)

job_satisfaction = st.sidebar.selectbox(
    "Job Satisfaction",
    [1, 2, 3, 4],
    index=2
)

job_involvement = st.sidebar.selectbox(
    "Job Involvement",
    [1, 2, 3, 4],
    index=2
)

relationship_satisfaction = st.sidebar.selectbox(
    "Relationship Satisfaction",
    [1, 2, 3, 4],
    index=2
)

work_life_balance = st.sidebar.selectbox(
    "Work-Life Balance",
    [1, 2, 3, 4],
    index=2
)

overtime = st.sidebar.selectbox(
    "OverTime",
    ["Yes", "No"]
)


# ------------------------------------------------------------
# Career information
# ------------------------------------------------------------

st.sidebar.subheader("Career Information")

total_working_years = st.sidebar.slider(
    "Total Working Years",
    min_value=0,
    max_value=40,
    value=10
)

years_at_company = st.sidebar.slider(
    "Years at Company",
    min_value=0,
    max_value=40,
    value=5
)

years_current_role = st.sidebar.slider(
    "Years in Current Role",
    min_value=0,
    max_value=20,
    value=3
)

years_since_promotion = st.sidebar.slider(
    "Years Since Last Promotion",
    min_value=0,
    max_value=15,
    value=2
)

years_with_manager = st.sidebar.slider(
    "Years With Current Manager",
    min_value=0,
    max_value=20,
    value=3
)

training_times = st.sidebar.slider(
    "Training Times Last Year",
    min_value=0,
    max_value=10,
    value=3
)

stock_option_level = st.sidebar.selectbox(
    "Stock Option Level",
    [0, 1, 2, 3],
    index=0
)


# ------------------------------------------------------------
# Compensation
# ------------------------------------------------------------

st.sidebar.subheader("Compensation & Performance")

daily_rate = st.sidebar.number_input(
    "Daily Rate",
    min_value=0,
    max_value=1500,
    value=800
)

hourly_rate = st.sidebar.number_input(
    "Hourly Rate",
    min_value=0,
    max_value=150,
    value=65
)

monthly_rate = st.sidebar.number_input(
    "Monthly Rate",
    min_value=0,
    max_value=30000,
    value=14000
)

percent_salary_hike = st.sidebar.slider(
    "Percent Salary Hike",
    min_value=10,
    max_value=30,
    value=15
)

performance_rating = st.sidebar.selectbox(
    "Performance Rating",
    [1, 2, 3, 4],
    index=2
)


# ============================================================
# CREATE INPUT DATAFRAME
# ============================================================

input_data = pd.DataFrame({
    "Age": [age],
    "BusinessTravel": [business_travel],
    "DailyRate": [daily_rate],
    "Department": [department],
    "DistanceFromHome": [distance_from_home],
    "Education": [education],
    "EducationField": [education_field],
    "EnvironmentSatisfaction": [environment_satisfaction],
    "Gender": [gender],
    "HourlyRate": [hourly_rate],
    "JobInvolvement": [job_involvement],
    "JobLevel": [job_level],
    "JobRole": [job_role],
    "JobSatisfaction": [job_satisfaction],
    "MaritalStatus": [marital_status],
    "MonthlyRate": [monthly_rate],
    "NumCompaniesWorked": [num_companies_worked],
    "OverTime": [overtime],
    "PercentSalaryHike": [percent_salary_hike],
    "PerformanceRating": [performance_rating],
    "RelationshipSatisfaction": [relationship_satisfaction],
    "StockOptionLevel": [stock_option_level],
    "TotalWorkingYears": [total_working_years],
    "TrainingTimesLastYear": [training_times],
    "WorkLifeBalance": [work_life_balance],
    "YearsAtCompany": [years_at_company],
    "YearsInCurrentRole": [years_current_role],
    "YearsSinceLastPromotion": [years_since_promotion],
    "YearsWithCurrManager": [years_with_manager]
})


# ============================================================
# MAIN TABS
# ============================================================

tab1, tab2, tab3 = st.tabs([
    "🔮 Predictions",
    "📈 Model Validation",
    "📋 Employee Profile"
])


# ============================================================
# TAB 1 — PREDICTIONS
# ============================================================

with tab1:

    st.header("Employee Predictions")

    col1, col2, col3 = st.columns(3)

    # --------------------------------------------------------
    # Logistic prediction
    # --------------------------------------------------------

    attrition_probability = logistic_model.predict_proba(
        input_data
    )[0][1]

    attrition_prediction = logistic_model.predict(
        input_data
    )[0]

    risk_percentage = attrition_probability * 100

    if risk_percentage < 30:
        risk_category = "Low Risk"
    elif risk_percentage < 60:
        risk_category = "Medium Risk"
    else:
        risk_category = "High Risk"

    with col1:

        st.metric(
            "Attrition Probability",
            f"{risk_percentage:.1f}%"
        )

    with col2:

        st.metric(
            "Risk Category",
            risk_category
        )

    # --------------------------------------------------------
    # Linear prediction
    # --------------------------------------------------------

    predicted_income = linear_model.predict(
        input_data
    )[0]

    with col3:

        st.metric(
            "Predicted Monthly Income",
            f"₹{predicted_income:,.0f}"
        )

    st.divider()

    # --------------------------------------------------------
    # Interpretation
    # --------------------------------------------------------

    st.subheader("Model Interpretation")

    if attrition_prediction == 1:

        st.warning(
            "The logistic regression model classifies this employee "
            "as having a higher likelihood of attrition."
        )

    else:

        st.success(
            "The logistic regression model classifies this employee "
            "as having a lower likelihood of attrition."
        )

    st.info(
        f"The estimated attrition probability is "
        f"**{risk_percentage:.1f}%**. "
        "This is a statistical prediction and should not be treated "
        "as a standalone HR decision."
    )

    # --------------------------------------------------------
    # What-if analysis
    # --------------------------------------------------------

    st.divider()

    st.subheader("🔄 What-If Analysis")

    st.write(
        "Change selected employee conditions and compare the "
        "resulting attrition probability."
    )

    whatif_col1, whatif_col2 = st.columns(2)

    with whatif_col1:

        whatif_overtime = st.selectbox(
            "What-if OverTime",
            ["Yes", "No"],
            index=0 if overtime == "Yes" else 1
        )

        whatif_job_satisfaction = st.selectbox(
            "What-if Job Satisfaction",
            [1, 2, 3, 4],
            index=job_satisfaction - 1
        )

    with whatif_col2:

        whatif_distance = st.slider(
            "What-if Distance From Home",
            min_value=1,
            max_value=30,
            value=distance_from_home
        )

        whatif_involvement = st.selectbox(
            "What-if Job Involvement",
            [1, 2, 3, 4],
            index=job_involvement - 1
        )

    whatif_data = input_data.copy()

    whatif_data["OverTime"] = whatif_overtime
    whatif_data["JobSatisfaction"] = whatif_job_satisfaction
    whatif_data["DistanceFromHome"] = whatif_distance
    whatif_data["JobInvolvement"] = whatif_involvement

    whatif_probability = logistic_model.predict_proba(
        whatif_data
    )[0][1]

    difference = (
        whatif_probability - attrition_probability
    ) * 100

    st.metric(
        "What-If Attrition Probability",
        f"{whatif_probability * 100:.1f}%",
        delta=f"{difference:+.1f} percentage points"
    )


# ============================================================
# TAB 2 — MODEL VALIDATION
# ============================================================

with tab2:

    st.header("📈 Model Validation")

    st.subheader("Logistic Regression — Employee Attrition")

    log_metrics = metrics["logistic_regression"]

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric(
            "Accuracy",
            f"{log_metrics['accuracy']:.3f}"
        )

    with col2:
        st.metric(
            "Precision",
            f"{log_metrics['precision']:.3f}"
        )

    with col3:
        st.metric(
            "Recall",
            f"{log_metrics['recall']:.3f}"
        )

    with col4:
        st.metric(
            "F1 Score",
            f"{log_metrics['f1_score']:.3f}"
        )

    with col5:
        st.metric(
            "ROC-AUC",
            f"{log_metrics['roc_auc']:.3f}"
        )

    st.write(
        f"5-fold Cross-Validation Mean F1: "
        f"**{log_metrics['cv_mean_f1']:.3f}**"
    )

    st.write(
        f"5-fold Cross-Validation Mean ROC-AUC: "
        f"**{log_metrics['cv_mean_roc_auc']:.3f}**"
    )

    st.subheader("Confusion Matrix")

    cm = np.array(
        log_metrics["confusion_matrix"]
    )

    cm_df = pd.DataFrame(
        cm,
        index=["Actual No", "Actual Yes"],
        columns=["Predicted No", "Predicted Yes"]
    )

    st.dataframe(
        cm_df,
        use_container_width=True
    )

    st.divider()

    st.subheader("Linear Regression — Monthly Income")

    lin_metrics = metrics["linear_regression"]

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "R²",
            f"{lin_metrics['r2']:.3f}"
        )

    with col2:

        st.metric(
            "MAE",
            f"₹{lin_metrics['mae']:,.0f}"
        )

    with col3:

        st.metric(
            "RMSE",
            f"₹{lin_metrics['rmse']:,.0f}"
        )

    st.write(
        f"5-fold Cross-Validation Mean R²: "
        f"**{lin_metrics['cv_mean_r2']:.3f}**"
    )

    st.write(
        f"5-fold Cross-Validation Mean MAE: "
        f"**₹{lin_metrics['cv_mean_mae']:,.0f}**"
    )


# ============================================================
# TAB 3 — EMPLOYEE PROFILE
# ============================================================

with tab3:

    st.header("📋 Employee Profile")

    st.dataframe(
        input_data.T.rename(
            columns={0: "Value"}
        ),
        use_container_width=True
    )

    st.caption(
        "The values above represent the employee profile submitted "
        "to the predictive models."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "ABC Ltd. Predictive Analytics | "
    "Logistic Regression + Linear Regression | "
    "Decision-support tool for managerial use"
)

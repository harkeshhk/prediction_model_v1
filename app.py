import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import requests
from pathlib import Path


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Z-Fire Ltd. Predictive Analytics",
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
# LOAD MODELS
# ============================================================

try:

    logistic_model, linear_model = load_models()
    metrics, config = load_json_files()

except Exception as e:

    st.error("The application could not load the trained models.")
    st.code(str(e))
    st.stop()





import requests
import json

def ask_ai(question, employee_profile, attrition_probability,
                risk_category, predicted_income, metrics):
    try:
        log_metrics = metrics["logistic_regression"]
        lin_metrics = metrics["linear_regression"]
        employee_text = employee_profile.to_dict(orient="records")[0]

        prompt = f"""
You are an AI Manager Assistant inside an academic
Predictive Analytics application for a fictional company
called Z-Fire Ltd.

Your role is to help a manager INTERPRET predictive analytics
outputs and think through managerial questions.

IMPORTANT RULES:
1. Do not claim that the model knows whether an employee will actually leave.
2. Treat the attrition probability as a statistical estimate, not a certainty.
3. Do not recommend firing, demotion, salary reduction, promotion, or employment action solely because of the prediction.
4. Combine model evidence with employee context and managerial judgment.
5. Distinguish between model predictions, employee profile, and managerial considerations.
6. Do not invent facts not present in the employee profile.
7. If unanswerable from supplied info, say so.
8. Keep response practical for an MBA-level manager.
9. Use simple language unless technical details are requested.

CURRENT EMPLOYEE PROFILE:
{employee_text}

MODEL OUTPUTS:
Attrition probability: {attrition_probability * 100:.2f}%
Risk category: {risk_category}
Predicted monthly income: ₹{predicted_income:,.0f}

LOGISTIC REGRESSION PERFORMANCE:
Accuracy: {log_metrics["accuracy"]:.3f}
Precision: {log_metrics["precision"]:.3f}
Recall: {log_metrics["recall"]:.3f}
F1: {log_metrics["f1_score"]:.3f}
ROC-AUC: {log_metrics["roc_auc"]:.3f}

LINEAR REGRESSION PERFORMANCE:
R²: {lin_metrics["r2"]:.3f}
MAE: ₹{lin_metrics["mae"]:,.0f}
RMSE: ₹{lin_metrics["rmse"]:,.0f}

MANAGER'S QUESTION:
{question}

Answer the manager's question structured using:
### What the model says
### What it may mean
### What the manager should consider
### Important limitation
"""

        # Google Apps Script Web App URL
        script_url = "https://script.google.com/macros/s/AKfycbxt17bDy6pMLVyEpONTI-o9wH67sqzB6WWitkUth5i0dZP5B8WXk0yxJehNQ5kJm5k-/exec"
        
        # Pack data neatly into URL parameters for a GET request
        params = {
            "call": "askGemini",
            "prompt": prompt
        }
        
        # Send GET request instead of POST to completely avoid redirect bugs
        response = requests.get(script_url, params=params, timeout=60)
        
        if response.status_code != 200:
            return f"Error from Apps Script server: HTTP {response.status_code}"
            
        raw_output = response.text.strip()
        
        # Check if Google returned an HTML login/error page
        if raw_output.startswith("<!DOCTYPE") or "<html" in raw_output.lower():
            return f"### Google Authorization/HTML Error:\n{raw_output[:300]}"

        # Parse standard JSON response
        try:
            result_json = json.loads(raw_output)
        except json.JSONDecodeError:
            return f"### Parsing Error\nCould not parse response as JSON. Raw output:\n```text\n{raw_output[:500]}\n```"
            
        if "error" in result_json:
            return f"### Gemini Execution Error\n`{result_json['error']}`"
            
        # Extract text from standard Gemini API JSON response structure
        try:
            answer_text = result_json["candidates"][0]["content"]["parts"][0]["text"]
            return answer_text
        except KeyError:
            return f"Unexpected JSON structure from script: {json.dumps(result_json)}"

    except Exception as e:
        return f"""
### Gemini could not generate a response

The application encountered the following issue:
`{str(e)}`

The predictive models are still available and their predictions are unaffected.
"""
        
# ============================================================
# GEMINI FUNCTION
# ============================================================

def ask_gemini(question, employee_profile, attrition_probability,
               risk_category, predicted_income, metrics):

    try:

        from google import genai

        api_key = st.secrets.get("GEMINI_API_KEY")

        if not api_key:
            return (
                "Gemini API key is not configured. "
                "Please add GEMINI_API_KEY to Streamlit secrets."
            )

        client = genai.Client(api_key=api_key)

        log_metrics = metrics["logistic_regression"]
        lin_metrics = metrics["linear_regression"]

        employee_text = employee_profile.to_dict(
            orient="records"
        )[0]

        prompt = f"""
You are an AI Manager Assistant inside an academic
Predictive Analytics application for a fictional company
called Z-Fire Ltd.

Your role is to help a manager INTERPRET predictive analytics
outputs and think through managerial questions.

IMPORTANT RULES:

1. Do not claim that the model knows whether an employee
   will actually leave.
2. Treat the attrition probability as a statistical estimate,
   not a certainty.
3. Do not recommend firing, demotion, salary reduction,
   promotion, or any other employment action solely because
   of the prediction.
4. Encourage the manager to combine model evidence with
   employee context, managerial judgement, and appropriate
   HR processes.
5. Clearly distinguish between:
   - what the model predicts,
   - what the employee profile shows,
   - and managerial considerations.
6. Do not invent facts that are not present in the supplied
   employee profile.
7. If the question cannot be answered from the supplied
   information, say so.
8. Keep the response practical and suitable for an MBA-level
   manager.
9. Use simple language unless the manager asks for technical
   detail.

CURRENT EMPLOYEE PROFILE:
{employee_text}

MODEL OUTPUTS:

Attrition probability:
{attrition_probability * 100:.2f}%

Risk category:
{risk_category}

Predicted monthly income:
₹{predicted_income:,.0f}

LOGISTIC REGRESSION PERFORMANCE:

Accuracy:
{log_metrics["accuracy"]:.3f}

Precision:
{log_metrics["precision"]:.3f}

Recall:
{log_metrics["recall"]:.3f}

F1:
{log_metrics["f1_score"]:.3f}

ROC-AUC:
{log_metrics["roc_auc"]:.3f}

LINEAR REGRESSION PERFORMANCE:

R²:
{lin_metrics["r2"]:.3f}

MAE:
₹{lin_metrics["mae"]:,.0f}

RMSE:
₹{lin_metrics["rmse"]:,.0f}

MANAGER'S QUESTION:
{question}

Answer the manager's question.

Where useful, structure the response using:

### What the model says
### What it may mean
### What the manager should consider
### Important limitation

Do not make employment decisions on behalf of the manager.
"""

        response = client.models.generate_content(
            model="gemini-3.1-flash-lite",
            contents=prompt
        )

        return response.text

    except Exception as e:

        return f"""
### Gemini could not generate a response

The application encountered the following issue:

`{str(e)}`

The predictive models are still available and their
predictions are unaffected.
"""


# ============================================================
# HEADER
# ============================================================

st.title("📊 Z-Fire Ltd. Predictive Analytics Dashboard")

st.markdown(
    """
### Predictive Analytics & Managerial Decision Support
#### Model crafted by Group 03
##### Aiswarya, Harkesh, Kalpesh, Sakshi, Santhossh, Siddhant, Suman



This application combines:

- **Logistic Regression** → employee attrition prediction
- **Linear Regression** → estimated monthly income
- **Generative AI** → natural-language managerial interpretation
"""
)

st.divider()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("👤 Employee Profile")

st.sidebar.caption(
    "Enter employee information to generate predictions."
)


# ------------------------------------------------------------
# DEMOGRAPHICS
# ------------------------------------------------------------

st.sidebar.subheader("Demographics")

age = st.sidebar.slider(
    "Age", 18, 60, 35
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
# JOB INFORMATION
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
    1, 30, 10
)

num_companies_worked = st.sidebar.slider(
    "Number of Companies Worked",
    0, 10, 2
)


# ------------------------------------------------------------
# EMPLOYEE EXPERIENCE
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
# CAREER
# ------------------------------------------------------------

st.sidebar.subheader("Career Information")

total_working_years = st.sidebar.slider(
    "Total Working Years",
    0, 40, 10
)

years_at_company = st.sidebar.slider(
    "Years at Company",
    0, 40, 5
)

years_current_role = st.sidebar.slider(
    "Years in Current Role",
    0, 20, 3
)

years_since_promotion = st.sidebar.slider(
    "Years Since Last Promotion",
    0, 15, 2
)

years_with_manager = st.sidebar.slider(
    "Years With Current Manager",
    0, 20, 3
)

training_times = st.sidebar.slider(
    "Training Times Last Year",
    0, 10, 3
)

stock_option_level = st.sidebar.selectbox(
    "Stock Option Level",
    [0, 1, 2, 3],
    index=0
)


# ------------------------------------------------------------
# COMPENSATION
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
    10, 30, 15
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
# PREDICTIONS
# ============================================================

attrition_probability = logistic_model.predict_proba(
    input_data
)[0][1]

attrition_prediction = logistic_model.predict(
    input_data
)[0]

predicted_income = linear_model.predict(
    input_data
)[0]

risk_percentage = attrition_probability * 100

if risk_percentage < 30:
    risk_category = "Low Risk"
elif risk_percentage < 60:
    risk_category = "Medium Risk"
else:
    risk_category = "High Risk"


# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3, tab4 = st.tabs([
    "🔮 Predictions",
    "📈 Model Validation",
    "📋 Employee Profile",
    "🤖 AI Manager Assistant"
])


# ============================================================
# TAB 1
# ============================================================

with tab1:

    st.header("Employee Predictions")

    col1, col2, col3 = st.columns(3)

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

    with col3:
        st.metric(
            "Predicted Monthly Income",
            f"₹{predicted_income:,.0f}"
        )

    st.divider()

    if attrition_prediction == 1:

        st.warning(
            "The model classifies this employee as having "
            "a higher likelihood of attrition."
        )

    else:

        st.success(
            "The model classifies this employee as having "
            "a lower likelihood of attrition."
        )

    st.info(
        f"The estimated attrition probability is "
        f"**{risk_percentage:.1f}%**. "
        "This is a statistical estimate rather than a certainty."
    )

    st.divider()

    st.subheader("🔄 What-If Analysis")

    st.write(
        "Change selected conditions to see how the model's "
        "attrition probability changes."
    )

    col1, col2 = st.columns(2)

    with col1:

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

    with col2:

        whatif_distance = st.slider(
            "What-if Distance From Home",
            1, 30,
            distance_from_home
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
# TAB 2
# ============================================================

with tab2:

    st.header("📈 Model Validation")

    st.subheader("Logistic Regression — Employee Attrition")

    log_metrics = metrics["logistic_regression"]

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric("Accuracy", f"{log_metrics['accuracy']:.3f}")
    c2.metric("Precision", f"{log_metrics['precision']:.3f}")
    c3.metric("Recall", f"{log_metrics['recall']:.3f}")
    c4.metric("F1 Score", f"{log_metrics['f1_score']:.3f}")
    c5.metric("ROC-AUC", f"{log_metrics['roc_auc']:.3f}")

    st.write(
        f"5-fold CV Mean F1: "
        f"**{log_metrics['cv_mean_f1']:.3f}**"
    )

    st.write(
        f"5-fold CV Mean ROC-AUC: "
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

    c1, c2, c3 = st.columns(3)

    c1.metric("R²", f"{lin_metrics['r2']:.3f}")
    c2.metric("MAE", f"₹{lin_metrics['mae']:,.0f}")
    c3.metric("RMSE", f"₹{lin_metrics['rmse']:,.0f}")

    st.write(
        f"5-fold CV Mean R²: "
        f"**{lin_metrics['cv_mean_r2']:.3f}**"
    )

    st.write(
        f"5-fold CV Mean MAE: "
        f"**₹{lin_metrics['cv_mean_mae']:,.0f}**"
    )


# ============================================================
# TAB 3
# ============================================================

with tab3:

    st.header("📋 Employee Profile")

    st.dataframe(
        input_data.T.rename(
            columns={0: "Value"}
        ),
        use_container_width=True
    )


# ============================================================
# TAB 4 — GEMINI
# ============================================================

with tab4:

    st.header("🤖 AI Manager Assistant")

    st.markdown(
        """
        Ask questions about the current employee profile,
        model predictions, or managerial interpretation.

        **The predictive models generate the numerical predictions;
        Gemini provides natural-language interpretation.**
        """
    )

    # --------------------------------------------------------
    # Current model context
    # --------------------------------------------------------

    st.subheader("Current Model Context")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Attrition Probability",
        f"{risk_percentage:.1f}%"
    )

    c2.metric(
        "Risk Category",
        risk_category
    )

    c3.metric(
        "Predicted Monthly Income",
        f"₹{predicted_income:,.0f}"
    )

    st.divider()

    # --------------------------------------------------------
    # Suggested questions
    # --------------------------------------------------------

    st.subheader("Suggested Questions")

    suggested_questions = [
        "Explain this employee's attrition prediction in simple managerial language.",
        "What aspects of this employee profile should a manager investigate?",
        "What are the main limitations of this prediction?",
        "How should I interpret the predicted monthly income?",
        "What should a manager consider before acting on this prediction?"
    ]

    selected_question = st.selectbox(
        "Choose a suggested question",
        ["-- Select --"] + suggested_questions
    )

    # --------------------------------------------------------
    # Manual question
    # --------------------------------------------------------

    question = st.text_area(
        "Or ask your own question:",
        placeholder=(
            "Example: Why might this employee have a high "
            "attrition probability?"
        ),
        height=120
    )

    if st.button(
        "Ask Gemini",
        type="primary"
    ):

        final_question = question.strip()

        if not final_question:

            if selected_question != "-- Select --":
                final_question = selected_question

        if not final_question:

            st.warning(
                "Please select a suggested question or "
                "type your own question."
            )

        else:

            with st.spinner(
                "Gemini is analysing the employee profile..."
            ):

                answer = ask_ai(
                    final_question,
                    input_data,
                    attrition_probability,
                    risk_category,
                    predicted_income,
                    metrics
                )

            st.markdown("### Gemini's Response")

            st.markdown(answer)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Z-Fire Ltd. Predictive Analytics | "
    "Logistic Regression + Linear Regression + Generative AI | "
    "Managerial Decision Support"
)

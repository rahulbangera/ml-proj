import streamlit as st
import pandas as pd
import numpy as np
import kagglehub
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import scipy.stats as stats

st.set_page_config(
    page_title="HR Performance Predictor",
    page_icon="📊",
    layout="wide",
)

@st.cache_resource
def load_and_train():
    path = kagglehub.dataset_download("pavansubhasht/ibm-hr-analytics-attrition-dataset")
    df = pd.read_csv(path + "/WA_Fn-UseC_-HR-Employee-Attrition.csv")

    columns_to_drop = [
        'EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours',
        'PercentSalaryHike', 'DailyRate', 'HourlyRate', 'MonthlyRate'
    ]
    df.drop(columns=columns_to_drop, inplace=True, errors='ignore')

    categorical_cols = list(df.select_dtypes(include=['object']).columns)
    cat_options = {}
    for col in categorical_cols:
        cat_options[col] = sorted(df[col].unique().tolist())
    num_info = {}
    for col in df.select_dtypes(include=[np.number]).columns:
        if col == 'PerformanceRating':
            continue
        num_info[col] = {
            'min': int(df[col].min()),
            'max': int(df[col].max()),
            'median': int(df[col].median()),
        }

    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

    X = df.drop('PerformanceRating', axis=1)
    y = df['PerformanceRating']
    feature_names = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    train_df = pd.DataFrame(X_train, columns=X.columns)
    train_df['PerformanceRating'] = y_train.reset_index(drop=True)
    df_maj = train_df[train_df['PerformanceRating'] == 3]
    df_min = train_df[train_df['PerformanceRating'] == 4]
    df_min_up = resample(df_min, replace=True, n_samples=len(df_maj), random_state=42)
    df_res = pd.concat([df_maj, df_min_up]).sample(frac=1, random_state=42).reset_index(drop=True)
    X_tr = df_res.drop('PerformanceRating', axis=1)
    y_tr = df_res['PerformanceRating']

    z = np.abs(stats.zscore(X_tr))
    mask = (z < 3).all(axis=1)
    X_tr = X_tr[mask]
    y_tr = y_tr[mask]

    # Scale
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_test)

    # Train models
    model_defs = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM (RBF)': SVC(kernel='rbf', random_state=42, probability=True),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Naive Bayes': GaussianNB(),
        'KNN (k=5)': KNeighborsClassifier(n_neighbors=5),
        'Decision Tree': DecisionTreeClassifier(max_depth=5, random_state=42),
    }

    trained = {}
    for name, model in model_defs.items():
        model.fit(X_tr_s, y_tr)
        acc = accuracy_score(y_test, model.predict(X_te_s))
        trained[name] = {'model': model, 'accuracy': round(acc * 100, 2)}

    return trained, scaler, feature_names, categorical_cols, cat_options, num_info, label_encoders


with st.spinner("🔄 Training models... (first load only)"):
    models, scaler, feature_names, categorical_cols, cat_options, num_info, label_encoders = load_and_train()

st.title("HR Performance Rating Predictor")
st.caption("Adjust employee attributes and predict their performance rating using 7 ML models.")

st.divider()

st.sidebar.header("🤖 Select Model")

model_names = list(models.keys())
model_labels = [f"{name}  —  {models[name]['accuracy']}%" for name in model_names]
selected_idx = st.sidebar.radio(
    "Choose a model:",
    range(len(model_names)),
    format_func=lambda i: model_labels[i],
    index=0,
)
selected_model = model_names[selected_idx]

st.sidebar.divider()
st.sidebar.subheader("📈 Model Accuracies")
acc_df = pd.DataFrame({
    'Model': list(models.keys()),
    'Accuracy (%)': [m['accuracy'] for m in models.values()]
}).sort_values('Accuracy (%)', ascending=False).reset_index(drop=True)
st.sidebar.dataframe(acc_df, width=None, hide_index=True)

st.subheader("Employee Attributes")

col_count = 3
cols = st.columns(col_count)

inputs = {}
for i, feat in enumerate(feature_names):
    col = cols[i % col_count]
    with col:
        # Nice label
        nice_label = feat.replace('_', ' ')
        import re
        nice_label = re.sub(r'([a-z])([A-Z])', r'\1 \2', nice_label)

        if feat in categorical_cols:
            options = cat_options[feat]
            inputs[feat] = st.selectbox(nice_label, options, key=f"input_{feat}")
        else:
            info = num_info[feat]
            inputs[feat] = st.number_input(
                nice_label,
                min_value=info['min'],
                max_value=info['max'],
                value=info['median'],
                step=1,
                key=f"input_{feat}",
            )

st.divider()

# ---- Predict button ----
if st.button("🔮 Predict Performance Rating", type="primary", use_container_width=True):
    # Build feature vector
    row = []
    for feat in feature_names:
        val = inputs[feat]
        if feat in label_encoders:
            val = label_encoders[feat].transform([str(val)])[0]
        else:
            val = float(val)
        row.append(val)

    X_input = pd.DataFrame([row], columns=feature_names)
    X_input_scaled = scaler.transform(X_input)

    model_obj = models[selected_model]['model']
    prediction = int(model_obj.predict(X_input_scaled)[0])

    # Result display
    st.divider()
    st.subheader("📋 Prediction Result")

    r1, r2 = st.columns([1, 2])

    with r1:
        if prediction == 4:
            st.success(f"## Rating: {prediction}")
            st.markdown("### ⭐ Excellent")
        else:
            st.warning(f"## Rating: {prediction}")
            st.markdown("### 👍 Good")

    with r2:
        st.markdown(f"**Model:** {selected_model}")
        st.markdown(f"**Test Accuracy:** {models[selected_model]['accuracy']}%")

        # Probability bars
        if hasattr(model_obj, 'predict_proba'):
            probs = model_obj.predict_proba(X_input_scaled)[0]
            classes = model_obj.classes_

            st.markdown("**Confidence:**")
            for cls, prob in zip(classes, probs):
                label = "Excellent (4)" if int(cls) == 4 else "Good (3)"
                st.progress(prob, text=f"{label}: {prob:.1%}")
        else:
            st.info("Probability estimates not available for this model.")

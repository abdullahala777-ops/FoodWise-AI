"""
FoodWise AI - Food Waste Prediction App
Run with:  streamlit run main.py

Put main.py in the project root. The files are found automatically in the
root, Models/, Data/Cleaned/, or Data/Original/:
    preprocessing.joblib, linear_regression.joblib, ridge.joblib,
    random_forest.joblib, gradient_boosting.joblib, baseline_mean.joblib
Optional (more features when present):
    model_results.csv, test_ready.csv, Food_Waste_Prediction_Dataset (1).xlsx
"""
from datetime import date
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="FoodWise AI", page_icon="🌱", layout="wide")
sns.set_theme(style="whitegrid")

# ----------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------
BASE = Path(__file__).resolve().parent
SEARCH_DIRS = [
    BASE / "Notebooks" / "Models",
    BASE,
    BASE / "Models", BASE / "models",
    BASE / "Data" / "Cleaned", BASE / "data" / "cleaned",
    BASE / "Data" / "Original", BASE / "data" / "original",
]
PREP_FILE = "preprocessing.joblib"
RESULTS_FILE = "model_results.csv"
TEST_FILE = "test_ready.csv"
RAW_FILE = "Food_Waste_Prediction_Dataset (1).xlsx"
TARGET = "Waste_Amount_kg"
BASELINE = "Baseline (mean)"
MODEL_FILES = {
    "Linear Regression": "linear_regression.joblib",
    "Ridge": "ridge.joblib",
    "Random Forest": "random_forest.joblib",
    "Gradient Boosting": "gradient_boosting.joblib",
    BASELINE: "baseline_mean.joblib",
}
DEFAULT_RANGES = {
    "Prepared_Quantity": (50, 300),
    "Temperature_C": (22.0, 40.0),
    "Discount_Percent": (0, 25),
    "Customers": (10, 320),
}
UPLOAD_REQUIRED = [
    "Date", "City", "Cuisine", "Food_Item", "Weather",
    "Prepared_Quantity", "Temperature_C", "Discount_Percent", "Customers",
]

st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem;}
    .stMarkdown p, .stMarkdown li {direction: rtl; text-align: right;}
    [data-testid="stAlert"] {direction: rtl; text-align: right;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------------
# Loading (cached)
# ----------------------------------------------------------------------------
def find_file(name):
    """Look for a file in the root and in Models/, Data/Cleaned/, Data/Original/."""
    for folder in SEARCH_DIRS:
        path = folder / name
        if path.exists():
            return path
    return BASE / name



def load_artifacts():
    prep = joblib.load(find_file(PREP_FILE))
    models = {name: joblib.load(find_file(f)) for name, f in MODEL_FILES.items()}
    return prep, models


@st.cache_data(show_spinner=False)
def load_results():
    path = find_file(RESULTS_FILE)
    return pd.read_csv(path) if path.exists() else None


@st.cache_data(show_spinner=False)
def load_test():
    path = find_file(TEST_FILE)
    return pd.read_csv(path) if path.exists() else None


def raw_path():
    """The Excel file; also finds copies like 'Food_Waste_Prediction_Dataset.xlsx'."""
    path = find_file(RAW_FILE)
    if path.exists():
        return path
    for folder in SEARCH_DIRS:
        found = sorted(folder.glob("Food_Waste_Prediction_Dataset*.xlsx"))
        if found:
            return found[0]
    return path


@st.cache_data(show_spinner=False)
def load_raw():
    path = raw_path()
    if not path.exists():
        return None
    raw = pd.read_excel(path)
    raw["Date"] = pd.to_datetime(raw["Date"])
    raw["Month"] = raw["Date"].dt.month
    return raw


# ----------------------------------------------------------------------------
# Core logic: the SAME steps as Notebook 01, using what was learned on TRAIN
# ----------------------------------------------------------------------------
def prepare(raw_df, prep):
    """Raw columns -> scaled model input (same as Notebook 01, PART 10)."""
    num_cols, cat_cols = prep["num_cols"], prep["cat_cols"]
    X = raw_df[num_cols + cat_cols].copy()
    X[num_cols] = prep["num_imputer"].transform(X[num_cols])
    X[cat_cols] = prep["cat_imputer"].transform(X[cat_cols])
    cont = prep["cont_cols"]
    X[cont] = X[cont].clip(prep["clip_low"], prep["clip_high"], axis=1)
    dummies = prep["encoder"].transform(X[cat_cols])
    X = pd.concat([X.drop(columns=cat_cols), dummies], axis=1)
    X = X[prep["feature_columns"]]
    scaled = prep["scaler"].transform(X)
    return pd.DataFrame(scaled, columns=prep["feature_columns"], index=raw_df.index)


def predict_all(raw_df, prep, models):
    """Predictions of every model (waste cannot be negative, so we clip at 0)."""
    X = prepare(raw_df, prep)
    return pd.DataFrame(
        {name: np.clip(model.predict(X), 0, None) for name, model in models.items()},
        index=raw_df.index,
    )


def what_if(row, model, prep, column, values):
    """Prediction when ONE input changes and everything else stays the same."""
    grid = pd.DataFrame([row] * len(values))
    grid[column] = list(values)
    return np.clip(model.predict(prepare(grid, prep)), 0, None)


def to_flag(value):
    key = str(value).strip().lower()
    mapping = {"yes": 1, "no": 0, "1": 1, "0": 0, "true": 1, "false": 0, "1.0": 1, "0.0": 0}
    if key not in mapping:
        raise ValueError(f"قيمة غير مفهومة في Holiday: {value!r} (استخدم Yes أو No)")
    return mapping[key]


def build_raw_from_upload(df):
    """Uploaded table -> the raw columns the preprocessing expects."""
    missing = [c for c in UPLOAD_REQUIRED if c not in df.columns]
    if missing:
        raise ValueError("أعمدة ناقصة في الملف: " + ", ".join(missing))
    out = df[UPLOAD_REQUIRED].copy()
    dates = pd.to_datetime(out["Date"], errors="coerce")
    if dates.isna().any():
        raise ValueError(f"{int(dates.isna().sum())} صف فيها تاريخ غير صالح في عمود Date")
    out["month"] = dates.dt.month
    out["Weekend"] = (dates.dt.dayofweek >= 5).astype(int)
    out["Holiday"] = df["Holiday"].map(to_flag) if "Holiday" in df.columns else 0
    for col in ["Prepared_Quantity", "Temperature_C", "Discount_Percent", "Customers"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
        if out[col].isna().any():
            raise ValueError(f"عمود {col} فيه قيم غير رقمية أو فارغة")
    for col in ["City", "Cuisine", "Food_Item", "Weather"]:
        out[col] = out[col].astype(str).str.strip()
    return out


def show_fig(fig):
    st.pyplot(fig)
    plt.close(fig)


def get_ranges(raw):
    if raw is None:
        return DEFAULT_RANGES
    ranges = {}
    for col in DEFAULT_RANGES:
        low, high = raw[col].min(), raw[col].max()
        if col == "Temperature_C":
            ranges[col] = (float(low), float(high))
        else:
            ranges[col] = (int(low), int(high))
    return ranges


# ----------------------------------------------------------------------------
# Check the required files first
# ----------------------------------------------------------------------------
required = [PREP_FILE] + list(MODEL_FILES.values())
missing_required = [name for name in required if not find_file(name).exists()]
if missing_required:
    st.error("ملفات ناقصة: التطبيق ما يقدر يشتغل بدونها.")
    st.code("\n".join(missing_required))
    st.info(
        "شغّل Notebook 01 و Notebook 02 أول، وتأكد إن ملفات الحفظ "
        "(joblib) موجودة في مجلد Models أو بجنب main.py."
    )
    st.stop()

prep, models = load_artifacts()
from sklearn.utils.validation import check_is_fitted

for name, model in models.items():
    try:
        check_is_fitted(model)
        st.write(name, "مدرّب")
    except Exception:
        st.write(name, "غير مدرّب")
st.write("مسار النماذج:", find_file("ridge.joblib"))
results = load_results()
test_df = load_test()
raw = load_raw()
ranges = get_ranges(raw)
cats = {col: [str(x) for x in values] for col, values in zip(prep["cat_cols"], prep["encoder"].categories_)}
real_models = [name for name in models if name != BASELINE]

best_name = real_models[0]
if results is not None and {"Model", "RMSE"} <= set(results.columns):
    candidates = results[results["Model"].isin(real_models)]
    if len(candidates) > 0:
        best_name = candidates.sort_values("RMSE").iloc[0]["Model"]


def model_metrics(name):
    if results is None or name not in set(results["Model"]):
        return None
    return results[results["Model"] == name].iloc[0]


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    st.title("🌱 FoodWise AI")
    st.caption("نظام توقع النفايات الغذائية في المطاعم")
    chosen = st.selectbox(
        "اختر الموديل",
        real_models,
        index=real_models.index(best_name),
        help="الموديل الافتراضي هو الأفضل على بيانات الاختبار.",
    )
    chosen_model = models[chosen]
    metrics = model_metrics(chosen)
    if metrics is not None:
        st.caption("أداء الموديل على بيانات الاختبار")
        st.metric("MAE (متوسط الخطأ بالكجم)", f"{metrics['MAE']:.2f}")
        st.metric("RMSE", f"{metrics['RMSE']:.2f}")
        st.metric("R²", f"{metrics['R2']:.3f}")
    if chosen == best_name:
        st.success("هذا هو أفضل موديل حسب RMSE.")

st.title("🌱 FoodWise AI")
st.write("توقع كمية الهدر الغذائي (كجم) قبل تجهيز الأكل، وقارن بين الموديلات، واستكشف البيانات.")

tab_pred, tab_compare, tab_explore, tab_batch, tab_about = st.tabs(
    ["🔮 التنبؤ", "📊 مقارنة الموديلات", "🔎 استكشاف البيانات", "📁 تنبؤ جماعي", "ℹ️ عن المشروع"]
)

# ----------------------------------------------------------------------------
# Tab 1: single prediction
# ----------------------------------------------------------------------------
with tab_pred:
    st.subheader("أدخل تفاصيل اليوم")
    with st.form("predict_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            city = st.selectbox("المدينة (City)", cats["City"])
            cuisine = st.selectbox("نوع المطبخ (Cuisine)", cats["Cuisine"])
            food_item = st.selectbox("الوجبة (Food_Item)", cats["Food_Item"])
        with c2:
            weather = st.selectbox("الطقس (Weather)", cats["Weather"])
            day = st.date_input("التاريخ", value=date(2026, 6, 15))
            holiday = st.checkbox("اليوم إجازة رسمية (Holiday)", value=False)
        with c3:
            p_low, p_high = ranges["Prepared_Quantity"]
            c_low, c_high = ranges["Customers"]
            t_low, t_high = ranges["Temperature_C"]
            prepared = st.number_input(
                "الكمية المحضّرة (Prepared_Quantity)", min_value=0, value=int((p_low + p_high) / 2), step=1
            )
            customers = st.number_input(
                "عدد الزبائن المتوقع (Customers)", min_value=0, value=int((c_low + c_high) / 2), step=1
            )
            temperature = st.slider(
                "درجة الحرارة (Temperature_C)", min(t_low, 15.0), max(t_high, 45.0), float(round((t_low + t_high) / 2)), 0.5
            )
        discount_options = [0, 5, 10, 15, 20, 25]
        if raw is not None:
            discount_options = sorted(int(x) for x in raw["Discount_Percent"].unique())
        discount = st.select_slider("نسبة الخصم % (Discount_Percent)", options=discount_options, value=discount_options[0])
        submitted = st.form_submit_button("احسب التوقع")

    if submitted:
        st.session_state["last_input"] = {
            "City": city,
            "Cuisine": cuisine,
            "Food_Item": food_item,
            "Prepared_Quantity": float(prepared),
            "Temperature_C": float(temperature),
            "Weather": weather,
            "Weekend": 1 if day.weekday() >= 5 else 0,
            "Holiday": 1 if holiday else 0,
            "Discount_Percent": float(discount),
            "Customers": float(customers),
            "month": int(day.month),
        }

    row = st.session_state.get("last_input")
    if row is None:
        st.info("عبّي النموذج واضغط «احسب التوقع».")
    else:
        row_df = pd.DataFrame([row])
        preds = predict_all(row_df, prep, models).iloc[0]
        value = float(preds[chosen])

        for col, label in [("Prepared_Quantity", "الكمية المحضّرة"), ("Customers", "عدد الزبائن"), ("Temperature_C", "درجة الحرارة")]:
            low, high = ranges[col]
            if row[col] < low or row[col] > high:
                st.warning(f"{label} ({row[col]:g}) خارج النطاق اللي تدرب عليه الموديل ({low:g} إلى {high:g}). النتيجة أقل موثوقية.")

        m = model_metrics(chosen)
        k1, k2, k3 = st.columns(3)
        k1.metric(f"الهدر المتوقع ({chosen})", f"{value:.2f} كجم")
        if m is not None:
            k2.metric("هامش الخطأ المعتاد (MAE)", f"± {m['MAE']:.2f} كجم")
            k3.metric("المدى المتوقع", f"{max(0.0, value - m['MAE']):.1f} إلى {value + m['MAE']:.1f} كجم")
        if raw is not None:
            avg = float(raw[TARGET].mean())
            diff = value - avg
            word = "أعلى" if diff >= 0 else "أقل"
            st.write(f"المتوسط العام في البيانات {avg:.2f} كجم، والتوقع {word} منه بـ {abs(diff):.2f} كجم.")
        if row["Weekend"] == 1:
            st.caption("التاريخ المختار يوم عطلة نهاية أسبوع (سبت أو أحد).")

        st.markdown("#### مقارنة كل الموديلات لنفس المدخلات")
        comp = pd.DataFrame({"Model": preds.index, "Predicted_kg": preds.values.round(2)})
        left, right = st.columns([1, 2])
        with left:
            st.dataframe(comp)
            spread = comp[comp["Model"] != BASELINE]["Predicted_kg"]
            st.caption(f"الموديلات الأربعة تتراوح بين {spread.min():.2f} و {spread.max():.2f} كجم.")
        with right:
            fig, ax = plt.subplots(figsize=(7, 3.2))
            colors = ["#9e9e9e" if n == BASELINE else ("#d9534f" if n == chosen else "#4a90c2") for n in comp["Model"]]
            ax.barh(comp["Model"], comp["Predicted_kg"], color=colors)
            ax.set_xlabel("Predicted waste (kg)")
            ax.set_title("Same input, every model")
            for i, v in enumerate(comp["Predicted_kg"]):
                ax.text(v, i, f" {v:.2f}", va="center")
            ax.set_xlim(0, max(comp["Predicted_kg"].max() * 1.2, 1))
            fig.tight_layout()
            show_fig(fig)

        st.markdown("#### ماذا لو غيّرت المدخلات؟ (What-if)")
        w1, w2 = st.columns(2)
        with w1:
            low, high = ranges["Prepared_Quantity"]
            qty = np.linspace(low, high, 30)
            curve = what_if(row, chosen_model, prep, "Prepared_Quantity", qty)
            fig, ax = plt.subplots(figsize=(6, 3.5))
            ax.plot(qty, curve, linewidth=2)
            ax.scatter([row["Prepared_Quantity"]], [value], color="red", zorder=3, label="Your input")
            ax.set_xlabel("Prepared_Quantity")
            ax.set_ylabel("Predicted waste (kg)")
            ax.set_title("Waste vs prepared quantity")
            ax.legend()
            fig.tight_layout()
            show_fig(fig)
        with w2:
            disc_values = sorted(discount_options)
            curve = what_if(row, chosen_model, prep, "Discount_Percent", disc_values)
            fig, ax = plt.subplots(figsize=(6, 3.5))
            ax.plot(disc_values, curve, marker="o", linewidth=2)
            ax.scatter([row["Discount_Percent"]], [value], color="red", zorder=3, label="Your input")
            ax.set_xlabel("Discount_Percent")
            ax.set_ylabel("Predicted waste (kg)")
            ax.set_title("Waste vs discount")
            ax.legend()
            fig.tight_layout()
            show_fig(fig)

        saving = []
        for pct in (10, 20):
            smaller = what_if(row, chosen_model, prep, "Prepared_Quantity", [row["Prepared_Quantity"] * (1 - pct / 100)])[0]
            saving.append((pct, value - float(smaller)))
        st.write(
            "لو قلّلت الكمية المحضّرة: "
            + "، ".join(f"بنسبة {pct}% يتغير الهدر المتوقع بـ {-s:+.2f} كجم" for pct, s in saving)
            + ". (هذا توقع من الموديل، مو ضمان.)"
        )

# ----------------------------------------------------------------------------
# Tab 2: model comparison
# ----------------------------------------------------------------------------
with tab_compare:
    st.subheader("مقارنة الموديلات على بيانات الاختبار")
    if results is None:
        st.warning(f"ملف {RESULTS_FILE} غير موجود في المجلد.")
    else:
        table = results.sort_values("RMSE").reset_index(drop=True)
        st.dataframe(table)
        st.success(f"أفضل موديل: {best_name} (أقل RMSE بين الموديلات الأربعة).")
        fig, axes = plt.subplots(1, 3, figsize=(15, 3.6))
        for ax, metric in zip(axes, ["MAE", "RMSE", "R2"]):
            sns.barplot(x=table[metric], y=table["Model"], color="#4a90c2", ax=ax)
            ax.set_title(metric)
            ax.set_ylabel("")
        fig.tight_layout()
        show_fig(fig)
        st.caption("MAE و RMSE كل ما قلّ أحسن (بالكجم). R² كل ما زاد أحسن. موديل Baseline يتنبأ بالمتوسط دائماً، وهو مرجع للمقارنة فقط.")

    st.subheader(f"تفاصيل الموديل المختار: {chosen}")
    if test_df is None or TARGET not in test_df.columns:
        st.warning(f"ملف {TEST_FILE} غير موجود في المجلد.")
    else:
        X_test = test_df.drop(columns=[TARGET])[prep["feature_columns"]]
        y_test = test_df[TARGET]
        pred = chosen_model.predict(X_test)
        a, b, c = st.columns(3)
        a.metric("MAE", f"{mean_absolute_error(y_test, pred):.3f}")
        b.metric("RMSE", f"{np.sqrt(mean_squared_error(y_test, pred)):.3f}")
        c.metric("R²", f"{r2_score(y_test, pred):.3f}")

        fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
        axes[0].scatter(y_test, pred, alpha=0.5, edgecolor="k")
        lims = [min(y_test.min(), pred.min()), max(y_test.max(), pred.max())]
        axes[0].plot(lims, lims, "r--")
        axes[0].set_title("Actual vs Predicted")
        axes[0].set_xlabel("Actual (kg)")
        axes[0].set_ylabel("Predicted (kg)")
        sns.histplot(y_test - pred, kde=True, bins=20, ax=axes[1])
        axes[1].axvline(0, color="red", linestyle="--")
        axes[1].set_title("Residuals (Actual - Predicted)")
        axes[1].set_xlabel("Error (kg)")
        fig.tight_layout()
        show_fig(fig)

    if hasattr(chosen_model, "feature_importances_"):
        importance = pd.Series(chosen_model.feature_importances_, index=prep["feature_columns"])
        top = importance.sort_values(ascending=False).head(10)
        colors = "#4a90c2"
        title = "Top 10 feature importances"
    else:
        importance = pd.Series(chosen_model.coef_, index=prep["feature_columns"])
        top = importance.reindex(importance.abs().sort_values(ascending=False).head(10).index)
        colors = ["#4a90c2" if v >= 0 else "#d9534f" for v in top.values]
        title = "Top 10 coefficients (blue = raises waste, red = lowers waste)"
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(top.index[::-1], top.values[::-1], color=colors if isinstance(colors, str) else colors[::-1])
    ax.set_title(title)
    fig.tight_layout()
    show_fig(fig)

# ----------------------------------------------------------------------------
# Tab 3: data explorer
# ----------------------------------------------------------------------------
with tab_explore:
    st.subheader("استكشاف البيانات الأصلية")
    if raw is None:
        st.warning(f"ضع ملف الإكسل «{RAW_FILE}» في مجلد Data/Original عشان يشتغل هذا القسم.")
    else:
        f1, f2, f3 = st.columns(3)
        sel_city = f1.multiselect("المدينة", sorted(raw["City"].unique()), default=sorted(raw["City"].unique()))
        sel_weather = f2.multiselect("الطقس", sorted(raw["Weather"].unique()), default=sorted(raw["Weather"].unique()))
        sel_cuisine = f3.multiselect("المطبخ", sorted(raw["Cuisine"].unique()), default=sorted(raw["Cuisine"].unique()))
        months = st.slider("الأشهر", 1, 12, (1, 12))
        view = raw[
            raw["City"].isin(sel_city)
            & raw["Weather"].isin(sel_weather)
            & raw["Cuisine"].isin(sel_cuisine)
            & raw["Month"].between(months[0], months[1])
        ]
        if len(view) == 0:
            st.warning("ما فيه صفوف تطابق الفلاتر.")
        else:
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("عدد الصفوف", f"{len(view):,}")
            k2.metric("متوسط الهدر (كجم)", f"{view[TARGET].mean():.2f}")
            k3.metric("إجمالي الهدر (كجم)", f"{view[TARGET].sum():,.0f}")
            k4.metric("أيام بدون هدر", f"{(view[TARGET] == 0).mean():.1%}")

            g1, g2 = st.columns(2)
            with g1:
                by_food = view.groupby("Food_Item")[TARGET].mean().sort_values()
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.barh(by_food.index, by_food.values, color="#4a90c2")
                ax.set_title("Average waste by food item (kg)")
                fig.tight_layout()
                show_fig(fig)
            with g2:
                monthly = view.groupby("Month")[TARGET].agg(["mean", "sum"])
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.plot(monthly.index, monthly["mean"], marker="o", linewidth=2)
                ax.set_title("Average waste per month (kg)")
                ax.set_xlabel("Month")
                fig.tight_layout()
                show_fig(fig)

            g3, g4 = st.columns(2)
            with g3:
                by_weather = view.groupby("Weather")[TARGET].mean().sort_values()
                fig, ax = plt.subplots(figsize=(6, 3.6))
                ax.bar(by_weather.index, by_weather.values, color="#4a90c2")
                ax.set_title("Average waste by weather (kg)")
                fig.tight_layout()
                show_fig(fig)
            with g4:
                corr = view.select_dtypes(include="number").drop(columns=["Month"]).corr()
                fig, ax = plt.subplots(figsize=(6, 4.6))
                sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax, annot_kws={"size": 7})
                ax.set_title("Correlation heatmap")
                fig.tight_layout()
                show_fig(fig)

            st.caption("ملاحظة: Remaining_Quantity مرتبط بالهدر بدرجة 0.95، لكنه معلومة ما نعرفها قبل التجهيز، فاستبعدناه من الموديلات (تسريب هدف).")
            st.dataframe(view.head(200))
            st.download_button(
                "تحميل البيانات المفلترة (CSV)",
                view.to_csv(index=False).encode("utf-8-sig"),
                file_name="filtered_food_waste.csv",
                mime="text/csv",
            )

# ----------------------------------------------------------------------------
# Tab 4: batch prediction
# ----------------------------------------------------------------------------
with tab_batch:
    st.subheader("تنبؤ جماعي من ملف")
    st.write(
        "ارفع ملف CSV أو Excel فيه الأعمدة: " + ", ".join(UPLOAD_REQUIRED)
        + ". وعمود Holiday اختياري (Yes أو No). عطلة نهاية الأسبوع والشهر تُحسب من التاريخ."
    )
    template = pd.DataFrame(
        {
            "Date": ["2026-06-15", "2026-06-20"],
            "City": [cats["City"][0]] * 2,
            "Cuisine": [cats["Cuisine"][0]] * 2,
            "Food_Item": [cats["Food_Item"][0]] * 2,
            "Weather": [cats["Weather"][0]] * 2,
            "Prepared_Quantity": [150, 200],
            "Temperature_C": [30, 34],
            "Discount_Percent": [10, 0],
            "Customers": [100, 140],
            "Holiday": ["No", "Yes"],
        }
    )
    st.download_button(
        "تحميل قالب جاهز (CSV)",
        template.to_csv(index=False).encode("utf-8-sig"),
        file_name="prediction_template.csv",
        mime="text/csv",
    )
    uploaded = st.file_uploader("ارفع الملف", type=["csv", "xlsx"])
    if uploaded is not None:
        try:
            data = pd.read_excel(uploaded) if uploaded.name.lower().endswith("xlsx") else pd.read_csv(uploaded)
            batch_raw = build_raw_from_upload(data)
            unknown = {
                col: sorted(set(batch_raw[col]) - set(cats[col]))
                for col in ["City", "Cuisine", "Food_Item", "Weather"]
                if set(batch_raw[col]) - set(cats[col])
            }
            if unknown:
                st.warning("قيم جديدة ما شافها الموديل (تُعامل كقيمة غير معروفة): " + str(unknown))
            batch_preds = predict_all(batch_raw, prep, models)
            out = data.copy()
            out["Predicted_Waste_kg"] = batch_preds[chosen].round(2).values
            for name in real_models:
                out[f"Pred_{name.replace(' ', '_')}"] = batch_preds[name].round(2).values
            st.success(f"تم التنبؤ لـ {len(out)} صف بالموديل: {chosen}")
            st.dataframe(out.head(200))
            if TARGET in data.columns:
                actual = pd.to_numeric(data[TARGET], errors="coerce")
                ok = actual.notna()
                if ok.sum() > 1:
                    st.markdown("#### مقارنة بالقيم الحقيقية الموجودة في ملفك")
                    q1, q2, q3 = st.columns(3)
                    q1.metric("MAE", f"{mean_absolute_error(actual[ok], batch_preds[chosen][ok]):.3f}")
                    q2.metric("RMSE", f"{np.sqrt(mean_squared_error(actual[ok], batch_preds[chosen][ok])):.3f}")
                    q3.metric("R²", f"{r2_score(actual[ok], batch_preds[chosen][ok]):.3f}")
                    st.caption("لو الملف يحتوي بيانات التدريب نفسها، النتيجة أفضل من الواقع.")
            st.download_button(
                "تحميل النتائج (CSV)",
                out.to_csv(index=False).encode("utf-8-sig"),
                file_name="predictions.csv",
                mime="text/csv",
            )
        except Exception as error:
            st.error(f"ما قدرت أعالج الملف: {error}")

# -------------------------------------------------
# Tab 5: about
# -------------------------------------------------
with tab_about:
    st.subheader("عن المشروع")
    st.markdown(
        """
**الهدف:** توقع كمية الهدر الغذائي `Waste_Amount_kg` لكل مطعم في كل يوم، عشان نحضّر كمية أنسب من الأكل.

**كيف بُني:**
1. **Notebook 01:** فحص البيانات وتنظيفها وEDA، ثم تقسيم train/test، ثم imputer وclip وone-hot وscaler (كلها تتعلم من التدريب فقط).
2. **Notebook 02:** تدريب 4 موديلات (Linear Regression وRidge وRandom Forest وGradient Boosting) مع cross-validation وضبط المعاملات، وتقييمها بـ MAE وRMSE وR².
3. **هذا التطبيق:** يحمّل الموديلات المحفوظة وأدوات التجهيز نفسها، فمدخلاتك تتجهّز بنفس الطريقة بالضبط.

**قرارات مهمة:**
- حذفنا `Remaining_Quantity` و`Sold_Quantity` لأنهم يكشفون الهدر (تسريب هدف)، فالنتائج أقل لكنها صادقة.
- الأعمدة `Restaurant_ID` و`Restaurant_Name` تكرر `City`، فاستبعدناه.
- `Weekend` يُحسب من التاريخ (السبت والأحد)، و`month` من التاريخ.

**حدود المشروع:**
- البيانات سنة واحدة (2025) وخمسة مطاعم، فالتوقع خارج هذا النطاق أقل موثوقية.
- `Customers` ما يُعرف بدقة قبل اليوم، فاستخدمه كعدد متوقع.
- التقسيم عشوائي، والتقسيم الزمني أنسب للتنبؤ بالمستقبل.
"""
    )
    with st.expander("الملفات المستخدمة"):
        files = [PREP_FILE] + list(MODEL_FILES.values()) + [RESULTS_FILE, TEST_FILE, RAW_FILE]
        st.dataframe(pd.DataFrame({"File": files, "Found": [find_file(f).exists() or (f == RAW_FILE and raw_path().exists()) for f in files]}))

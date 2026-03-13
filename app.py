import streamlit as st
import pandas as pd
from weather_model import (
    load_weather_data,
    prepare_dataset,
    train_model,
    predict_next_day,
    save_to_csv
)

st.set_page_config(page_title="Прогноз опадів", layout="centered")

st.title("Прогноз опадів на основі Open-Meteo")

st.write("Міні-сервіс для завантаження метеоданих, навчання моделі та прогнозу опадів.")

st.subheader("1. Параметри")
latitude = st.number_input("Latitude", value=50.45, format="%.4f")
longitude = st.number_input("Longitude", value=30.52, format="%.4f")
past_days = st.slider("Кількість минулих днів для даних", min_value=30, max_value=180, value=90)
forecast_days = st.slider("Кількість днів прогнозу", min_value=1, max_value=7, value=3)

if "df" not in st.session_state:
    st.session_state.df = None

if "model" not in st.session_state:
    st.session_state.model = None

if "feature_cols" not in st.session_state:
    st.session_state.feature_cols = None

st.subheader("2. Отримання даних")

if st.button("Отримати дані з Open-Meteo"):
    try:
        df = load_weather_data(latitude, longitude, past_days, forecast_days)
        save_to_csv(df)
        st.session_state.df = df
        st.success("Дані успішно завантажено та збережено у weather_daily.csv")
        st.dataframe(df.tail(10))
    except Exception as e:
        st.error(f"Помилка під час завантаження даних: {e}")

uploaded_file = st.file_uploader("Або завантажте CSV-файл", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.session_state.df = df
        st.success("CSV успішно завантажено")
        st.dataframe(df.head())
    except Exception as e:
        st.error(f"Не вдалося прочитати CSV: {e}")

st.subheader("3. Навчання моделі")

if st.button("Навчити модель"):
    if st.session_state.df is None:
        st.warning("Спочатку отримайте або завантажте дані.")
    else:
        try:
            prepared_df, X, y, feature_cols = prepare_dataset(st.session_state.df)
            model, metrics = train_model(X, y)

            st.session_state.model = model
            st.session_state.feature_cols = feature_cols
            st.session_state.prepared_df = prepared_df

            st.success("Модель навчено")
            st.write(f"Accuracy: {metrics['accuracy']:.4f}")
            st.text("Classification report:")
            st.text(metrics["classification_report"])
            st.text("Confusion matrix:")
            st.write(metrics["confusion_matrix"])
        except Exception as e:
            st.error(f"Помилка під час навчання: {e}")

st.subheader("4. Прогноз")

if st.button("Зробити прогноз"):
    if st.session_state.model is None or st.session_state.df is None:
        st.warning("Спочатку завантажте дані і навчіть модель.")
    else:
        try:
            pred_class, pred_prob = predict_next_day(
                st.session_state.model,
                st.session_state.df,
                st.session_state.feature_cols
            )

            if pred_class == 1:
                st.success(f"Очікуються опади. Ймовірність опадів = {pred_prob:.2%}")
            else:
                st.info(f"Опадів не очікується. Ймовірність опадів = {pred_prob:.2%}")
        except Exception as e:
            st.error(f"Помилка під час прогнозу: {e}")

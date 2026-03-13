import streamlit as st
import pandas as pd
import requests
from datetime import datetime, timedelta, date
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

МІСТА = {
    "Київ": (50.45, 30.52),
    "Харків": (49.99, 36.23),
    "Одеса": (46.48, 30.72),
    "Дніпро": (48.46, 35.04),
    "Запоріжжя": (47.84, 35.14),
    "Львів": (49.84, 24.03),
    "Вінниця": (49.23, 28.47),
    "Полтава": (49.59, 34.55),
    "Чернівці": (48.29, 25.94),
    "Ужгород": (48.62, 22.30),
    "Інше (вручну)": (50.45, 30.52),
}

# Ознаки для навчання моделі
FEATURES = [
    "temperature_2m_max",
    "temperature_2m_min",
    "windspeed_10m_max",
    "rain_sum",
    "relative_humidity_2m_max",
    "surface_pressure_mean",
]

CSV_FILE = "weather_daily.csv"

# 1. ФУНКЦІЯ ОТРИМАННЯ ДАНИХ (API)
def get_weather_data(lat, lon, start: date, end: date):
    """Запит до Open-Meteo Archive API та повернення DataFrame."""
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start.strftime("%Y-%m-%d"),
        "end_date": end.strftime("%Y-%m-%d"),
        "daily": [
            "precipitation_sum",  # обов'язкове — сума опадів
            "rain_sum",  # обов'язкове — сума дощу
            "temperature_2m_max",  # макс. температура
            "temperature_2m_min",  # мін. температура
            "windspeed_10m_max",  # швидкість вітру
            "relative_humidity_2m_max",  # вологість
            "surface_pressure_mean",  # атмосферний тиск
        ],
        "timezone": "Europe/Kyiv"
    }

    try:
        response = requests.get(url, params=params, timeout=15)
        data = response.json()["daily"]
        df = pd.DataFrame(data)
        df.rename(columns={"time": "date"}, inplace=True)
        df["date"] = pd.to_datetime(df["date"])
        df.fillna(0, inplace=True)

        # Цільова змінна: 1 — є опади, 0 — немає
        df["target"] = (df["precipitation_sum"] > 0).astype(int)

        # Зберігаємо у CSV
        df.to_csv(CSV_FILE, index=False, encoding="utf-8")
        return df
    except Exception as e:
        st.error(f"Помилка завантаження даних: {e}")
        return None


# 2. ІНТЕРФЕЙС STREAMLIT
st.set_page_config(page_title="Weather ML Forecast", page_icon="🌧️", layout="wide")
st.title("🌦️ Прогноз опадів (ML класифікація)")

# Бокова панель налаштувань
st.sidebar.header("📍 Геопозиція та Дані")

# Вибір міста зі списку
місто = st.sidebar.selectbox("🏙️ Оберіть місто", list(МІСТА.keys()))
lat_default, lon_default = МІСТА[місто]

# Якщо "Інше" — показуємо ручне введення координат
if місто == "Інше (вручну)":
    lat = st.sidebar.number_input("Широта (Latitude)", value=lat_default, format="%.4f")
    lon = st.sidebar.number_input("Довгота (Longitude)", value=lon_default, format="%.4f")
else:
    lat, lon = lat_default, lon_default
    st.sidebar.caption(f"📌 Координати: {lat}, {lon}")

st.sidebar.divider()

# Вибір дати
st.sidebar.subheader("📅 Період даних")
end_default = date.today() - timedelta(days=1)
start_default = end_default - timedelta(days=364)

start_date = st.sidebar.date_input(
    "Початок періоду",
    value=start_default,
    max_value=end_default - timedelta(days=60),
)
end_date = st.sidebar.date_input(
    "Кінець періоду",
    value=end_default,
    min_value=start_date + timedelta(days=60),
    max_value=end_default,
)

# Підрахунок і попередження про кількість днів
days_count = (end_date - start_date).days
st.sidebar.caption(f"📆 Обрано: **{days_count} днів**")

if days_count < 80:
    st.sidebar.warning(f"⚠️ Лише {days_count} днів — рекомендується мінімум 80 для точного прогнозу!")
elif days_count < 180:
    st.sidebar.info("💡 Для кращої точності рекомендується 180+ днів.")
else:
    st.sidebar.success(f"✅ {days_count} днів — чудово для навчання!")

st.sidebar.divider()

# Кнопка отримання даних
if st.button("🔌 Отримати дані та Навчити модель", type="primary"):
    with st.spinner("Завантаження даних та навчання..."):
        df = get_weather_data(lat, lon, start_date, end_date)
        if df is not None:
            st.session_state["df"] = df

            # Підготовка даних для ML
            X = df[FEATURES]
            y = df["target"]

            # Навчання моделі
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)

            # Оцінка якості
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred)

            st.session_state["model"] = model
            st.session_state["accuracy"] = acc
            st.session_state["report"] = report

            st.success(f"✅ Дані отримано ({len(df)} днів)! Модель навчено. Збережено у `{CSV_FILE}`")

# Вивід метрик
if "df" in st.session_state:
    df = st.session_state["df"]

    st.divider()

    # Статистика у верхніх плашках
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📅 Всього днів", len(df))
    col2.metric("🌧 Днів з опадами", int(df["target"].sum()))
    col3.metric("☀️ Без опадів", int((df["target"] == 0).sum()))
    col4.metric("💧 Середні опади", f"{df['precipitation_sum'].mean():.1f} мм")

    # Графік температур на всю ширину
    st.subheader("📈 Тренд температур за вибраний період")
    st.line_chart(df.set_index("date")[["temperature_2m_max", "temperature_2m_min"]], height=300)

    # Таблиця з даними
    st.subheader("📋 Останні дані з бази")
    st.dataframe(
        df[["date", "precipitation_sum", "rain_sum",
            "temperature_2m_max", "temperature_2m_min",
            "windspeed_10m_max", "relative_humidity_2m_max",
            "surface_pressure_mean", "target"]].tail(10),
        use_container_width=True,
    )

    # Показники моделі
    st.subheader("📊 Показники моделі")
    st.metric("Точність (Accuracy)", f"{st.session_state['accuracy']:.2%}")
    with st.expander("📄 Детальний звіт класифікації"):
        st.code(st.session_state["report"])

# 3. БЛОК ПРОГНОЗУ
if "model" in st.session_state:
    st.divider()
    st.subheader("🔮 Зробити прогноз на обраний день")

    df = st.session_state["df"]
    вкладки = st.tabs(["📅 Обрана дата з датасету", "🔜 Наступний день"])

    # Вкладка 1: вибір дати зі списку — модель сама бере дані
    with вкладки[0]:
        дати = df["date"].dt.strftime("%Y-%m-%d").tolist()
        обрана = st.selectbox("Оберіть дату:", дати, index=len(дати) - 1)

        if st.button("🌦️ Прогноз для обраної дати"):
            рядок = df[df["date"] == pd.to_datetime(обрана)]
            if not рядок.empty:
                X_row = рядок[FEATURES].values
                prediction = st.session_state["model"].predict(X_row)[0]
                prob = st.session_state["model"].predict_proba(X_row)[0][1]

                st.write(f"### 📅 Прогноз на {обрана}:")
                if prediction == 1:
                    st.error(f"🌧️ Опади ОЧІКУЮТЬСЯ  (Ймовірність: {prob:.1%})")
                else:
                    st.success(f"☀️ Опадів НЕ очікується  (Ймовірність: {prob:.1%})")

                # Реальний результат для перевірки
                реальне = "🌧 Опади були" if рядок.iloc[0]["target"] == 1 else "☀️ Опадів не було"
                st.info(f"📊 Реальний результат: **{реальне}** "
                        f"(опади: {рядок.iloc[0]['precipitation_sum']:.1f} мм)")

    # Вкладка 2: прогноз на наступний день (автоматично, без ручного вводу)
    with вкладки[1]:
        останній = df.iloc[-1]
        наступна = (pd.to_datetime(останній["date"]) + timedelta(days=1)).strftime("%Y-%m-%d")

        st.caption(
            f"Модель використовує дані за **{останній['date'].strftime('%Y-%m-%d')}** "
            f"щоб передбачити **{наступна}**"
        )

        with st.expander("👀 Вхідні дані для прогнозу"):
            st.json({f: round(float(останній[f]), 2) for f in FEATURES})

        if st.button("🔜 Прогноз на наступний день"):
            X_next = pd.DataFrame([{f: останній[f] for f in FEATURES}])
            prediction = st.session_state["model"].predict(X_next)[0]
            prob = st.session_state["model"].predict_proba(X_next)[0][1]

            st.write(f"### 📅 Прогноз на {наступна}:")
            if prediction == 1:
                st.error(f"🌧️ Опади ОЧІКУЮТЬСЯ  (Ймовірність: {prob:.1%})")
            else:
                st.success(f"☀️ Опадів НЕ очікується  (Ймовірність: {prob:.1%})")

st.caption("Дані: Open-Meteo Archive API | Модель: Random Forest | scikit-learn + Streamlit")

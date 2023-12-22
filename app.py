import plotly.express as px
import scipy.stats
import streamlit as st

from utils.data_utils import join_tables


MAIN_IMAGE_LINK = """
https://img.freepik.com/premium-vector/bank-building-and-money-bank-
financing-money-exchange-financial-services-atm-giving-out-money_625536-194.jpg
"""

DATA = st.cache_data(join_tables)(dropna=True)
DATA_PLOT = st.cache_data(join_tables)(dropna=True, drop_outliers=True)

ID_COLUMNS = ["ID_CLIENT", "AGREEMENT_RK"]
CATEGORICAL_COLUMNS = DATA.select_dtypes(exclude="number").columns.tolist()
NUMERIC_COLUMNS = DATA.columns.difference(CATEGORICAL_COLUMNS + ID_COLUMNS).tolist()
FEATURE_COLUMNS = DATA.columns.difference(ID_COLUMNS).tolist()

CORRELATION_FN = {
    "Коэффицент Пирсона": scipy.stats.pearsonr,
    "Коэффицент Спирмена": scipy.stats.spearmanr,
    "Коэффицент Кендалла": scipy.stats.kendalltau,
}

DASHBOARD_DESCRIPTION = """\
    ℹ️ В данном дашборде реализован разведочный анализ данных о клиентах банка.
"""

FEATURE_DESCRIPTION = {
    "ID_CLIENT": "идентификатор записи",
    "AGE": "возраст клиента",
    "GENDER": "пол клиента (1 — мужчина, 0 — женщина)",
    "EDUCATION": "образование",
    "MARITAL_STATUS": "семейное положение",
    "CHILD_TOTAL": "количество детей клиента",
    "DEPENDANTS": "количество иждивенцев клиента",
    "SOCSTATUS_WORK_FL": "социальный статус клиента относительно работы (1 — работает, 0 — не работает)",
    "SOCSTATUS_PENS_FL": "социальный статус клиента относительно пенсии (1 — пенсионер, 0 — не пенсионер)",
    "SOCSTATUS_WORK_DESC": "расшифровка статуса работы",
    "SOCSTATUS_PENS_DESC": "расшифровка статуса пенсии",
    "REG_ADDRESS_PROVINCE": "область регистрации клиента",
    "FACT_ADDRESS_PROVINCE": "область фактического пребывания клиента",
    "POSTAL_ADDRESS_PROVINCE": "почтовый адрес области",
    "FL_PRESENCE_FL": "флаг наличия в собственности квартиры",
    "OWN_AUTO": "количество автомобилей в собственности",
    "GEN_INDUSTRY": "отрасль работы клиента",
    "GEN_TITLE": "должность",
    "JOB_DIR": "направление деятельности внутри компании",
    "WORK_TIME": "время работы на текущем месте (в месяцах)",
    "FAMILY_INCOME": "семейный доход (несколько категорий)",
    "CREDIT": "сумма последнего кредита клиента (в рублях)",
    "TERM": "срок кредита",
    "FST_PAYMENT": "первоначальный взнос (в рублях)",
    "PERSONAL_INCOME": "личный доход клиента (в рублях)",
    "LOAN_NUM_TOTAL": "количество ссуд клиента",
    "LOAN_NUM_CLOSED": "количество погашенных ссуд клиента",
    "AGREEMENT_RK": "уникальный идентификатор объекта в выборке",
    "TARGET": "целевая переменная: флаг отклика на маркетинговую кампанию",
}


def _get_str_description(feature_dict: dict = FEATURE_DESCRIPTION):
    return "\n".join(map(lambda x: f"* **{x[0]}** — {x[1]}", feature_dict.items()))


def show_data_description():
    st.subheader("👀 Посмотрим на данные")
    if st.button("🎲 Сгенерировать новую подвыборку"):
        st.dataframe(DATA.sample(5), hide_index=True)
    else:
        st.dataframe(DATA.sample(5), hide_index=True)
    
    if st.checkbox("🔎 Показать описание признаков", value=False):
        st.info(_get_str_description())


def show_distribution_section():
    st.info("В данной секции можно увидеть распределения признаков 💹")

    st.sidebar.subheader("Распределения")
    if st.sidebar.checkbox("📈 Распределение числовых признаков", value=True):
        st.subheader("📈 Распределение числовых признаков")
        numeric_feature = st.selectbox(
            "Выберите числовой признак", NUMERIC_COLUMNS
        )
        st.info(f"{numeric_feature} — {FEATURE_DESCRIPTION[numeric_feature]}")

        if st.checkbox("🎯 Добавить зависимость от таргета для числового признака", value=False):
            extra_params = {"color": "TARGET"}
        else:
            extra_params = {}

        fig = px.histogram(
            DATA_PLOT,
            x=numeric_feature,
            histnorm="probability",
            title=f"Гистограмма значений признака {numeric_feature}",
            marginal="violin",
            **extra_params,
        )
        st.plotly_chart(fig)

    if st.sidebar.checkbox("📊 Распределение категориальных признаков", value=True):
        st.subheader("📊 Распределение категориальных признаков")
        cat_feature = st.selectbox(
            "Выберите категориальный признак", CATEGORICAL_COLUMNS
        )
        st.info(f"{cat_feature} — {FEATURE_DESCRIPTION[cat_feature]}")

        if st.checkbox("🎯 Добавить зависимость от таргета для категориального признака", value=False):
            extra_params = {"color": "TARGET", "barmode": "group"}
        else:
            extra_params = {}

        fig = px.histogram(
            DATA_PLOT,
            x=cat_feature,
            histnorm="probability",
            title=f"Распеделение значений признака {cat_feature}",
            **extra_params,
        )
        st.plotly_chart(fig)


    if st.sidebar.checkbox("👬 Попарные распределения признаков", value=True):
        st.subheader("👬 Попарные распределения признаков")
        choice_1, choice_2 = st.columns(2)
        with choice_1:
            first_feat = st.selectbox("Выберите первый признак", FEATURE_COLUMNS)
        with choice_2:
            second_feat = st.selectbox("Выберите второй признак", set(FEATURE_COLUMNS) - set([first_feat]))

        st.info(
            f"{first_feat} — {FEATURE_DESCRIPTION[first_feat]}\n\n{second_feat} — {FEATURE_DESCRIPTION[second_feat]}"
        )

        if st.checkbox("🎯 Добавить зависимость от таргета для scatter графика", value=False):
            extra_params = {"color": "TARGET"}
        else:
            extra_params = {}

        fig = px.scatter(
            DATA_PLOT,
            x=first_feat,
            y=second_feat,
            title=f"Scatterplot {first_feat} | {second_feat}",
            **extra_params,
        )
        st.plotly_chart(fig)

def show_correlation_section():
    st.info("В данной секции можно посмотреть на зависимость признаков между собой ➕➖")

    st.sidebar.subheader("Корреляция")
    if st.sidebar.checkbox("🔢 Матрица корреляций", value=True):
        st.subheader("🔢 Матрица корреляций")

        correlation_matrix = DATA[NUMERIC_COLUMNS + ["TARGET"]].corr().round(2)

        colormap = [
            [0.0, "red"],
            [0.5, "white"],
            [1.0, "green"],
        ]

        fig = px.imshow(
            correlation_matrix,
            title="Корреляционная матрица для числовых признаков и таргета",
            text_auto=True,
            color_continuous_scale=colormap,
            zmin=-1,
            zmax=1,
        )

        st.plotly_chart(fig)

    if st.sidebar.checkbox("🔗 Коэффициенты корреляции", value=True):
        st.subheader("🔗 Коэффициенты корреляции")

        choice_1, choice_2 = st.columns(2)
        CORR_FEATURES = NUMERIC_COLUMNS + ["TARGET"]
        with choice_1:
            first_feat = st.selectbox("Выберите первый признак", CORR_FEATURES)
        with choice_2:
            second_feat = st.selectbox("Выберите второй признак", CORR_FEATURES)

        if first_feat == second_feat:
            st.info(f"{first_feat} — {FEATURE_DESCRIPTION[first_feat]}")
        else:
            st.info(
                f"{first_feat} — {FEATURE_DESCRIPTION[first_feat]}\n\n{second_feat} — {FEATURE_DESCRIPTION[second_feat]}"
            )

        correlations = {}
        for corr_coef in CORRELATION_FN:
            coef_ = CORRELATION_FN[corr_coef](
                DATA[first_feat],
                DATA[second_feat]
            ).statistic
            correlations[corr_coef] = round(coef_, 3)

        output_str = "\n".join(map(lambda x: f"* **{x[0]}:** {x[1]}", correlations.items()))
        st.success(output_str)

def show_statistics_section():
    st.info("В данной секции можно посчитать статистики для признаков 📱")

    st.sidebar.subheader("Статистики")
    if st.sidebar.checkbox("7️⃣ Статистики для числовых признаков", value=True):
        st.subheader("7️⃣ Статистики для числовых признаков")

        if st.checkbox("🎯 Разделить по статистики для числовых признаков по таргету", value=False):
            st.write("**Статистики числовых признаков для положительного класса:**")
            st.dataframe(DATA.loc[DATA.TARGET == 1, NUMERIC_COLUMNS].describe())

            st.write("**Статистики числовых признаков для отрицательного класса:**")
            st.dataframe(DATA.loc[DATA.TARGET == 0, NUMERIC_COLUMNS].describe())
        else:
            st.write("**Статистики числовых признаков:**")
            st.dataframe(DATA[NUMERIC_COLUMNS].describe())

    if st.sidebar.checkbox("📚 Статистики для категориальных признаков", value=True):
        st.subheader("📚 Статистики для категориальных признаков")

        if st.checkbox("🎯 Разделить по статистики для категориальных признаков по таргету", value=False):
            st.write("**Статистики категориальных признаков для положительного класса:**")
            st.dataframe(DATA.loc[DATA.TARGET == 1, CATEGORICAL_COLUMNS].describe())

            st.write("**Статистики категориальных признаков для отрицательного класса:**")
            st.dataframe(DATA.loc[DATA.TARGET == 0, CATEGORICAL_COLUMNS].describe())
        else:
            st.write("**Статистики категориальных признаков:**")
            st.dataframe(DATA[CATEGORICAL_COLUMNS].describe())

    if st.sidebar.checkbox("🐼 Квантили для числовых признаков", value=True):
        st.subheader("🐼 Квантили для числовых признаков")

        feature = st.selectbox("Выберите числовой признак для поиска квантиля", NUMERIC_COLUMNS)
        st.info(f"{feature} — {FEATURE_DESCRIPTION[feature]}")

        q = st.slider(label="Квантиль", min_value=0., max_value=1., step=0.001)
        q_value = DATA[feature].quantile(q)
        st.success(f"Квантиль {q} для {feature}: {q_value}")



def show_main_page():
    st.title("💰 Разведочный анализ данных для банка")
    st.image(MAIN_IMAGE_LINK.strip())
    st.info(DASHBOARD_DESCRIPTION)

    st.sidebar.title("EDA для банка")
    st.sidebar.info(
        """
        **Основные разделы:**
        - Распределения признаков
        - Зависимость между признаками
        - Статистики для признаков
        """
    )

    show_data_description()

    section_distr, section_corr, section_stats = st.tabs(["Распределения", "Корреляция", "Статистики"])

    with section_distr:
        show_distribution_section()

    with section_corr:
        show_correlation_section()

    with section_stats:
        show_statistics_section()



    



if __name__ == "__main__":
    show_main_page()
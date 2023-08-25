import streamlit as st
from PIL import Image
from eda.eda import (open_data, data_processor, number_variable_exploration, corr,
                     target_variable_exploration, log_variable, dependence_on_target)


def process_main_page():
    show_main_page()
    df_dict = open_data(path="data/datasets")
    data = data_processor(df_dict)
    st.markdown("**Часть исследуемых данных**")
    st.dataframe(data.head(7))
    st.markdown("___")
    show_chart(
        "**Посмотрим на распределение классов**", "Классы не сбалансированы",
        target_variable_exploration(data, "TARGET")
    )
    show_chart(
        "**Построим графики распределений числовых признаков**",
        "Попробуем прологарифмировать признак и добиться нормального распределения.",
        number_variable_exploration(data)
    )
    fig, data = log_variable(data)
    show_chart(
        "**Прологарифмированный признак PERSONAL_INCOME**",
        "Распределение стало похоже на нормальное", fig
    )
    show_chart(
        "**Построим графики зависимостей целевой переменной и признаков**",
        """Явного разбиения по одному признаку нет, но есть некоторые зависимости:
        люди с более серьезным образованием и большими доходами откликаются на предложения банка реже""",
        dependence_on_target(data)
    )
    show_chart(
        "**Построим матрицу корреляций**",
        "Линейные зависимости небольшие, за исключением признаков *DEPENDANTS* и *CHILD_TOTAL*",
        corr(data)
    )


def show_main_page():
    image = Image.open('data/images/img.png')

    st.set_page_config(
        layout="wide",
        initial_sidebar_state="auto",
        page_title="Отклик на предложения банка",
        page_icon=image,

    )
    st.markdown(
        """
        # Cклонность к отклику
        ### Определяем, кто из клиентов банка склонен к отклику на новое предложение.
        материал - данные о клиентах банка.
        """
    )
    st.image(image)
    st.markdown("Исследуем предварительно обработанные данные")


def show_chart(title, text, fig):
    st.markdown(title)
    st.pyplot(fig, use_container_width=False)
    st.markdown(text)
    st.markdown("___")


process_main_page()

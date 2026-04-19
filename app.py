"""
Главное приложение DeepScreen AI.
Автоматический скрининг резюме с использованием ИИ.
"""

import streamlit as st
import pandas as pd
from pathlib import Path
import time

# Импорт модулей проекта
from resume_parser import ResumeParser
from ai_analyzer import AIResumeAnalyzer
from utils import (
    save_uploaded_files,
    cleanup_temp_files,
    create_results_dataframe,
    get_score_color,
    format_skills_list
)


# Настройка страницы
st.set_page_config(
    page_title="DeepScreen AI",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)


# CSS стили
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .score-badge {
        font-size: 1.5rem;
        font-weight: bold;
        padding: 0.5rem 1rem;
        border-radius: 10px;
        text-align: center;
    }
    .candidate-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 5px solid #1f77b4;
    }
    .skill-tag {
        display: inline-block;
        background-color: #e3f2fd;
        color: #1565c0;
        padding: 0.25rem 0.75rem;
        border-radius: 15px;
        margin: 0.25rem;
        font-size: 0.9rem;
    }
    .footer {
        text-align: center;
        margin-top: 3rem;
        padding: 1rem;
        color: #999;
        border-top: 1px solid #eee;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Инициализация состояния сессии Streamlit."""
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = None
    if 'analyzing' not in st.session_state:
        st.session_state.analyzing = False


def main():
    """Главная функция приложения."""
    
    init_session_state()
    
    # Заголовок
    st.markdown('<div class="main-header">🎯 DeepScreen AI</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Интеллектуальный скрининг резюме с помощью ИИ</div>',
        unsafe_allow_html=True
    )

    # Боковая панель
    with st.sidebar:
        st.header("⚙️ Настройки")
        
        # Выбор модели Ollama
        st.subheader("🤖 Модель LLM")
        
        # Получаем доступные модели
        try:
            from ai_analyzer import AIResumeAnalyzer
            available_models = AIResumeAnalyzer.get_available_models()
            
            if available_models:
                model_options = list(available_models.keys())
                selected_model = st.selectbox(
                    "Выберите модель",
                    options=model_options,
                    help="Локальные модели через Ollama"
                )
                
                # Показываем информацию о модели
                if selected_model in available_models:
                    info = available_models[selected_model]
                    st.caption(f"📦 Размер: {info['size_gb']} GB")
                    if info.get('description'):
                        st.caption(f"📝 {info['description']}")
            else:
                selected_model = st.text_input(
                    "Название модели",
                    value="qwen2.5:7b",
                    help="Например: llama3.2, qwen2.5:7b, gemma2:9b"
                )
                st.warning("⚠️ Ollama не запущен или нет моделей")
                
        except Exception as e:
            selected_model = "qwen2.5:7b"
            st.error(f"Ошибка подключения к Ollama: {str(e)}")
            st.info("Убедитесь, что Ollama установлен и запущен")
        
        # Кнопка проверки подключения
        if st.button("🔌 Проверить подключение"):
            try:
                analyzer = AIResumeAnalyzer(model_name=selected_model)
                st.session_state.analyzer = analyzer
                st.success(f"✅ Подключено к {selected_model}")
            except Exception as e:
                st.error(f"❌ Ошибка: {str(e)}")
        
        # Автоматическая инициализация
        if 'analyzer' not in st.session_state or st.session_state.analyzer is None:
            try:
                st.session_state.analyzer = AIResumeAnalyzer(model_name=selected_model)
            except Exception as e:
                pass  # Будет показано при проверке
        
        st.divider()
    
    # Основная область
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📎 Загрузка резюме")
        uploaded_files = st.file_uploader(
            "Выберите файлы резюме",
            type=['pdf', 'docx', 'txt'],
            accept_multiple_files=True,
            help="Поддерживаются форматы PDF, DOCX, TXT"
        )
        
        if uploaded_files:
            st.success(f"Загружено файлов: {len(uploaded_files)}")
            
            # Показываем список загруженных файлов
            with st.expander("📋 Список загруженных файлов"):
                for i, f in enumerate(uploaded_files, 1):
                    st.text(f"{i}. {f.name}")
    
    with col2:
        st.subheader("📝 Описание вакансии")
        
        job_title = st.text_input(
            "Название должности",
            placeholder="Например: Senior Python Developer",
            help="Укажите название вакансии"
        )
        
        job_description = st.text_area(
            "Описание вакансии",
            placeholder="Вставьте текст вакансии: требования, обязанности, условия...",
            height=200,
            help="Чем подробнее описание, тем точнее будет анализ"
        )
    
    # Кнопка анализа
    st.divider()
    
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        analyze_button = st.button(
            "🚀 Начать анализ резюме",
            type="primary",
            use_container_width=True,
            disabled=not (uploaded_files and job_description and st.session_state.analyzer)
        )
    
    # Логика анализа
    if analyze_button and uploaded_files and job_description and st.session_state.analyzer:
        st.session_state.analyzing = True
        
        # Сохраняем загруженные файлы
        with st.spinner("📁 Сохранение файлов..."):
            file_paths = save_uploaded_files(uploaded_files)
        
        # Парсинг резюме
        resumes_data = []
        progress_bar = st.progress(0, text="📄 Парсинг резюме...")
        
        for i, file_path in enumerate(file_paths):
            progress_bar.progress(
                (i + 1) / len(file_paths),
                text=f"📄 Парсинг: {Path(file_path).name}"
            )
            
            try:
                parser = ResumeParser(file_path)
                text = parser.extract_text()
                contacts = parser.extract_contacts()
                name = parser.extract_name()
                
                resumes_data.append({
                    'file_name': Path(file_path).name,
                    'text': text,
                    'contacts': contacts,
                    'extracted_name': name
                })
            except Exception as e:
                st.error(f"Ошибка парсинга {Path(file_path).name}: {str(e)}")
                resumes_data.append({
                    'file_name': Path(file_path).name,
                    'text': f"[Ошибка парсинга: {str(e)}]",
                    'contacts': {},
                    'extracted_name': None
                })
        
        # ИИ-анализ
        progress_bar.progress(1.0, text="🤖 ИИ-анализ резюме...")
        
        with st.spinner("🤖 Искусственный интеллект анализирует резюме..."):
            try:
                # Подготовка данных для анализа
                resumes_for_analysis = [
                    (r['file_name'], r['text']) for r in resumes_data
                ]
                
                # Пакетный анализ
                results = st.session_state.analyzer.batch_analyze(
                    resumes_for_analysis,
                    job_description,
                    job_title
                )
                
                # Добавляем контакты к результатам
                for i, result in enumerate(results):
                    if i < len(resumes_data):
                        result['contacts'] = resumes_data[i]['contacts']
                        result['extracted_name'] = resumes_data[i]['extracted_name']
                
                st.session_state.analysis_results = results
                
            except Exception as e:
                st.error(f"❌ Ошибка анализа: {str(e)}")
                st.session_state.analysis_results = None
        
        progress_bar.empty()
        st.session_state.analyzing = False
        
        # Очистка временных файлов
        cleanup_temp_files()
    
    # Отображение результатов
    if st.session_state.analysis_results:
        st.divider()
        st.header("📊 Результаты анализа")
        
        results = st.session_state.analysis_results
        
        # Сводная статистика
        total = len(results)
        success_count = sum(1 for r in results if r.get('status') != 'error')
        high_match = sum(1 for r in results if r.get('match_score', 0) >= 80)
        medium_match = sum(1 for r in results if 60 <= r.get('match_score', 0) < 80)
        
        col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
        with col_stat1:
            st.metric("Всего резюме", total)
        with col_stat2:
            st.metric("Успешно обработано", success_count)
        with col_stat3:
            st.metric("Отличное совпадение (80%+)", high_match)
        with col_stat4:
            st.metric("Хорошее совпадение (60%+)", medium_match)
        
        st.divider()
        
        # Таблица с результатами
        st.subheader("📋 Сводная таблица")
        df = create_results_dataframe(results)
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Файл": st.column_config.TextColumn("Файл", width="medium"),
                "Соответствие": st.column_config.TextColumn("Соответствие", width="small"),
                "Навыки": st.column_config.TextColumn("Навыки", width="large"),
                "Рекомендация": st.column_config.TextColumn("Рекомендация", width="medium"),
                "Статус": st.column_config.TextColumn("Статус", width="small")
            }
        )
        
        st.divider()
        
        # Детальные карточки кандидатов
        st.subheader("🔍 Детальный анализ кандидатов")
        
        for i, result in enumerate(results):
            if result.get('status') == 'error':
                with st.expander(f"❌ {result.get('file_name', 'Неизвестный файл')} - Ошибка"):
                    st.error(result.get('error', 'Неизвестная ошибка'))
                continue
            
            score = result.get('match_score', 0)
            score_color = get_score_color(score)
            
            with st.expander(
                f"{'🟢' if score >= 80 else '🟡' if score >= 60 else '🔴'} "
                f"{result.get('file_name', 'Неизвестно')} - "
                f"Соответствие: {score}% - "
                f"{result.get('recommendation', 'Нет рекомендации')}"
            ):
                col_left, col_right = st.columns([2, 1])
                
                with col_left:
                    st.markdown(f"""
                    <div class="candidate-card">
                        <h3>📄 {result.get('file_name', 'Неизвестно')}</h3>
                        
                        <p><strong>📊 Соответствие:</strong> 
                        <span style="color: {score_color}; font-weight: bold; font-size: 1.3rem;">
                            {score}%
                        </span>
                        </p>
                        
                        <p><strong>💡 Рекомендация:</strong> {result.get('recommendation', 'Не определена')}</p>
                        
                        <p><strong>📝 Опыт:</strong> {result.get('experience_summary', 'Нет данных')}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Навыки
                    st.markdown("#### 🛠 Навыки")
                    skills = result.get('skills_match', [])
                    if skills:
                        skills_html = ''.join([
                            f'<span class="skill-tag">{skill}</span>'
                            for skill in skills[:15]
                        ])
                        st.markdown(f'<div>{skills_html}</div>', unsafe_allow_html=True)
                    else:
                        st.info("Навыки не определены")
                    
                    # Отсутствующие навыки
                    missing = result.get('missing_skills', [])
                    if missing:
                        st.markdown("#### ⚠️ Отсутствующие навыки")
                        st.write(", ".join(missing))
                    
                    # Плюсы и минусы
                    col_pros, col_cons = st.columns(2)
                    with col_pros:
                        st.markdown("#### ✅ Сильные стороны")
                        for pro in result.get('pros', []):
                            st.markdown(f"- {pro}")
                    
                    with col_cons:
                        st.markdown("#### ❌ Слабые стороны")
                        for con in result.get('cons', []):
                            st.markdown(f"- {con}")
                
                with col_right:
                    # Контакты
                    contacts = result.get('contacts', {})
                    extracted_name = result.get('extracted_name')
                    
                    st.markdown("#### 📞 Контакты")
                    if extracted_name:
                        st.write(f"**Имя:** {extracted_name}")
                    if contacts.get('email'):
                        st.write(f"**Email:** {contacts['email']}")
                    if contacts.get('phone'):
                        st.write(f"**Телефон:** {contacts['phone']}")
                    
                    if not contacts.get('email') and not contacts.get('phone'):
                        st.info("Контакты не найдены")
        
        # Экспорт результатов
        st.divider()
        st.subheader("📥 Экспорт результатов")
        
        # Создаем CSV для скачивания
        csv_data = []
        for r in results:
            if r.get('status') != 'error':
                csv_data.append({
                    'Файл': r.get('file_name', ''),
                    'Имя': r.get('extracted_name', ''),
                    'Email': r.get('contacts', {}).get('email', ''),
                    'Телефон': r.get('contacts', {}).get('phone', ''),
                    'Соответствие': r.get('match_score', 0),
                    'Рекомендация': r.get('recommendation', ''),
                    'Навыки': ', '.join(r.get('skills_match', [])),
                    'Отсутствуют': ', '.join(r.get('missing_skills', []))
                })
        
        if csv_data:
            df_export = pd.DataFrame(csv_data)
            csv = df_export.to_csv(index=False).encode('utf-8-sig')
            
            st.download_button(
                label="📥 Скачать результаты (CSV)",
                data=csv,
                file_name=f"deepscreen_results_{time.strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    # Футер
    st.markdown("""
    <div class="footer">
        DeepScreen AI — Интеллектуальный скрининг резюме<br>
        Powered by OpenAI • Streamlit • Python
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
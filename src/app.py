import streamlit as st
import pandas as pd
from pathlib import Path
import time

from resume_parser import ResumeParser
from ai_analyzer import AIResumeAnalyzer
from utils import (
    save_uploaded_files,
    cleanup_temp_files,
    create_results_dataframe,
    get_score_color,
    format_skills_list
)

st.set_page_config(
    page_title="DeepScreen AI",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
    .stat-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
    }
    .stat-box-green {
        background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
    }
    .stat-box-orange {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
    }
    .stat-box-blue {
        background: linear-gradient(135deg, #30cfd0 0%, #330867 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
    }
    .stat-number {
        font-size: 2.5rem;
        font-weight: bold;
    }
    .stat-label {
        font-size: 1rem;
        opacity: 0.9;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = None
    if 'analyzing' not in st.session_state:
        st.session_state.analyzing = False
    if 'model_connected' not in st.session_state:
        st.session_state.model_connected = False


def main():
    
    init_session_state()
    
    # Заголовок
    st.markdown('<div class="main-header">🎯 DeepScreen AI</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Интеллектуальный скрининг резюме с помощью ИИ</div>',
        unsafe_allow_html=True
    )
    
    with st.sidebar:
        st.header("⚙️ Настройки")
        
        st.subheader("🤖 Модель LLM")
        
        selected_model = st.selectbox(
            "Выберите модель",
            options=["qwen2.5:7b", "llama3.2", "gemma2:9b", "mistral", "llama3.1:8b"],
            index=0,
            help="Локальные модели через Ollama"
        )
        
        custom_model = st.text_input(
            "Или укажите свою модель",
            placeholder="например: qwen2.5:7b",
            help="Введите название модели, установленной в Ollama"
        )
        
        if custom_model:
            selected_model = custom_model
        
        col_btn1, col_btn2 = st.columns([2, 1])
        with col_btn1:
            if st.button("🔌 Подключить модель", use_container_width=True):
                with st.spinner("Подключение к Ollama..."):
                    try:
                        st.session_state.analyzer = AIResumeAnalyzer(model_name=selected_model)
                        st.session_state.model_connected = True
                        st.success(f"✅ Подключено к **{selected_model}**")
                    except Exception as e:
                        st.session_state.model_connected = False
                        st.error(f"❌ Ошибка подключения")
                        st.info(f"```{str(e)}```")
        
        with col_btn2:
            if st.button("🔄", help="Проверить статус", use_container_width=True):
                try:
                    import ollama
                    models = ollama.list()
                    model_names = [m.get('name', '') for m in models.get('models', [])]
                    if model_names:
                        st.success(f"✅ Доступно моделей: {len(model_names)}")
                        with st.expander("Список моделей"):
                            for m in model_names:
                                st.text(f"• {m}")
                    else:
                        st.warning("⚠️ Нет доступных моделей")
                except Exception as e:
                    st.error("❌ Ollama не запущен")
        
        # Статус подключения
        if st.session_state.model_connected:
            st.success(f"🟢 Подключено: **{selected_model}**")
        else:
            st.warning("🟡 Модель не подключена")
        
        st.divider()
        
        # Инструкция по установке
        with st.expander("📚 Как установить Ollama"):
            st.markdown("""
            ### Установка Ollama
            
            **1. Скачайте и установите:**
            - Перейдите на [ollama.com](https://ollama.com/)
            - Скачайте версию для вашей ОС
            - Установите и запустите
            
            **2. Скачайте модель:**
            ```bash
            # Рекомендуемая для русского языка:
            ollama pull qwen2.5:7b
            
            # Английский:
            ollama pull llama3.2
            ```
            
            **3. Проверьте установку:**
            ```bash
            ollama list
            ```
            
            **4. Запустите сервер (если не запущен):**
            ```bash
            ollama serve
            ```
            """)
        
        st.divider()
        
        # Информация о приватности
        with st.expander("🔒 Приватность и безопасность"):
            st.markdown("""
            ### Ваши данные в безопасности
            
            - ✅ **100% локальная обработка**
            - ✅ Данные **НЕ** отправляются в интернет
            - ✅ Файлы удаляются после анализа
            - ✅ Никакой телеметрии
            
            **DeepScreen AI** работает полностью на вашем компьютере.
            """)
        
        st.divider()
        
        # Инструкция
        st.markdown("""
        ### 📋 Инструкция
        
        1. ✅ Подключите модель
        2. 📎 Загрузите резюме (PDF/DOCX/TXT)
        3. 📝 Вставьте описание вакансии
        4. 🚀 Нажмите "Начать анализ"
        
        ---
        
        ### 📊 Оценки
        
        | % | Значение |
        |:---:|:---|
        | 🟢 80-100 | Отлично |
        | 🟡 60-79 | Хорошо |
        | 🔴 0-59 | Низко |
        """)
    
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
            st.success(f"📁 Загружено файлов: **{len(uploaded_files)}**")
            
            with st.expander("📋 Список загруженных файлов"):
                for i, f in enumerate(uploaded_files, 1):
                    st.text(f"{i}. {f.name} ({f.size // 1024} КБ)")
        else:
            st.info("👆 Загрузите одно или несколько резюме для анализа")
    
    with col2:
        st.subheader("📝 Описание вакансии")
        
        job_title = st.text_input(
            "Название должности",
            placeholder="Например: Senior Python Developer",
            help="Укажите название вакансии для более точного анализа"
        )
        
        job_description = st.text_area(
            "Описание вакансии",
            placeholder="Вставьте текст вакансии, включая:\n- Обязанности\n- Требования к кандидату\n- Необходимые навыки\n- Условия работы\n\nЧем подробнее описание, тем точнее будет анализ.",
            height=200,
            help="Подробное описание повышает точность анализа"
        )
        
        if job_description:
            char_count = len(job_description)
            st.caption(f"📝 Символов: {char_count}")
    
    # Кнопка анализа
    st.divider()
    
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        analyze_disabled = not (
            uploaded_files 
            and job_description 
            and st.session_state.analyzer 
            and st.session_state.model_connected
        )
        
        analyze_button = st.button(
            "🚀 Начать анализ резюме",
            type="primary",
            use_container_width=True,
            disabled=analyze_disabled
        )
        
        if analyze_disabled:
            if not st.session_state.model_connected:
                st.warning("⚠️ Сначала подключите модель в боковой панели")
            elif not uploaded_files:
                st.info("📎 Загрузите резюме")
            elif not job_description:
                st.info("📝 Введите описание вакансии")
    
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
                st.error(f"❌ Ошибка парсинга {Path(file_path).name}: {str(e)}")
                resumes_data.append({
                    'file_name': Path(file_path).name,
                    'text': f"[Ошибка парсинга: {str(e)}]",
                    'contacts': {},
                    'extracted_name': None
                })
        
        # ИИ-анализ
        progress_bar.progress(1.0, text="🤖 ИИ-анализ резюме...")
        
        with st.spinner("🤖 Искусственный интеллект анализирует резюме... Это может занять некоторое время."):
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
            st.markdown(f"""
            <div class="stat-box-blue">
                <div class="stat-number">{total}</div>
                <div class="stat-label">Всего резюме</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col_stat2:
            st.markdown(f"""
            <div class="stat-box-green">
                <div class="stat-number">{success_count}</div>
                <div class="stat-label">Обработано</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col_stat3:
            st.markdown(f"""
            <div class="stat-box">
                <div class="stat-number">{high_match}</div>
                <div class="stat-label">Отлично (80%+)</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col_stat4:
            st.markdown(f"""
            <div class="stat-box-orange">
                <div class="stat-number">{medium_match}</div>
                <div class="stat-label">Хорошо (60%+)</div>
            </div>
            """, unsafe_allow_html=True)
        
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
            
            if score >= 80:
                emoji = "🟢"
            elif score >= 60:
                emoji = "🟡"
            else:
                emoji = "🔴"
            
            with st.expander(
                f"{emoji} {result.get('file_name', 'Неизвестно')} - "
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
                        for skill in missing[:5]:
                            st.markdown(f"- ❌ {skill}")
                    
                    # Плюсы и минусы
                    col_pros, col_cons = st.columns(2)
                    with col_pros:
                        st.markdown("#### ✅ Сильные стороны")
                        pros = result.get('pros', [])
                        if pros:
                            for pro in pros:
                                st.markdown(f"- {pro}")
                        else:
                            st.info("Нет данных")
                    
                    with col_cons:
                        st.markdown("#### ❌ Слабые стороны")
                        cons = result.get('cons', [])
                        if cons:
                            for con in cons:
                                st.markdown(f"- {con}")
                        else:
                            st.info("Нет данных")
                
                with col_right:
                    # Контакты
                    contacts = result.get('contacts', {})
                    extracted_name = result.get('extracted_name') or result.get('candidate_name')
                    
                    st.markdown("#### 📞 Контакты")
                    
                    if extracted_name:
                        st.markdown(f"**👤 Имя:** {extracted_name}")
                    
                    if contacts.get('email'):
                        st.markdown(f"**📧 Email:** {contacts['email']}")
                    
                    if contacts.get('phone'):
                        st.markdown(f"**📱 Телефон:** {contacts['phone']}")
                    
                    if not contacts.get('email') and not contacts.get('phone'):
                        st.info("Контакты не найдены")
                    
                    st.divider()
                    
                    # Опыт
                    years = result.get('total_experience_years')
                    if years:
                        st.markdown(f"**⏱️ Опыт:** {years} лет")
        
        # Экспорт результатов
        st.divider()
        st.subheader("📥 Экспорт результатов")
        
        # Создаем CSV для скачивания
        csv_data = []
        for r in results:
            if r.get('status') != 'error':
                csv_data.append({
                    'Файл': r.get('file_name', ''),
                    'Имя': r.get('extracted_name') or r.get('candidate_name', ''),
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
            
            col_down1, col_down2, col_down3 = st.columns([1, 2, 1])
            with col_down2:
                st.download_button(
                    label="📥 Скачать результаты (CSV)",
                    data=csv,
                    file_name=f"deepscreen_results_{time.strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
    
    # Футер
    st.markdown("""
    <div class="footer">
        DeepScreen AI — Интеллектуальный скрининг резюме<br>
        🎯 Powered by Ollama • Streamlit • Python 🐍<br>
        <span style="font-size: 0.8rem;"></span>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
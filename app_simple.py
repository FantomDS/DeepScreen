"""
Упрощенная версия DeepScreen - только парсинг резюме.
"""
import streamlit as st
import pandas as pd
from pathlib import Path
import tempfile
import os

# Импорт только парсера
from src.resume_parser import ResumeParser

st.set_page_config(page_title="DeepScreen - Тест", layout="wide")
st.title("📄 Тест парсинга резюме")

uploaded_files = st.file_uploader(
    "Загрузите резюме",
    type=['pdf', 'docx', 'txt'],
    accept_multiple_files=True
)

if uploaded_files:
    results = []
    
    for uploaded_file in uploaded_files:
        # Сохраняем во временный файл
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp:
            tmp.write(uploaded_file.getbuffer())
            tmp_path = tmp.name
        
        try:
            parser = ResumeParser(tmp_path)
            text = parser.extract_text()
            contacts = parser.extract_contacts()
            name = parser.extract_name()
            
            results.append({
                'Файл': uploaded_file.name,
                'Имя': name or 'Не найдено',
                'Email': contacts.get('email') or '-',
                'Телефон': contacts.get('phone') or '-',
                'Текст (первые 200 символов)': text[:200] + '...' if len(text) > 200 else text
            })
        except Exception as e:
            results.append({
                'Файл': uploaded_file.name,
                'Имя': 'Ошибка',
                'Email': '-',
                'Телефон': '-',
                'Текст (первые 200 символов)': f'Ошибка: {str(e)}'
            })
        finally:
            os.unlink(tmp_path)
    
    df = pd.DataFrame(results)
    st.dataframe(df, use_container_width=True)
"""
Вспомогательные функции для проекта.
"""

import os
import shutil
import tempfile
from pathlib import Path
from typing import List, Optional
import streamlit as st
import pandas as pd


def save_uploaded_files(uploaded_files) -> List[str]:
    """
    Сохраняет загруженные файлы во временную директорию.
    
    Args:
        uploaded_files: Файлы из st.file_uploader
        
    Returns:
        List[str]: Список путей к сохраненным файлам
    """
    # Создаем временную директорию в сессии
    if 'temp_dir' not in st.session_state:
        st.session_state.temp_dir = tempfile.mkdtemp()
    
    saved_paths = []
    
    for uploaded_file in uploaded_files:
        # Очищаем имя файла
        safe_filename = "".join(c for c in uploaded_file.name if c.isalnum() or c in '._- ')
        file_path = os.path.join(st.session_state.temp_dir, safe_filename)
        
        # Сохраняем файл
        with open(file_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        
        saved_paths.append(file_path)
    
    return saved_paths


def cleanup_temp_files():
    """Удаляет временные файлы."""
    if 'temp_dir' in st.session_state:
        try:
            shutil.rmtree(st.session_state.temp_dir)
        except:
            pass
        del st.session_state.temp_dir


def create_results_dataframe(results: List[dict]) -> pd.DataFrame:
    """
    Создает pandas DataFrame из результатов анализа.
    
    Args:
        results: Список результатов анализа
        
    Returns:
        pd.DataFrame: Таблица с результатами
    """
    rows = []
    
    for r in results:
        if r.get('status') == 'error':
            rows.append({
                'Файл': r.get('file_name', 'Неизвестно'),
                'Соответствие': 'Ошибка',
                'Навыки': '-',
                'Рекомендация': 'Ошибка',
                'Статус': r.get('error', 'Неизвестная ошибка')
            })
        else:
            rows.append({
                'Файл': r.get('file_name', 'Неизвестно'),
                'Соответствие': f"{r.get('match_score', 0)}%",
                'Навыки': ', '.join(r.get('skills_match', [])[:5]),
                'Рекомендация': r.get('recommendation', 'Не определена'),
                'Статус': 'Успешно'
            })
    
    return pd.DataFrame(rows)


def get_score_color(score: int) -> str:
    """
    Возвращает цвет для отображения score.
    
    Args:
        score: Процент соответствия
        
    Returns:
        str: CSS-цвет
    """
    if score >= 80:
        return '#28a745'  # Зеленый
    elif score >= 60:
        return '#ffc107'  # Желтый
    else:
        return '#dc3545'  # Красный


def format_skills_list(skills: List[str]) -> str:
    """
    Форматирует список навыков для отображения.
    
    Args:
        skills: Список навыков
        
    Returns:
        str: Отформатированная строка
    """
    if not skills:
        return "Навыки не найдены"
    return " • ".join(skills[:10]) + (" ..." if len(skills) > 10 else "")
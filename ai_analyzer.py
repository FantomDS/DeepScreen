"""
Модуль для анализа резюме с помощью локальной LLM через Ollama.
"""

import json
import re
from typing import List, Dict, Optional, Tuple, Any
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Импорт Ollama
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    logger.error("Ollama не установлен. Выполните: pip install ollama")


class AIResumeAnalyzer:
    """
    Класс для анализа резюме с помощью локальных LLM моделей через Ollama.
    
    Преимущества:
    - Полностью локально, без отправки данных в облако
    - Бесплатно (после загрузки модели)
    - Не требует интернета
    - Конфиденциальность данных
    """
    
    # Доступные модели и их особенности
    AVAILABLE_MODELS = {
        'llama3.2': {
            'description': 'Meta Llama 3.2 - хороший баланс качество/скорость',
            'context_window': 4096,
            'languages': ['en', 'ru']
        },
        'qwen2.5:7b': {
            'description': 'Qwen 2.5 - отличная поддержка русского языка',
            'context_window': 8192,
            'languages': ['ru', 'en', 'zh']
        },
        'gemma2:9b': {
            'description': 'Google Gemma 2 - мощная модель',
            'context_window': 8192,
            'languages': ['en', 'ru']
        },
        'mistral': {
            'description': 'Mistral - быстрая и эффективная',
            'context_window': 4096,
            'languages': ['en']
        }
    }
    
    def __init__(self, model_name: str = "qwen2.5:7b", host: str = None):
        """
        Инициализация анализатора с Ollama.
        
        Args:
            model_name: Название модели в Ollama (по умолчанию qwen2.5:7b)
            host: Адрес Ollama сервера (если не локальный)
        """
        if not OLLAMA_AVAILABLE:
            raise ImportError(
                "Библиотека ollama не установлена. "
                "Выполните: pip install ollama"
            )
        
        self.model_name = model_name
        self.host = host
        
        # Проверяем доступность модели
        self._check_model_availability()
        
        logger.info(f"Инициализирован анализатор с моделью: {model_name}")
    
    def _check_model_availability(self):
        """Проверяет, доступна ли указанная модель в Ollama."""
        try:
            models = ollama.list()
            model_names = [m.get('name', '') for m in models.get('models', [])]
            
            # Проверяем точное совпадение или с тегом
            base_model = self.model_name.split(':')[0]
            
            found = any(
                m == self.model_name or m.startswith(f"{base_model}:")
                for m in model_names
            )
            
            if not found:
                logger.warning(
                    f"Модель '{self.model_name}' не найдена в Ollama. "
                    f"Установите её командой: ollama pull {self.model_name}"
                )
                logger.info(f"Доступные модели: {model_names}")
        except Exception as e:
            logger.error(f"Ошибка подключения к Ollama: {e}")
            logger.info("Убедитесь, что Ollama запущен (команда 'ollama serve')")
    
    def _call_ollama(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.3,
        max_tokens: int = 2000,
        format_json: bool = False
    ) -> str:
        """
        Вызов Ollama API.
        
        Args:
            prompt: Промпт для модели
            system_prompt: Системный промпт
            temperature: Температура генерации (0.0 - 1.0)
            max_tokens: Максимальное количество токенов в ответе
            format_json: Форсировать JSON формат ответа
            
        Returns:
            str: Ответ модели
        """
        try:
            options = {
                'temperature': temperature,
                'num_predict': max_tokens,
            }
            
            # Формируем сообщения
            messages = []
            if system_prompt:
                messages.append({'role': 'system', 'content': system_prompt})
            messages.append({'role': 'user', 'content': prompt})
            
            # Дополнительные параметры для JSON формата
            if format_json:
                options['format'] = 'json'
            
            response = ollama.chat(
                model=self.model_name,
                messages=messages,
                options=options,
                host=self.host
            )
            
            return response['message']['content']
            
        except Exception as e:
            logger.error(f"Ошибка вызова Ollama: {e}")
            raise Exception(f"Ошибка Ollama API: {str(e)}")
    
    def analyze_resume(
        self,
        resume_text: str,
        job_description: str,
        job_title: str = ""
    ) -> Dict:
        """
        Анализирует резюме на соответствие вакансии.
        
        Args:
            resume_text: Текст резюме
            job_description: Описание вакансии
            job_title: Название должности
            
        Returns:
            Dict: Результаты анализа
        """
        # Ограничиваем длину текста для контекстного окна модели
        max_resume_length = 3500
        max_job_length = 2000
        
        resume_text = resume_text[:max_resume_length]
        job_description = job_description[:max_job_length]
        
        system_prompt = """Ты — опытный HR-специалист и рекрутер с 10-летним стажем. 
Твоя задача — профессионально анализировать резюме кандидатов на соответствие вакансии.
Отвечай ТОЛЬКО валидным JSON без дополнительного текста и комментариев."""
        
        prompt = f"""
Проанализируй резюме кандидата на соответствие вакансии.

ВАКАНСИЯ{f' ({job_title})' if job_title else ''}:
{job_description}

РЕЗЮМЕ КАНДИДАТА:
{resume_text}

Верни СТРОГО JSON объект (без markdown-оберток) со следующей структурой:
{{
    "match_score": число от 0 до 100,
    "skills_match": ["навык1", "навык2", ...],
    "missing_skills": ["отсутствующий_навык1", ...],
    "experience_summary": "краткое резюме опыта кандидата (1-2 предложения на русском)",
    "recommendation": "одно из: 'Пригласить на собеседование', 'Рассмотреть', 'Отклонить'",
    "pros": ["сильная сторона 1", "сильная сторона 2", ...],
    "cons": ["слабая сторона 1", "риск 1", ...],
    "candidate_name": "имя кандидата или null если не найдено",
    "total_experience_years": примерное количество лет опыта или null
}}

ВАЖНО:
1. match_score - объективная оценка от 0 до 100
2. skills_match - только навыки, которые ЕСТЬ в резюме И соответствуют вакансии
3. missing_skills - важные навыки из вакансии, которых НЕТ в резюме
4. recommendation - четкая рекомендация на основе анализа
5. Отвечай ТОЛЬКО JSON, без ```json и других оберток
"""
        
        try:
            response = self._call_ollama(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.2,  # Низкая температура для стабильности
                max_tokens=1500,
                format_json=True
            )
            
            # Очистка ответа и парсинг JSON
            result = self._parse_json_response(response)
            
            # Валидация и нормализация результата
            result = self._validate_and_normalize_result(result)
            
            return result
            
        except json.JSONDecodeError as e:
            logger.error(f"Ошибка парсинга JSON: {e}")
            logger.debug(f"Сырой ответ: {response if 'response' in locals() else 'No response'}")
            
            # Fallback в случае ошибки
            return self._get_fallback_result("Ошибка парсинга ответа модели")
            
        except Exception as e:
            logger.error(f"Ошибка анализа: {e}")
            return self._get_fallback_result(str(e))
    
    def _parse_json_response(self, response: str) -> Dict:
        """
        Парсит JSON из ответа модели, очищая от возможных артефактов.
        """
        response = response.strip()
        
        # Удаляем markdown-обертки если есть
        if response.startswith('```json'):
            response = response[7:]
        elif response.startswith('```'):
            response = response[3:]
        
        if response.endswith('```'):
            response = response[:-3]
        
        # Ищем JSON объект в тексте
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            response = json_match.group(0)
        
        return json.loads(response.strip())
    
    def _validate_and_normalize_result(self, result: Dict) -> Dict:
        """
        Проверяет и нормализует структуру результата.
        """
        normalized = {
            'match_score': 0,
            'skills_match': [],
            'missing_skills': [],
            'experience_summary': '',
            'recommendation': 'Рассмотреть',
            'pros': [],
            'cons': [],
            'candidate_name': None,
            'total_experience_years': None
        }
        
        # Копируем существующие поля с проверкой типов
        for key in normalized.keys():
            if key in result:
                if key == 'match_score':
                    try:
                        normalized[key] = int(float(result[key]))
                        normalized[key] = max(0, min(100, normalized[key]))
                    except (ValueError, TypeError):
                        pass
                elif key in ['skills_match', 'missing_skills', 'pros', 'cons']:
                    if isinstance(result[key], list):
                        normalized[key] = [str(item) for item in result[key]]
                elif key == 'experience_summary':
                    normalized[key] = str(result[key]) if result[key] else ''
                elif key == 'recommendation':
                    rec = str(result[key])
                    valid_recs = ['Пригласить на собеседование', 'Рассмотреть', 'Отклонить']
                    normalized[key] = rec if rec in valid_recs else 'Рассмотреть'
                else:
                    normalized[key] = result[key]
        
        return normalized
    
    def _get_fallback_result(self, error_message: str) -> Dict:
        """
        Возвращает результат по умолчанию в случае ошибки.
        """
        return {
            'match_score': 0,
            'skills_match': [],
            'missing_skills': [],
            'experience_summary': f'Ошибка анализа: {error_message}',
            'recommendation': 'Ошибка',
            'pros': [],
            'cons': [],
            'candidate_name': None,
            'total_experience_years': None,
            'error': error_message
        }
    
    def extract_skills(self, text: str) -> List[str]:
        """
        Извлекает навыки из текста с помощью LLM.
        
        Args:
            text: Текст для анализа
            
        Returns:
            List[str]: Список извлеченных навыков
        """
        text = text[:3000]
        
        prompt = f"""
Извлеки все профессиональные навыки и технологии из следующего текста резюме.
Верни результат в виде JSON массива строк.

Текст:
{text}

Пример ответа: ["Python", "SQL", "Project Management", "Agile", "Docker"]

Верни ТОЛЬКО JSON массив, без дополнительного текста.
"""
        
        try:
            response = self._call_ollama(
                prompt=prompt,
                system_prompt="Ты извлекаешь навыки из резюме. Отвечай только JSON массивом.",
                temperature=0.1,
                max_tokens=500,
                format_json=True
            )
            
            # Парсим ответ
            response = response.strip()
            if response.startswith('['):
                skills = json.loads(response)
            else:
                # Ищем массив в тексте
                match = re.search(r'\[.*\]', response, re.DOTALL)
                if match:
                    skills = json.loads(match.group(0))
                else:
                    skills = []
            
            return skills if isinstance(skills, list) else []
            
        except Exception as e:
            logger.error(f"Ошибка извлечения навыков: {e}")
            return []
    
    def batch_analyze(
        self,
        resumes: List[Tuple[str, str]],  # (file_name, resume_text)
        job_description: str,
        job_title: str = "",
        progress_callback=None
    ) -> List[Dict]:
        """
        Пакетный анализ нескольких резюме.
        
        Args:
            resumes: Список кортежей (имя_файла, текст_резюме)
            job_description: Описание вакансии
            job_title: Название должности
            progress_callback: Функция обратного вызова для отчета о прогрессе
            
        Returns:
            List[Dict]: Список результатов анализа, отсортированный по match_score
        """
        results = []
        total = len(resumes)
        
        for i, (file_name, resume_text) in enumerate(resumes):
            logger.info(f"Анализ {i+1}/{total}: {file_name}")
            
            if progress_callback:
                progress_callback(i, total, file_name)
            
            try:
                analysis = self.analyze_resume(resume_text, job_description, job_title)
                analysis['file_name'] = file_name
                analysis['status'] = 'success'
                results.append(analysis)
                
            except Exception as e:
                logger.error(f"Ошибка анализа {file_name}: {e}")
                results.append({
                    'file_name': file_name,
                    'status': 'error',
                    'error': str(e),
                    'match_score': 0,
                    'recommendation': 'Ошибка анализа',
                    'skills_match': [],
                    'missing_skills': [],
                    'experience_summary': '',
                    'pros': [],
                    'cons': []
                })
        
        # Сортируем по убыванию match_score
        results.sort(key=lambda x: x.get('match_score', 0), reverse=True)
        
        return results
    
    @classmethod
    def get_available_models(cls) -> Dict:
        """
        Возвращает список доступных в Ollama моделей.
        """
        try:
            models = ollama.list()
            available = {}
            
            for model in models.get('models', []):
                name = model.get('name', '')
                size = model.get('size', 0)
                modified = model.get('modified', '')
                
                # Форматируем размер
                size_gb = size / (1024**3) if size else 0
                
                available[name] = {
                    'size_gb': round(size_gb, 2),
                    'modified': modified,
                    'description': cls.AVAILABLE_MODELS.get(
                        name.split(':')[0], 
                        {}
                    ).get('description', 'Неизвестная модель')
                }
            
            return available
            
        except Exception as e:
            logger.error(f"Ошибка получения списка моделей: {e}")
            return {}
    
    @classmethod
    def pull_model(cls, model_name: str) -> bool:
        """
        Загружает модель в Ollama.
        
        Args:
            model_name: Название модели
            
        Returns:
            bool: True если успешно
        """
        try:
            logger.info(f"Загрузка модели {model_name}...")
            ollama.pull(model_name)
            logger.info(f"Модель {model_name} успешно загружена")
            return True
        except Exception as e:
            logger.error(f"Ошибка загрузки модели: {e}")
            return False
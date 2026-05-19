import json
import re
import logging
from typing import Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False


class ResumeAdvisor:
    def __init__(self, model_name: str = "qwen2.5:7b"):
        """
        Инициализация советника.
        
        Args:
            model_name: Название модели Ollama
        """
        if not OLLAMA_AVAILABLE:
            raise ImportError("Библиотека ollama не установлена.")
        
        self.model_name = model_name
        logger.info(f"ResumeAdvisor инициализирован с моделью: {model_name}")
    
    def _call_ollama(self, prompt: str, system_prompt: str = "", temperature: float = 0.3, max_tokens: int = 2000, format_json: bool = False) -> str:
        """Вызов Ollama API."""
        try:
            messages = []
            if system_prompt:
                messages.append({'role': 'system', 'content': system_prompt})
            messages.append({'role': 'user', 'content': prompt})
            
            options = {'temperature': temperature, 'num_predict': max_tokens}
            if format_json:
                options['format'] = 'json'
            
            response = ollama.chat(model=self.model_name, messages=messages, options=options)
            return response['message']['content']
        except Exception as e:
            logger.error(f"Ошибка вызова Ollama: {e}")
            raise Exception(f"Ошибка Ollama API: {str(e)}")
    
    def _parse_json_response(self, response: str) -> Dict:
        """Безопасный парсинг JSON из ответа модели."""
        response = response.strip()
        if response.startswith('```json'):
            response = response[7:]
        elif response.startswith('```'):
            response = response[3:]
        if response.endswith('```'):
            response = response[:-3]
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            response = json_match.group(0)
        return json.loads(response.strip())
    
    def generate_improvement_tips(
        self,
        resume_text: str,
        job_description: str,
        job_title: str = "",
        current_score: int = 0,
        missing_skills: List[str] = None
    ) -> Dict:
        """
        Генерирует конкретные рекомендации по улучшению резюме.
        
        Args:
            resume_text: Текст резюме
            job_description: Описание вакансии
            job_title: Название должности
            current_score: Текущий score соответствия
            missing_skills: Список отсутствующих навыков
            
        Returns:
            Dict с рекомендациями
        """
        resume_text = resume_text[:3000]
        job_description = job_description[:2000]
        missing_str = ', '.join(missing_skills) if missing_skills else 'не определены'
        
        system_prompt = """Ты — карьерный консультант и эксперт по составлению резюме с 10-летним опытом.
Твоя задача — дать КОНКРЕТНЫЕ, ПРАКТИЧЕСКИЕ советы по улучшению резюме.
Отвечай ТОЛЬКО валидным JSON без дополнительного текста.
Советы должны быть максимально конкретными: что именно добавить, как переформулировать, какие слова использовать."""
        
        prompt = f"""
Проанализируй резюме кандидата и описание вакансии. Дай КОНКРЕТНЫЕ рекомендации по улучшению резюме.

ВАКАНСИЯ{f' ({job_title})' if job_title else ''}:
{job_description}

РЕЗЮМЕ:
{resume_text}

ТЕКУЩИЙ SCORE: {current_score}%
ОТСУТСТВУЮЩИЕ НАВЫКИ: {missing_str}

Верни СТРОГО JSON:
{{
    "critical_fixes": [
        {{
            "priority": "critical|important|recommended",
            "category": "skills|experience|formatting|wording|structure",
            "problem": "что не так (1 предложение)",
            "solution": "конкретный совет что сделать (2-3 предложения)",
            "example_before": "как сейчас (если применимо) или null",
            "example_after": "как должно быть (если применимо) или null"
        }}
    ],
    "keywords_to_add": ["ключевое слово 1", "ключевое слово 2"],
    "phrases_to_remove": ["слабая формулировка 1"],
    "predicted_score_after": число 0-100 (прогноз score после всех исправлений)
}}

ПРАВИЛА:
1. critical_fixes: 3-5 самых важных исправлений
2. Приоритеты: critical (без этого не возьмут), important (сильно повысит шансы), recommended (желательно)
3. Категории: skills (навыки), experience (опыт), formatting (оформление), wording (формулировки), structure (структура)
4. Советы должны быть КОНКРЕТНЫМИ: не "улучшите резюме", а "добавьте в раздел Навыки: Docker, Kubernetes"
5. keywords_to_add: слова, которые стоит добавить в резюме чтобы пройти ATS
6. phrases_to_remove: слабые формулировки, которые лучше заменить
"""
        
        try:
            response = self._call_ollama(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.3,
                max_tokens=2000,
                format_json=True
            )
            result = self._parse_json_response(response)
            return self._validate_tips_result(result)
        except Exception as e:
            logger.error(f"Ошибка генерации советов: {e}")
            return self._get_fallback_tips(str(e))
    
    def _validate_tips_result(self, result: Dict) -> Dict:
        """Валидация и нормализация результата."""
        normalized = {
            'critical_fixes': [],
            'keywords_to_add': [],
            'phrases_to_remove': [],
            'predicted_score_after': 0
        }
        
        if 'critical_fixes' in result and isinstance(result['critical_fixes'], list):
            for fix in result['critical_fixes']:
                normalized['critical_fixes'].append({
                    'priority': fix.get('priority', 'recommended'),
                    'category': fix.get('category', 'wording'),
                    'problem': fix.get('problem', ''),
                    'solution': fix.get('solution', ''),
                    'example_before': fix.get('example_before'),
                    'example_after': fix.get('example_after')
                })
        
        if 'keywords_to_add' in result and isinstance(result['keywords_to_add'], list):
            normalized['keywords_to_add'] = [str(k) for k in result['keywords_to_add']]
        
        if 'phrases_to_remove' in result and isinstance(result['phrases_to_remove'], list):
            normalized['phrases_to_remove'] = [str(p) for p in result['phrases_to_remove']]
        
        if 'predicted_score_after' in result:
            try:
                normalized['predicted_score_after'] = max(0, min(100, int(float(result['predicted_score_after']))))
            except (ValueError, TypeError):
                pass
        
        return normalized
    
    def _get_fallback_tips(self, error_message: str) -> Dict:
        """Запасной результат при ошибке."""
        return {
            'critical_fixes': [{
                'priority': 'important',
                'category': 'wording',
                'problem': f'Не удалось сгенерировать рекомендации: {error_message}',
                'solution': 'Попробуйте повторить анализ или проверьте подключение к Ollama.',
                'example_before': None,
                'example_after': None
            }],
            'keywords_to_add': [],
            'phrases_to_remove': [],
            'predicted_score_after': 0
        }
    
    def analyze_structure(self, resume_text: str) -> Dict:
        """
        Анализирует структуру резюме: наличие разделов, полноту.
        
        Args:
            resume_text: Текст резюме
            
        Returns:
            Dict с результатами анализа структуры
        """
        resume_text = resume_text[:3000]
        
        system_prompt = "Ты — эксперт по структуре резюме. Отвечай ТОЛЬКО валидным JSON."
        
        prompt = f"""
Проанализируй СТРУКТУРУ резюме. Проверь наличие и качество ключевых разделов.

РЕЗЮМЕ:
{resume_text}

Верни СТРОГО JSON:
{{
    "sections": {{
        "contacts": {{"present": true/false, "quality": "good|partial|missing", "comment": "комментарий"}},
        "about": {{"present": true/false, "quality": "good|partial|missing", "comment": "комментарий"}},
        "experience": {{"present": true/false, "quality": "good|partial|missing", "comment": "комментарий"}},
        "skills": {{"present": true/false, "quality": "good|partial|missing", "comment": "комментарий"}},
        "education": {{"present": true/false, "quality": "good|partial|missing", "comment": "комментарий"}},
        "achievements": {{"present": true/false, "quality": "good|partial|missing", "comment": "комментарий"}}
    }},
    "overall_assessment": "общая оценка структуры (1-2 предложения)",
    "structure_score": число 0-100
}}
"""
        
        try:
            response = self._call_ollama(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.2,
                max_tokens=1000,
                format_json=True
            )
            return self._parse_json_response(response)
        except Exception as e:
            logger.error(f"Ошибка анализа структуры: {e}")
            return {
                'sections': {},
                'overall_assessment': f'Ошибка анализа: {str(e)}',
                'structure_score': 0
            }
    
    def generate_checklist(
        self,
        improvement_tips: Dict,
        structure_analysis: Dict = None
    ) -> Dict:
        """
        Формирует пошаговый чек-лист действий на основе рекомендаций.
        
        Args:
            improvement_tips: Результат generate_improvement_tips()
            structure_analysis: Результат analyze_structure() (опционально)
            
        Returns:
            Dict с чек-листом
        """
        checklist = []
        step = 1
        
        # Критические исправления
        critical = [f for f in improvement_tips.get('critical_fixes', []) if f.get('priority') == 'critical']
        for fix in critical:
            checklist.append({
                'step': step,
                'priority': 'critical',
                'action': fix.get('solution', ''),
                'category': fix.get('category', ''),
                'icon': '🔴'
            })
            step += 1
        
        # Важные исправления
        important = [f for f in improvement_tips.get('critical_fixes', []) if f.get('priority') == 'important']
        for fix in important:
            checklist.append({
                'step': step,
                'priority': 'important',
                'action': fix.get('solution', ''),
                'category': fix.get('category', ''),
                'icon': '🟡'
            })
            step += 1
        
        # Рекомендации
        recommended = [f for f in improvement_tips.get('critical_fixes', []) if f.get('priority') == 'recommended']
        for fix in recommended:
            checklist.append({
                'step': step,
                'priority': 'recommended',
                'action': fix.get('solution', ''),
                'category': fix.get('category', ''),
                'icon': '🟢'
            })
            step += 1
        
        # Добавляем ключевые слова
        keywords = improvement_tips.get('keywords_to_add', [])
        if keywords:
            checklist.append({
                'step': step,
                'priority': 'important',
                'action': f"Добавьте в резюме ключевые слова: {', '.join(keywords)}",
                'category': 'keywords',
                'icon': '🟡'
            })
            step += 1
        
        # Добавляем информацию о структуре
        if structure_analysis and structure_analysis.get('sections'):
            sections = structure_analysis['sections']
            missing_sections = [name for name, data in sections.items() if not data.get('present')]
            if missing_sections:
                checklist.append({
                    'step': step,
                    'priority': 'important',
                    'action': f"Добавьте отсутствующие разделы: {', '.join(missing_sections)}",
                    'category': 'structure',
                    'icon': '🟡'
                })
                step += 1
            
            partial_sections = [name for name, data in sections.items() if data.get('quality') == 'partial']
            if partial_sections:
                checklist.append({
                    'step': step,
                    'priority': 'recommended',
                    'action': f"Улучшите разделы: {', '.join(partial_sections)}",
                    'category': 'structure',
                    'icon': '🟢'
                })
        
        return {
            'checklist': checklist,
            'total_steps': len(checklist),
            'predicted_score_after': improvement_tips.get('predicted_score_after', 0)
        }
    
    def get_priority_label(self, priority: str) -> str:
        """Возвращает читаемую метку приоритета."""
        labels = {
            'critical': '🔴 КРИТИЧНО',
            'important': '🟡 ВАЖНО',
            'recommended': '🟢 РЕКОМЕНДАЦИЯ'
        }
        return labels.get(priority, '⚪ СОВЕТ')
    
    def get_category_label(self, category: str) -> str:
        """Возвращает читаемую метку категории."""
        labels = {
            'skills': 'Навыки',
            'experience': 'Опыт работы',
            'formatting': 'Оформление',
            'wording': 'Формулировки',
            'structure': 'Структура',
            'keywords': 'Ключевые слова'
        }
        return labels.get(category, category)
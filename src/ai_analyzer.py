import json
import re
import logging
from typing import List, Dict, Optional, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    logger.error("Ollama не установлен. Выполните: pip install ollama")


class AIResumeAnalyzer: 
    AVAILABLE_MODELS = {
        'llama3.2': {'description': 'Meta Llama 3.2', 'context_window': 4096},
        'qwen2.5:7b': {'description': 'Qwen 2.5 - русский язык', 'context_window': 8192},
        'gemma2:9b': {'description': 'Google Gemma 2', 'context_window': 8192},
        'mistral': {'description': 'Mistral - быстрая', 'context_window': 4096}
    }
    
    def __init__(self, model_name: str = "qwen2.5:7b", host: str = None):

        if not OLLAMA_AVAILABLE:
            raise ImportError(
                "Библиотека ollama не установлена. "
                "Выполните: pip install ollama"
            )
        
        self.model_name = model_name
        self.host = host 
        
        self._check_model_availability()
        
        logger.info(f"Инициализирован анализатор с моделью: {model_name}")
    
    def _check_model_availability(self):
        try:
            models = ollama.list()
            model_names = [m.get('name', '') for m in models.get('models', [])]
            
            base_model = self.model_name.split(':')[0]
            found = any(
                m == self.model_name or m.startswith(f"{base_model}:")
                for m in model_names
            )
            
            if not found:
                logger.warning(
                    f"Модель '{self.model_name}' не найдена. "
                    f"Установите: ollama pull {self.model_name}"
                )
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

        try:
            options = {
                'temperature': temperature,
                'num_predict': max_tokens,
            }
            
            messages = []
            if system_prompt:
                messages.append({'role': 'system', 'content': system_prompt})
            messages.append({'role': 'user', 'content': prompt})
            
            if format_json:
                options['format'] = 'json'
            
            response = ollama.chat(
                model=self.model_name,
                messages=messages,
                options=options
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

        # Ограничиваем длину текста
        max_resume_length = 3500
        max_job_length = 2000
    
        resume_text = resume_text[:max_resume_length]
        job_description = job_description[:max_job_length]
    
        system_prompt = """Ты — эксперт по оценке резюме и строгий математик.
Твоя задача — оценить резюме по МАТЕМАТИЧЕСКОЙ МОДЕЛИ, а не интуитивно.
Отвечай ТОЛЬКО валидным JSON без дополнительного текста и комментариев."""
    
        prompt = f"""
Ты должен оценить резюме кандидата по СТРОГОЙ МАТЕМАТИЧЕСКОЙ МОДЕЛИ из 5 критериев.

=== МАТЕМАТИЧЕСКАЯ МОДЕЛЬ ОЦЕНКИ ===

Итоговый Score вычисляется по формуле:
Score = 100 × (0.35 × S_skills + 0.30 × S_exp + 0.15 × S_extra + 0.10 × S_edu + 0.10 × S_struct)

где каждый S_i ∈ [0, 1] оценивается следующим образом:

1. S_skills (Вес 35%) — оценка совпадения навыков:
   S_skills = (N_match / N_required)^0.7
   где N_match — число навыков кандидата, совпадающих с вакансией
   N_required — общее число обязательных навыков в вакансии
   Степерь 0.7 даёт НЕЛИНЕЙНОСТЬ: первые навыки важнее последних.

2. S_exp (Вес 30%) — оценка релевантности опыта:
   S_exp = 1 / (1 + e^(-3 × (r - 0.7)))
   где r = опыт_кандидата_в_годах / требуемый_опыт_в_годах
   Это СИГМОИДА: 70% требуемого опыта даёт оценку 0.5.
   При r → ∞ оценка стремится к 1.0 (насыщение).

3. S_extra (Вес 15%) — дополнительные требования:
   S_extra = min(1.0; сумма бонусов):
   - Знание иностранного языка (если требуется): +0.25
   - Профессиональный сертификат: +0.15
   - Владение смежными технологиями: +0.10 за каждую (макс 2)
   - Опыт в смежной отрасли: +0.10

4. S_edu (Вес 10%) — оценка образования:
   1.0 — профиль и уровень полностью совпадают
   0.6 — профиль совпадает, но уровень ниже
   0.3 — образование указано, но профиль не совпадает
   0.0 — образование не указано

5. S_struct (Вес 10%) — оценка структуры:
   S_struct = сумма баллов за:
   - Наличие раздела «О себе» / «Цель»: +0.15
   - Наличие количественных результатов (цифр, метрик): +0.25
   - Использование глаголов действия: +0.20
   - Отсутствие орфографических ошибок: +0.15
   - Адаптация резюме под вакансию: +0.25

=== ВАКАНСИЯ{f' ({job_title})' if job_title else ''} ===
{job_description}

=== РЕЗЮМЕ ===
{resume_text}

=== ЗАДАНИЕ ===
1. Оцени каждый критерий S_i по формулам выше.
2. Вычисли Score по формуле.
3. Сформируй рекомендацию:
   - Score ≥ 80 → "Резюме готово к отправке"
   - 60 ≤ Score < 80 → "Требуются улучшения"
   - Score < 60 → "Рекомендована переработка"

Верни СТРОГО JSON:
{{
    "subscores": {{
        "skills": {{"value": число_от_0_до_1, "N_match": число, "N_required": число, "weight": 0.35, "contribution": число_процентов}},
        "experience": {{"value": число_от_0_до_1, "r": число, "weight": 0.30, "contribution": число_процентов}},
        "extra": {{"value": число_от_0_до_1, "bonuses": ["бонус1"], "weight": 0.15, "contribution": число_процентов}},
        "education": {{"value": число_от_0_до_1, "level": "полное/частичное/отсутствует", "weight": 0.10, "contribution": число_процентов}},
        "structure": {{"value": число_от_0_до_1, "checks": ["что выполнено"], "weight": 0.10, "contribution": число_процентов}}
    }},
    "match_score": число_от_0_до_100,
    "skills_match": ["навык1", "навык2"],
    "missing_skills": ["отсутствующий1"],
    "experience_summary": "краткое резюме опыта",
    "recommendation": "Резюме готово к отправке|Требуются улучшения|Рекомендована переработка",
    "pros": ["плюс1"],
    "cons": ["минус1"],
    "candidate_name": "имя или null",
    "total_experience_years": число или null
}}

ВАЖНО:
- Каждый S_i должен быть ОБОСНОВАН формулой, а не интуицией.
- contribution = S_i × weight × 100 (вклад критерия в итоговый Score).
- Сумма всех contribution должна быть РАВНА match_score (проверь арифметику!).
"""
    
        try:
            response = self._call_ollama(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.2,
                max_tokens=2000,
                format_json=True
            )
        
            result = self._parse_json_response(response)
            result = self._validate_and_normalize_result(result)
            return result
        
        except json.JSONDecodeError as e:
            logger.error(f"Ошибка парсинга JSON: {e}")
            return self._get_fallback_result("Ошибка парсинга ответа модели")
        except Exception as e:
            logger.error(f"Ошибка анализа: {e}")
            return self._get_fallback_result(str(e))
    
    def _parse_json_response(self, response: str) -> Dict:
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
    
    def _validate_and_normalize_result(self, result: Dict) -> Dict:
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
                    valid_recs = ['Резюме готово к отправке', 'Требуются улучшения', 'Рекомендована переработка']
                    normalized[key] = rec if rec in valid_recs else 'Рассмотреть'
                else:
                    normalized[key] = result[key]
        
        return normalized
    
    def _get_fallback_result(self, error_message: str) -> Dict:
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
    
    def batch_analyze(
        self,
        resumes: List[Tuple[str, str]],
        job_description: str,
        job_title: str = "",
        progress_callback=None
    ) -> List[Dict]:
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
                results.append({
                    'file_name': file_name,
                    'status': 'error',
                    'error': str(e),
                    'match_score': 0,
                    'recommendation': 'Ошибка анализа'
                })
        
        results.sort(key=lambda x: x.get('match_score', 0), reverse=True)
        return results

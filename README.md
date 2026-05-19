# DeepScreen
**Учебный проект**

*Для Windows/Mac/Linux:<p>*
1.Скачайте и установите Ollama с официального сайта: https://ollama.com/<p>
2.После установки откройте терминал и скачайте модель:
```
ollama pull qwen2.5:7b
```
Далее в терминале выполните:
```
#Перейдите в папку проекта
cd deepscreen

#Создание виртуального окружения
python -m venv venv

#Активация
venv\Scripts\activate

```
*Для Mac/Linux:*
```
#Создание виртуального окружения
python3 -m venv venv

#Активация
source venv/bin/activate
```
Установите все зависимости из файла
```
pip install -r requirements.txt
```
Запустите приложение
```
streamlit run src/app.py

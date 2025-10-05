### lmarena wrapper (en)

this is a small wrapper around lmarena.

cookies
- fill the required cookies in `lmarena.py` under `DEFAULT_COOKIES` before use.

functions
- ask: send a text prompt to a model id and get the response.
- generate_image: create an image from a prompt with an image-capable model.
- list_models: return formatted lines of model labels (optionally include ids). accepts optional `model_type` of `text` or `image`.

models
- models are read from `models.json`, to get new models extract them from <script> on lmarena.ai page.

known bugs:
- token updating is still not done, so you'll have to replace them manually once they expired, i will try to solve this soon.

### враппер над lmarena (ru)

это небольшой враппер над lmarena.

cookies
- заполните нужные куки в `lmarena.py` в `DEFAULT_COOKIES` перед использованием.

функции
- ask: отправляет текстовый запрос по id модели и возвращает ответ.
- generate_image: создаёт изображение по текстовому запросу для модели с поддержкой изображений.
- list_models: возвращает список отформатированных названий моделей (опционально с id). принимает необязательный `model_type`: `text` или `image`.

модели
- модели читаются из `models.json`, для получения нового списка вытащите его из <script> на lmarena.ai

известные баги:
- обновление cookies пока не реализовано, поэтому их придется обновлять вручную, постараюсь это реализовать в ближайшее время

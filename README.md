### lmarena wrapper (en)

this is a small wrapper around lmarena.

cookies
- set `COOKIES` in a local `.env` file to the exact raw cookie header string you copy from your browser DevTools (1:1). it will be sent as-is.

notes
- `COOKIES` - used for normal mode, sent as-is in the `Cookie` header
- `ANONYMOUS_MODE=true` - auto sign-up via 2captcha, caches session cookies
- `ANONYMOUS_COOKIES` - Cloudflare cookies used only during sign-up (if you get 429)
- `.env` is auto-loaded via `python-dotenv`
- ~~crimeflare~~ cloudflare is strict on the anonymous endpoints, so 9/10 times you'll need to insert the anonymous cookies (_cf_uam, cf_clearance)
- you can get the 2captcha token at https://2captcha.com/

functions
- ask: send a text prompt to a model id and get the response.
- generate_image: create an image from a prompt with an image-capable model.
- list_models: return formatted lines of model labels (optionally include ids). accepts optional `model_type` of `text` or `image`.

models
- models are read from `models.json`, to get new models extract them from <script> on lmarena.ai page.


### враппер над lmarena (ru)

это небольшой враппер над lmarena.

cookies
- заполните нужные куки в `lmarena.py` в `DEFAULT_COOKIES` перед использованием.

укажите `COOKIES` в `.env` как сырую строку заголовка Cookie (1:1) — она будет отправлена без изменений.


заметки
- `COOKIES` - для обычного режима, отправляется как есть в заголовке `Cookie`
- `ANONYMOUS_MODE=true` - авто-регистрация через 2captcha, кеширует куки сессии
- `ANONYMOUS_COOKIES` - Cloudflare куки только для sign-up (если получите 429)
- `.env` загружается автоматически через `python-dotenv`
- ~~crimeflare~~ cloudflare очень строго относится к запросам на анонимные эндпоинты, поэтому в 9/10 случаев вас будет блокировать, лучше сразу вставить куки (_cf_uam, cf_clearance)
- 2captcha токен можно получить на https://2captcha.com/

функции
- ask: отправляет текстовый запрос по id модели и возвращает ответ.
- generate_image: создаёт изображение по текстовому запросу для модели с поддержкой изображений.
- list_models: возвращает список отформатированных названий моделей (опционально с id). принимает необязательный `model_type`: `text` или `image`.

модели
- модели читаются из `models.json`, для получения нового списка вытащите его из <script> на lmarena.ai

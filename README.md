# Business Dossier Agent

ایجنتی که هر روز مقاله‌های تازهٔ ژورنال‌ها و نشریات مدیریتی (HBR، MIT Sloan، AMJ، SMJ، McKinsey و …) را از RSS می‌خواند، با مدل زبانی (از طریق AvalAI) خلاصه می‌کند و گزارش «Daily Strategic Intelligence Dossier» را به تلگرام می‌فرستد.

## نحوهٔ کار
1. دریافت مقاله‌های `DAYS_TO_CHECK` روز اخیر از فیدهای `JOURNAL_FEEDS` (با حذف موارد تکراری)
2. خلاصهٔ جداگانهٔ هر مقاله (Deep Dives)
3. نوشتن یک خلاصهٔ مدیریتی از کل خلاصه‌ها (Executive Summary)
4. ارسال به تلگرام در چند پیام، بدون شکستن تگ‌های HTML

## اجرا
روی GitHub Actions هر روز ساعت 16:30 UTC اجرا می‌شود (یا دستی از تب Actions). این secretها لازم‌اند:

| Secret | توضیح |
|---|---|
| `AVALAI_API_KEY` | کلید API سرویس AvalAI |
| `TELEGRAM_BOT_TOKEN` | توکن ربات تلگرام |
| `TELEGRAM_CHAT_ID` | شناسهٔ چت یا کانال مقصد |

اجرای محلی:
```bash
pip install -r requirements.txt
export AVALAI_API_KEY=... TELEGRAM_BOT_TOKEN=... TELEGRAM_CHAT_ID=...
python agent.py
```

متغیرهای اختیاری: `DAYS_TO_CHECK` (پیش‌فرض 1)، `MODEL_TO_USE` (پیش‌فرض `gpt-4o-mini`)، `API_DELAY_SECONDS` (پیش‌فرض 10).

اگر ارسال به تلگرام شکست بخورد یا کلید API نباشد، برنامه با کد خطا خارج می‌شود تا اجرای Actions قرمز شود.

## تست
```bash
pip install pytest
python -m pytest -q
```

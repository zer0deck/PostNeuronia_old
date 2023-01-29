from telegram import InlineKeyboardButton

MARK_1 = [
    ['🎑 Создать мем'],
    ['🧑‍💻 Обучить бота', '🛠 Тест сервера'],
    ['⚙ Саппорт', '💵 Донат'],
    ['ℹ Инфо', '🚪Завершить']
]
MARK_2 = [['🔙 Назад']]
MARK_3 = [['✅ Да', '⭕ Нет']]
MARK_4 = [
    [InlineKeyboardButton('Автор', url="https://t.me/zer0deck")],
    [InlineKeyboardButton('Админ', url="https://t.me/zarnitskiy")]
]
MARK_5 = [
    [InlineKeyboardButton('PayPal', url="https://paypal.me/grandilevskii")],
    [InlineKeyboardButton('СБП', url="https://www.tinkoff.ru/rm/grandilevskiy.aleksey1/yGNZr42325")], 
    [InlineKeyboardButton('Crypto (USTD)', callback_data=str(99))]
]
MARK_6 = [[InlineKeyboardButton('Леха', url="https://t.me/zer0deck")]]
MARK_7 = [
    ['🧾 Добавить обучающих данных'],
    ['🤹 Я программист, хочу помочь'],
    ['🔙 Назад']
]
MARK_8 = [
    ['🔙 Назад'], 
    ['🚪Главное меню']
]
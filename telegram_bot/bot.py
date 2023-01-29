# pylint:disable=[all]

########################################
# IMPORTS
########################################

import logging
from network import generate_image
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update, ReplyKeyboardMarkup, ReplyKeyboardRemove
from telegram.ext import (
    MessageHandler,
    filters,
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    ConversationHandler,
    ApplicationBuilder
)
from typing import Dict
from messages import (
    MESSAGE_1,
    MESSAGE_2,
    MESSAGE_3,
    MESSAGE_4,
    MESSAGE_5,
    MESSAGE_6,
    MESSAGE_7,
    MESSAGE_8,
    MESSAGE_9,
    MESSAGE_10,
    MESSAGE_11,
    MESSAGE_12,
    MESSAGE_13,
    MESSAGE_14,
    MESSAGE_15,
    MESSAGE_16,
    MESSAGE_17
)
from marks import (
    MARK_1,
    MARK_2,
    MARK_3,
    MARK_4,
    MARK_5,
    MARK_6,
    MARK_7,
    MARK_8
)

########################################
# FLAGS
########################################

MAIN_MENU, BACK_SELECTOR, LOAD_PHOTO, LOAD_TEXT, TRAINER_SELECTOR, YES_NO_SELECTOR, SAVE_SELECTOR, GENERATE, GENERATE_MORE = range(9)
TIMEOUT = 120.

########################################
# LOGGING
########################################

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

########################################
# MARKUPS
########################################

main_menu_markup = ReplyKeyboardMarkup(MARK_1, one_time_keyboard=True)
back_menu_markup = ReplyKeyboardMarkup(MARK_2, one_time_keyboard=True)
agree_menu_markup = ReplyKeyboardMarkup(MARK_3, one_time_keyboard=True)
train_menu_markup = ReplyKeyboardMarkup(MARK_7, one_time_keyboard=True)
contact_dev_menu_markup = ReplyKeyboardMarkup(MARK_8, one_time_keyboard=True)

support_links_markup = InlineKeyboardMarkup(MARK_4)
payment_links_markup = InlineKeyboardMarkup(MARK_5)
dev_links_markup = InlineKeyboardMarkup(MARK_6)

########################################
# TECH
########################################

def remove_job_if_exists(name: str, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Remove job with given name. Returns whether job was removed."""
    current_jobs = context.job_queue.get_jobs_by_name(name)
    if not current_jobs:
        return None
    for job in current_jobs:
        job.schedule_removal()

async def timeout(context: ContextTypes.DEFAULT_TYPE) -> None:
    await context.bot.send_message(
        context.job.chat_id,
        text=MESSAGE_6,
        reply_markup=ReplyKeyboardRemove(),
    )

async def return_main_menu(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    chat_id = update.effective_message.chat_id
    remove_job_if_exists(str(chat_id), context)
    context.job_queue.run_once(timeout, TIMEOUT, chat_id=chat_id, name=str(chat_id))    
    await update.message.reply_text(
        text='Возврат в главное меню',
        reply_markup=main_menu_markup,
        parse_mode='HTML'
    )

    return MAIN_MENU

########################################
# START/EXIT
########################################

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    chat_id = update.effective_message.chat_id
    remove_job_if_exists(str(chat_id), context)
    context.job_queue.run_once(timeout, TIMEOUT, chat_id=chat_id, name=str(chat_id))
    await update.message.reply_text(
        text=MESSAGE_1,
        reply_markup=main_menu_markup,
        parse_mode='HTML'
    )
    
    return MAIN_MENU

async def done(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    remove_job_if_exists(str(update.effective_message.chat_id), context)
    await update.message.reply_text(
        text=MESSAGE_11,
        reply_markup=ReplyKeyboardRemove(),
    )

    return ConversationHandler.END

########################################
# MAIN MENU
########################################

async def server_test(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    chat_id = update.effective_message.chat_id
    remove_job_if_exists(str(chat_id), context)
    context.job_queue.run_once(timeout, TIMEOUT, chat_id=chat_id, name=str(chat_id))
    await update.message.reply_text(
        text='SERVER_MESSAGE',
        parse_mode='HTML',
        reply_markup=main_menu_markup       
    )

    return MAIN_MENU

async def support(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    chat_id = update.effective_message.chat_id
    remove_job_if_exists(str(chat_id), context)
    context.job_queue.run_once(timeout, TIMEOUT, chat_id=chat_id, name=str(chat_id))
    await update.message.reply_text(
        text=MESSAGE_3, 
        reply_markup=support_links_markup,
        parse_mode='HTML'
    )
    await update.message.reply_text(
        text=MESSAGE_12, 
        reply_markup=back_menu_markup,
        parse_mode='HTML'
    )

    return BACK_SELECTOR

async def pay(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    chat_id = update.effective_message.chat_id
    remove_job_if_exists(str(chat_id), context)
    context.job_queue.run_once(timeout, TIMEOUT, chat_id=chat_id, name=str(chat_id))
    await update.message.reply_text(
        text=MESSAGE_2, 
        reply_markup=payment_links_markup,
        parse_mode='HTML'
    )
    await update.message.reply_text(
        text=MESSAGE_12, 
        reply_markup=back_menu_markup,
        parse_mode='HTML'
    )

    return BACK_SELECTOR

########################################
# TRAINER
########################################

async def trainer(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    chat_id = update.effective_message.chat_id
    remove_job_if_exists(str(chat_id), context)
    context.job_queue.run_once(timeout, TIMEOUT, chat_id=chat_id, name=str(chat_id))
    await update.message.reply_text(
        text=MESSAGE_5, 
        reply_markup=train_menu_markup,
        parse_mode='HTML'
    )

    return BACK_SELECTOR

async def add_data(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    chat_id = update.effective_message.chat_id
    remove_job_if_exists(str(chat_id), context)
    context.job_queue.run_once(timeout, TIMEOUT, chat_id=chat_id, name=str(chat_id))
    await update.message.reply_text(
            text=MESSAGE_13, 
            reply_markup=ReplyKeyboardRemove(),
            parse_mode='HTML'
        )

    return LOAD_PHOTO

async def load_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    chat_id = update.effective_message.chat_id
    remove_job_if_exists(str(chat_id), context)
    context.job_queue.run_once(timeout, TIMEOUT, chat_id=chat_id, name=str(chat_id))

    try:
        photo_file = await update.message.photo[-1].get_file()
        context.user_data['train_image_id'] = str(id(photo_file)) + '.jpg'
        await photo_file.download_to_drive(f"trainer/{context.user_data['train_image_id']}")

        await update.message.reply_text(
            text=MESSAGE_15, 
            reply_markup=ReplyKeyboardRemove(),
            parse_mode='HTML'
        )        

        return LOAD_TEXT

    except (IndexError, ValueError):
        remove_job_if_exists(str(chat_id), context) 
        context.job_queue.run_once(timeout, TIMEOUT, chat_id=chat_id, name=str(chat_id))

        await context.bot.send_message(
            chat_id=chat_id,
            text=MESSAGE_8, 
            parse_mode='HTML'
        )

        return BACK_SELECTOR

async def load_description(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    chat_id = update.effective_message.chat_id
    remove_job_if_exists(str(chat_id), context)
    context.job_queue.run_once(timeout, TIMEOUT, chat_id=chat_id, name=str(chat_id))
    try: 
        text = update.message.text
        logger.info('User send %s', text)

        await context.bot.send_photo(
            chat_id=chat_id,
            photo = open(context.user_data['train_image_id'], "rb")
        )
        await update.message.reply_text(
            text=f'<b>Ты прислал нам:</b> \n<em>"{text}"</em> \n\nСохранить?', 
            reply_markup=agree_menu_markup,
            parse_mode='HTML'
        )

        return SAVE_SELECTOR

    except (IndexError, ValueError):
        remove_job_if_exists(str(chat_id), context) 
        context.job_queue.run_once(timeout, TIMEOUT, chat_id=chat_id, name=str(chat_id))

        await context.bot.send_message(
            chat_id=chat_id,
            text=MESSAGE_8, 
            parse_mode='HTML'
        )

        return BACK_SELECTOR

async def load_error(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    chat_id = update.effective_message.chat_id
    remove_job_if_exists(str(chat_id), context) 
    context.job_queue.run_once(timeout, TIMEOUT, chat_id=chat_id, name=str(chat_id))

    await update.message.reply_text(
        text=MESSAGE_8, 
        reply_markup=train_menu_markup,
        parse_mode='HTML'
    )

    return BACK_SELECTOR   

async def save(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    chat_id = update.effective_message.chat_id
    remove_job_if_exists(str(chat_id), context)
    context.job_queue.run_once(timeout, TIMEOUT, chat_id=chat_id, name=str(chat_id))    
    t_t:str = MESSAGE_9

    if update.message.text =='Да':
        t_t = t_t.replace('SCP-579: <span class="tg-spoiler">[ДАННЫЕ УДАЛЕНЫ]</span>', 'Вроде готово ✅', 1)
        
    await update.message.reply_text(
        text=t_t, 
        reply_markup=agree_menu_markup,
        parse_mode='HTML'
    )

    return YES_NO_SELECTOR

async def thank_for_data(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    chat_id = update.effective_message.chat_id
    remove_job_if_exists(str(chat_id), context)
    context.job_queue.run_once(timeout, TIMEOUT, chat_id=chat_id, name=str(chat_id))    

    await update.message.reply_text(
        text=MESSAGE_10,
        reply_markup=main_menu_markup,
        parse_mode='HTML'
    )
    return MAIN_MENU

async def link_to_dev(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:

    chat_id = update.effective_message.chat_id
    remove_job_if_exists(str(chat_id), context)
    context.job_queue.run_once(timeout, TIMEOUT, chat_id=chat_id, name=str(chat_id))    
    await update.message.reply_text(
        text=MESSAGE_14, 
        reply_markup=contact_dev_menu_markup,
        parse_mode='HTML'
    )

    return TRAINER_SELECTOR

########################################
# GENERATOR
########################################

async def generate_menu(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    chat_id = update.effective_message.chat_id
    remove_job_if_exists(str(chat_id), context)
    context.job_queue.run_once(timeout, TIMEOUT, chat_id=chat_id, name=str(chat_id))
    await update.message.reply_text(
            text=MESSAGE_4, 
            reply_markup=back_menu_markup,
            parse_mode='HTML'
        )

    return GENERATE


async def generate(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    chat_id = update.effective_message.chat_id
    remove_job_if_exists(str(chat_id), context)
    context.job_queue.run_once(timeout, TIMEOUT, chat_id=chat_id, name=str(chat_id))

    try:
        photo_file = await update.message.photo[-1].get_file()
        context.user_data['generate_image_id'] = str(id(photo_file)) + '.jpg'
        await photo_file.download_to_drive(f"uploaded/{context.user_data['generate_image_id']}")
        await generate_image(context.user_data['generate_image_id'])
        await context.bot.send_photo(
            chat_id=chat_id,
            photo = open(f"generated/{context.user_data['generate_image_id']}", "rb")
        )        
        await update.message.reply_text(
            text='Вот твой мем. <b>Хочешь сделаю еще?</b>', 
            reply_markup=agree_menu_markup,
            parse_mode='HTML'
        )        

        return GENERATE_MORE

    except (IndexError, ValueError):
        remove_job_if_exists(str(chat_id), context) 
        context.job_queue.run_once(timeout, TIMEOUT, chat_id=chat_id, name=str(chat_id))

        await context.bot.send_message(
            chat_id=chat_id,
            text=MESSAGE_17, 
            parse_mode='HTML'
        )

        return GENERATE

async def generate_error(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    chat_id = update.effective_message.chat_id
    remove_job_if_exists(str(chat_id), context) 
    context.job_queue.run_once(timeout, TIMEOUT, chat_id=chat_id, name=str(chat_id))

    await update.message.reply_text(
        text=MESSAGE_17, 
        reply_markup=back_menu_markup,
        parse_mode='HTML'
    )

    return GENERATE

async def thank_for_generating(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_message.chat_id
    remove_job_if_exists(str(chat_id), context)
    context.job_queue.run_once(timeout, TIMEOUT, chat_id=chat_id, name=str(chat_id))    

    await update.message.reply_text(
        text=MESSAGE_16,
        reply_markup=main_menu_markup,
        parse_mode='HTML'
    )

    return MAIN_MENU

########################################
# RUN
########################################

if __name__ == '__main__':
    # token
    application = ApplicationBuilder().token('TOKEN').build()

    job_queue = application.job_queue

    main_handler  = ConversationHandler(
        entry_points=[CommandHandler("start", start)],
        states={
            MAIN_MENU: [
                MessageHandler(filters.Regex("^ℹ Инфо$"), start),
                MessageHandler(filters.Regex("^🛠 Тест сервера$"), server_test),
                MessageHandler(filters.Regex("^⚙ Саппорт$"), support),
                MessageHandler(filters.Regex("^💵 Донат$"), pay),
                MessageHandler(filters.Regex("^🎑 Создать мем$"), generate_menu),
                MessageHandler(filters.Regex("^🧑‍💻 Обучить бота$"), trainer)
            ],
            BACK_SELECTOR: [
                MessageHandler(filters.Regex("^🔙 Назад$"), return_main_menu),
                MessageHandler(filters.Regex("^🧾 Добавить обучающих данных$"), add_data),
                MessageHandler(filters.Regex("^🤹 Я программист, хочу помочь$"), link_to_dev)
            ],
            TRAINER_SELECTOR: [
                MessageHandler(filters.Regex("^🔙 Назад$"), trainer),
                MessageHandler(filters.Regex("^🚪Главное меню$"), return_main_menu)
            ],
            LOAD_PHOTO: [
                MessageHandler(filters.PHOTO, load_photo), 
                MessageHandler(filters.TEXT, load_error),
            ],
            LOAD_TEXT: [
                MessageHandler(filters.PHOTO, load_error), 
                MessageHandler(filters.TEXT, load_description),                
            ],
            GENERATE: [
                MessageHandler(filters.PHOTO, generate), 
                MessageHandler(filters.TEXT & ~(filters.Regex("^🔙 Назад$")), generate_error),   
                MessageHandler(filters.Regex("^🔙 Назад$"), return_main_menu),           
            ],
            GENERATE_MORE: [
                MessageHandler(filters.Regex("^⭕ Нет$"), thank_for_generating),
                MessageHandler(filters.Regex("^✅ Да$"), generate_menu)
            ],
            SAVE_SELECTOR: [
                MessageHandler(filters.TEXT, save),
            ],
            YES_NO_SELECTOR: [
                MessageHandler(filters.Regex("^⭕ Нет$"), thank_for_data),
                MessageHandler(filters.Regex("^✅ Да$"), add_data)
            ]
        },
        fallbacks=[MessageHandler(filters.Regex("^🚪Завершить$"), done)],
        conversation_timeout=TIMEOUT, 
    )


    application.add_handler(main_handler)
    application.run_polling()
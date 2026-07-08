import logging
import os
import shutil
import functools

# Добавляем новые уровни (должны быть уникальными!)
CONFIG_LEVEL = 21 # 1
CRISTAE_LEVEL = 11
DRAW_LEVEL = 12
SHELL_LEVEL = 13
CELL_LEVEL = 14
ORGANELLE_LEVEL = 31
MAIN_LEVEL = 32

"""
Стандартные уровни logging:
DEBUG    — 10
INFO     — 20
WARNING  — 30
ERROR    — 40
CRITICAL — 50
NOTSET   — 0  (специальный уровень для наследования настроек, не используется напрямую для сообщений).
"""

# Глобальный объект логгера
logger_name = "generator_logger"
# Глобальный логгер
logger = None

# ANSI-коды цветов
LOG_COLORS = {
    'CONFIG':    '\033[36m',  # Циан
    'DRAW':      '\033[33m',  # Желтый
    'CRISTAE':   '\033[38;2;100;100;100m',  # Серый
    'ORGANELLE': '\033[38;2;200;200;200m',  # Белый
    'MAIN':      '\033[38;2;200;200;200m',  # Белый
    'CELL':      '\033[33m',  # Желтый
    'SHELL':     '\033[33m',  # Желтый
    'DEBUG':     '\033[36m',  # Циан
    'INFO':      '\033[32m',  # Зеленый
    'WARNING':   '\033[33m',  # Желтый
    'ERROR':     '\033[31m',  # Красный
    'CRITICAL':  '\033[1;31m' # Жирный красный
}
RESET_COLOR = '\033[0m'

class ColorFormatter(logging.Formatter):
    def format(self, record):
        color = LOG_COLORS.get(record.levelname, '')
        message = super().format(record)
        return f"{color}{message}{RESET_COLOR}"


def add_user_lavelnames():
    global logger
    logging.addLevelName(CRISTAE_LEVEL, "CRISTAE")
    logging.addLevelName(CONFIG_LEVEL, "CONFIG")
    logging.addLevelName(DRAW_LEVEL, "DRAW")
    logging.addLevelName(ORGANELLE_LEVEL, "ORGANELLE")
    logging.addLevelName(MAIN_LEVEL, "MAIN")
    logging.addLevelName(SHELL_LEVEL, "SHELL")
    logging.addLevelName(CELL_LEVEL, "CELL")

    # Расширяем логгер для новых методов
    def config(self, message, *args, **kwargs):
        if self.isEnabledFor(CONFIG_LEVEL):
            self._log(CONFIG_LEVEL, message, args, **kwargs)

    def draw(self, message, *args, **kwargs):
        if self.isEnabledFor(DRAW_LEVEL):
            self._log(DRAW_LEVEL, message, args, **kwargs)

    def organelle(self, message, *args, **kwargs):
        if self.isEnabledFor(ORGANELLE_LEVEL):
            self._log(ORGANELLE_LEVEL, message, args, **kwargs)

    def main(self, message, *args, **kwargs):
        if self.isEnabledFor(MAIN_LEVEL):
            self._log(MAIN_LEVEL, message, args, **kwargs)

    def shell(self, message, *args, **kwargs):
        if self.isEnabledFor(SHELL_LEVEL):
            self._log(SHELL_LEVEL, message, args, **kwargs)

    def cell(self, message, *args, **kwargs):
        if self.isEnabledFor(CELL_LEVEL):
            self._log(CELL_LEVEL, message, args, **kwargs)

    def cristae(self, message, *args, **kwargs):
        if self.isEnabledFor(CRISTAE_LEVEL):
            self._log(CRISTAE_LEVEL, message, args, **kwargs)

    # Добавляем методы к логгеру
    logger.config = config.__get__(logger)
    logger.draw = draw.__get__(logger)
    logger.organelle = organelle.__get__(logger)
    logger.main = main.__get__(logger)
    logger.shell = shell.__get__(logger)
    logger.cell = cell.__get__(logger)
    logger.cristae = cristae.__get__(logger)

def init_custom_logger():
    global logger
    # Проверка, есть ли уже логгер с этим именем
    if logger is None:
        logger = logging.getLogger(logger_name)
        logger.handlers.clear()

        # Создаем или получаем логгер
        logger.setLevel(logging.DEBUG)

        # Создаем обработчик
        handler = logging.StreamHandler()
        formatter = ColorFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        # Добавляем пользовательские уровни
        add_user_lavelnames()

        # Обработчик для файла по умолчанию
        default_log_path = os.path.join(os.getcwd(), 'generate.log')
        file_handler = logging.FileHandler(default_log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)

        # Помечаем, что логгер инициализирован
        logger.initialized = True

init_custom_logger()

# Функция для отключения уровня по имени или номеру
def logging_disable_level(level_name_or_number):
    global logger
    if logger is None:
        init_custom_logger()
    # Определяем уровень
    level = None
    if isinstance(level_name_or_number, int):
        level = level_name_or_number
    elif isinstance(level_name_or_number, str):
        # Попытка найти уровень по имени
        name_upper = level_name_or_number.upper()
        level = logging._nameToLevel.get(name_upper, None)
        if level is None:
            # Попытка преобразовать строку в число
            try:
                level = int(level_name_or_number)
            except:
                logger.error(f"Не удалось найти уровень по имени '{level_name_or_number}' в logger")
                return
    else:
        logger.error("Недопустимый тип аргумента logger")
        return

    # Создаем фильтр, который пропускает все уровни, кроме отключаемого
    class DisableLevelFilter(logging.Filter):
        def filter(self, record):
            return record.levelno != level

    # Добавляем фильтр в все обработчики логгера
    for h in logger.handlers:
        h.addFilter(DisableLevelFilter())

def set_logger_level(level):
    """
    Устанавливает уровень логгирования для глобального логгера.
    `level` — целое число или строка уровня.
    """
    global logger
    if logger is None:
        init_custom_logger()
    if isinstance(level, str):
        # Попытка преобразовать строку в уровень
        level_name = level.upper()
        numeric_level = logging._nameToLevel.get(level_name, None)
        if numeric_level is None:
            raise ValueError(f"Некорректный уровень логирования: {level}")
        level = numeric_level
    logger.setLevel(level)

def set_log_file_path(path):
    """
    Заменяет файл логирования на указанный путь.
    """
    global logger
    if logger is None:
        init_custom_logger()

    # Находим существующий файловый обработчик
    old_handler = None
    for h in logger.handlers:
        if isinstance(h, logging.FileHandler):
            old_handler = h
            break

    old_log_path = None
    if old_handler:
        old_log_path = old_handler.baseFilename

    new_path = os.path.abspath(path)

    # Удаляем существующий файловый обработчик
    for h in logger.handlers:
        if isinstance(h, logging.FileHandler):
            logger.removeHandler(h)
            h.close()

    # Если старый лог-файл существует, копируем его содержимое в новый
    if old_log_path and os.path.exists(old_log_path):
        # Убедимся, что новая папка существует
        os.makedirs(os.path.dirname(new_path), exist_ok=True)
        # Переносим содержимое
        shutil.copy2(old_log_path, new_path)
        # Удаляем старый файл
        os.remove(old_log_path)

    # Добавляем новый обработчик файла
    new_path = os.path.abspath(path)
    new_file_handler = logging.FileHandler(new_path)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    new_file_handler.setFormatter(formatter)
    logger.addHandler(new_file_handler)

def clear_log_file(log_file_path=None):
    if log_file_path is None:
        log_file_path = os.path.join(os.getcwd(), 'generate.log')
    with open(log_file_path, 'w'):
        pass  # Открываем файл в режиме записи, чтобы очистить содержимое

def log_execution(level='info'):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            log_method = getattr(logger, level, logger.info)
            log_method(f"Начало выполнения {func.__name__}")
            result = func(self, *args, **kwargs)
            log_method(f"Завершение выполнения {func.__name__}")
            return result
        return wrapper
    return decorator

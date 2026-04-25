
from Synthetic3D.src.utilities.logging_config import logger, clear_log_file, logging_disable_level

# Пример использования
if __name__ == "__main__":
    clear_log_file()
    # Логируем разные уровни
    logger.debug("Это DEBUG")
    logger.info("Это INFO")
    logger.draw("Это DRAW")
    logger.organelle("Это ORGANELLE")
    logger.main("Это MAIN")
    logger.warning("Это WARNING")
    logger.error("Это ERROR")

    # Отключаем уровень DRAW
    logging_disable_level("DRAW")
    logger.draw("Это DRAW после отключения")  # не должно отображаться
    logger.info("Это INFO после отключения DRAW")  # должно отображаться

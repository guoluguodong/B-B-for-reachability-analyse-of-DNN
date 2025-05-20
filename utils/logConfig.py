import logging.config
import configparser
def load_logging_config(log_path):
    config = configparser.ConfigParser()
    config.read('utils/logging.conf')

    # 修改路径
    config.set('handler_fileHandler', 'args', f"('{log_path}', 'w')")

    # 将修改后的配置写入临时文件
    with open('temp_logging.conf', 'w') as temp_conf:
        config.write(temp_conf)

    # 加载配置
    logging.config.fileConfig('temp_logging.conf', disable_existing_loggers=False)



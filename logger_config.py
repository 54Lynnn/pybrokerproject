# -*- coding: utf-8 -*-
"""日志配置模块 — 统一管理项目日志"""

import logging
import os
import sys
from datetime import datetime


def setup_logger(name=None, level=logging.INFO, log_to_file=True, log_dir=None):
    """
    配置并返回一个logger实例。
    
    参数:
        name: logger名称，None则返回root logger
        level: 日志级别 (DEBUG/INFO/WARNING/ERROR)
        log_to_file: 是否同时写入日志文件
        log_dir: 日志文件存放目录
    
    返回:
        logging.Logger: 配置好的logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # 避免重复添加handler
    if logger.handlers:
        return logger
    
    # 统一格式
    formatter = logging.Formatter(
        '[%(asctime)s] [%(levelname)s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 控制台输出
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 文件输出
    if log_to_file:
        if log_dir is None:
            log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
        os.makedirs(log_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d')
        log_file = os.path.join(log_dir, f'strategy_{timestamp}.log')
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        # 记录日志文件位置
        logger.info(f"日志文件: {log_file}")
    
    return logger


def get_logger(name=None):
    """获取已配置的logger，如未配置则自动初始化。"""
    logger = logging.getLogger(name)
    if not logger.handlers:
        return setup_logger(name)
    return logger

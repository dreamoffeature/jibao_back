# coding=utf-8
import sys
import json
import base64
import re
from typing import Optional, Dict

import cv2
import numpy as np
from urllib.request import urlopen, Request
from urllib.error import URLError
from urllib.parse import urlencode
from collections import defaultdict
import difflib

# 兼容Python2和Python3
IS_PY3 = sys.version_info.major == 3

# 忽略HTTPS证书验证
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

# 百度OCR配置
API_KEY = 'IlquMzBntv5vGDv6AIesxzWQ'
SECRET_KEY = '5F6xm3QO35FgP4mkoZQTm7R5njHCuhKp'

# 百度OCR接口地址
TOKEN_URL = 'https://aip.baidubce.com/oauth/2.0/token'
OCR_ACCURATE_URL = "https://aip.baidubce.com/rest/2.0/ocr/v1/accurate_basic"
OCR_FORM_URL = "https://aip.baidubce.com/rest/2.0/ocr/v1/table"


class ImagePreprocessor:
    """图像预处理类"""

    @staticmethod
    def preprocess_for_ocr(image_path):
        """OCR图像预处理"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                print(f"无法读取图像: {image_path}")
                return None

            # 统一图像尺寸
            height, width = image.shape[:2]
            if max(height, width) > 2000:
                scale = 2000 / max(height, width)
                new_width = int(width * scale)
                new_height = int(height * scale)
                image = cv2.resize(image, (new_width, new_height))

            # 灰度化
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image

            # 噪声去除
            denoised = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)

            # 对比度增强
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(denoised)

            # 锐化处理
            kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            sharpened = cv2.filter2D(enhanced, -1, kernel)

            # 转换为RGB
            result_image = cv2.cvtColor(sharpened, cv2.COLOR_GRAY2RGB)

            success, encoded_image = cv2.imencode('.jpg', result_image,
                                                  [cv2.IMWRITE_JPEG_QUALITY, 90])
            if success:
                image_base64 = base64.b64encode(encoded_image).decode('utf-8')
                return image_base64
            return None

        except Exception as e:
            print(f"图像预处理失败: {e}")
            return None

    @staticmethod
    def preprocess_for_table(image_path):
        """表格识别专用预处理"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                return None

            # 统一尺寸
            height, width = image.shape[:2]
            if max(height, width) > 1500:
                scale = 1500 / max(height, width)
                new_size = (int(width * scale), int(height * scale))
                image = cv2.resize(image, new_size)

            # 灰度化
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # 表格专用处理
            denoised = cv2.fastNlMeansDenoising(gray, None, h=5,
                                                templateWindowSize=5,
                                                searchWindowSize=15)

            # 轻微锐化
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
            sharpened = cv2.filter2D(denoised, -1, kernel)

            # 转换为RGB
            result_image = cv2.cvtColor(sharpened, cv2.COLOR_GRAY2RGB)

            success, encoded_image = cv2.imencode('.jpg', result_image,
                                                  [cv2.IMWRITE_JPEG_QUALITY, 95])
            if success:
                return base64.b64encode(encoded_image).decode('utf-8')
            return None

        except Exception as e:
            print(f"表格图像预处理失败: {e}")
            return None


class FormatDetector:
    """定值单格式检测器"""

    def __init__(self):
        self.format_patterns = {
            'format1': {
                'name': '定值名称+定值格式',
                'keywords': ['定值名称', '定值', '备注'],
            },
            'format2': {
                'name': '保护功能+名称+定值格式',
                'keywords': ['新列','保护功能', '名称', '定值', '备注'],
            }
        }

    def detect_format(self, ocr_text: str, table_data: list = None) -> str:
        """检测定值单格式"""
        text_lower = ocr_text.lower()

        # 基于表格结构检测
        if table_data:
            for table in table_data:
                headers = self._extract_table_headers(table)
                headers_text = ' '.join(headers).lower()

                # 检查格式1特征
                if any(keyword in headers_text for keyword in ['功能类型', '定值名称']):
                    return 'format1'
                # 检查格式2特征
                elif any(keyword in headers_text for keyword in ['保护功能', '名称']):
                    return 'format2'

        # 基于文本内容检测
        if any(keyword in text_lower for keyword in ['功能类型', '定值名称']):
            return 'format1'
        elif any(keyword in text_lower for keyword in ['保护功能', '名称']):
            return 'format2'
        else:
            return 'unknown'

    def _extract_table_headers(self, table: dict) -> list:
        """提取表格表头"""
        headers = []
        header_cells = table.get('header', [])
        if not isinstance(header_cells, list):
            header_cells = [header_cells]

        for cell in header_cells:
            if isinstance(cell, dict) and 'words' in cell:
                headers.append(cell['words'].strip())
        return headers



class KeywordBasedParameterMapper:
    """基于关键词模糊匹配的参数映射器 - 支持防重复映射"""

    def __init__(self):
        # 定义标准参数的关键词模式
        self.parameter_patterns = {'line_10': {
            # 过流保护相关
            '过流I段电流定值': {
                'must': ['过流'],
                'first': ['Ⅰ段', 'I段', '1段', '一段','过流段'],
                'primary': ['过流', '电流'],
                'secondary': ['定值', '值', '设置'],
                'exclude': ['时间', '时限', 'Ⅱ段', 'Ⅱ段', '2段', '二段', 'Ⅲ段', 'III段', '3段', '三段']
            },
            '过流I段时限定值': {
                'must': ['过流'],
                'first': ['Ⅰ段', 'I段', '1段', '一段','过流段'],
                'primary': ['过流', '时间', '时限'],
                'secondary': ['定值', '值', '设置'],
                'exclude': ['电流', 'Ⅱ段', 'Ⅱ段', '2段', '二段', 'Ⅲ段', 'III段', '3段', '三段', '反时限']
            },
            '过流II段电流定值': {
                'must': ['过流'],
                'first': ['Ⅱ段', 'II段', '2段', '二段'],
                'primary': ['过流', '电流'],
                'secondary': ['定值', '值', '设置'],
                'exclude': ['时间', '时限', 'Ⅰ段', 'I段', '1段', '一段', 'Ⅲ段', 'III段', '3段', '三段']
            },
            '过流II段时限定值': {
                'must': ['过流'],
                'first': ['Ⅱ段', 'II段', '2段', '二段'],
                'primary': ['过流', '时间', '时限'],
                'secondary': ['定值', '值', '设置'],
                'exclude': ['电流', 'Ⅰ段', 'I段', '1段', '一段', 'Ⅲ段', 'III段', '3段', '三段', '反时限']
            },
            '过流III段电流定值': {
                'must': ['过流'],
                'first': ['Ⅲ段', 'III段', '3段', '三段'],
                'primary': ['过流', '电流'],
                'secondary': ['定值', '值', '设置'],
                'exclude': ['时间', '时限', 'Ⅰ段', 'I段', '1段', '一段', 'Ⅱ段', 'II段', '2段', '二段']
            },
            '过流III段时限定值': {
                'must': ['过流'],
                'first': ['Ⅲ段', 'III段', '3段', '三段'],
                'primary': ['过流', '时间', '时限'],
                'secondary': ['定值', '值', '设置'],
                'exclude': ['电流', 'Ⅰ段', 'I段', '1段', '一段', 'Ⅱ段', 'II段', '2段', '二段', '反时限']
            },
            # 零序保护
            '零序过流I段电流定值': {
                'must': ['零'],
                'first': ['Ⅰ段', 'I段', '1段', '一段'],
                'primary': ['零序','零流','过流', '电流'],
                'secondary': ['定值', '值'],
                'exclude': ['时间', '时限', 'Ⅱ段', 'II段', '2段', '二段', 'Ⅲ段', 'III段', '3段', '三段']
            },
            '零序过流I段时限定值': {
                'must': ['零'],
                'first': ['Ⅰ段', 'I段', '1段', '一段'],
                'primary': ['零序','零流','过流', '时间', '时限'],
                'secondary': ['定值', '值'],
                'exclude': ['电流', 'Ⅰ段', 'I段', '1段', '一段', 'Ⅲ段', 'III段', '3段', '三段', '反时限']
            },
            '零序过流II段电流定值': {
                'must': ['零'],
                'first': ['Ⅱ段', 'II段', '2段', '二段'],
                'primary': ['零序','零流','过流', '电流'],
                'secondary': ['定值', '值'],
                'exclude': ['时间', '时限', 'Ⅰ段', 'I段', '1段', '一段', 'Ⅲ段', 'III段', '3段', '三段']
            },
            '零序过流II段时限定值': {
                'must': ['零'],
                'first': ['Ⅱ段', 'II段', '2段', '二段'],
                'primary': [ '零序','零流','过流', '时间', '时限'],
                'secondary': ['定值', '值'],
                'exclude': ['电流', 'Ⅰ段', 'I段', '1段', '一段', 'Ⅲ段', 'III段', '3段', '三段', '反时限']
            },
            '零序过流III段电流定值': {
                'must': ['零'],
                'first': ['Ⅲ段', 'III段', '3段', '三段'],
                'primary': ['零序','零流','过流', '电流'],
                'secondary': ['定值', '值'],
                'exclude': ['时间', '时限', 'Ⅰ段', 'I段', '1段', '一段', 'Ⅱ段', 'II段', '2段', '二段']
            },
            '零序过流III段时限定值': {
                'must': ['零'],
                'first': ['Ⅲ段', 'III段', '3段', '三段'],
                'primary': [ '零序','零流','过流', '时间', '时限',],
                'secondary': ['定值', '值'],
                'exclude': ['电流', 'Ⅰ段', 'I段', '1段', '一段', 'Ⅱ段', 'II段', '2段', '二段', '反时限']
            }
        },'line':{
            '零序补偿系数KZ': {
                'must': ['零序', '补偿'],
                'first': ['KZ','电抗'],
                'primary': ['零序', '补偿', '系数'],
                'secondary': ['KZ', '值', '参数'],
                'exclude': ['电流', '时间', '距离', '过流', '段','电阻']
            },
            '差动动作电流定值': {
                'must': ['差动'],
                'first': ['动作电流', '电流'],
                'primary': ['差动', '动作', '电流'],
                'secondary': ['定值', '值', '设置'],
                'exclude': ['时间', '时限', '补偿', '系数', '距离', '过流', '段']
            },
            '线路正序灵敏角': {
                'must': ['线路', '正序'],
                'first': ['灵敏角', '角度'],
                'primary': ['线路', '正序', '灵敏角'],
                'secondary': ['值', '角度', '参数'],
                'exclude': ['电流', '时间', '补偿', '系数', '距离', '过流', '段']
            },
            'CT变比': {
                'must': ['CT'],
                'first': ['变比'],
                'primary': ['CT', '变比'],
                'secondary': ['值', '参数', '比值'],
                'exclude': ['电流', '时间', '补偿', '系数', '距离', '过流', '段']
            },
            '接地距离I段定值': {
                'must': ['接地', '距离'],
                'first': ['Ⅰ段', 'I段', '1段', '一段'],
                'primary': ['接地', '距离', '段','阻抗'],
                'secondary': ['定值', '值', '设置'],
                'exclude': ['电流', '时间', 'Ⅱ段', 'II段', '2段', '二段', 'Ⅲ段', 'III段', '3段', '三段','相间']
            },
            '接地距离I段时间': {
                'must': ['接地', '距离'],
                'first': ['Ⅰ段', 'I段', '1段', '一段'],
                'primary': ['接地', '距离', '时间', '时限'],
                'secondary': ['值', '设置', '参数'],
                'exclude': ['电流','Ⅱ段', 'II段', '2段', '二段', 'Ⅲ段', 'III段', '3段', '三段', '反时限','相间']
            },
            '接地距离II段定值': {
                'must': ['接地', '距离'],
                'first': ['Ⅱ段', 'II段', '2段', '二段'],
                'primary': ['接地', '距离', '段','阻抗'],
                'secondary': ['定值', '值', '设置'],
                'exclude': ['电流', '时间', 'Ⅰ段', 'I段', '1段', '一段', 'Ⅲ段', 'III段', '3段', '三段','相间']
            },
            '接地距离II段时间': {
                'must': ['接地', '距离'],
                'first': ['Ⅱ段', 'II段', '2段', '二段'],
                'primary': ['接地', '距离', '时间', '时限'],
                'secondary': ['值', '设置', '参数'],
                'exclude': ['电流', 'Ⅰ段', 'I段', '1段', '一段', 'Ⅲ段', 'III段', '3段', '三段', '反时限','相间']
            },
            '接地距离III段定值': {
                'must': ['接地', '距离'],
                'first': ['Ⅲ段', 'III段', '3段', '三段'],
                'primary': ['接地', '距离', '段','阻抗'],
                'secondary': ['定值', '值', '设置'],
                'exclude': ['电流', '时间', 'Ⅰ段', 'I段', '1段', '一段', 'Ⅱ段', 'II段', '2段', '二段','相间']
            },
            '接地距离III段时间': {
                'must': ['接地', '距离'],
                'first': ['Ⅲ段', 'III段', '3段', '三段'],
                'primary': ['接地', '距离', '时间', '时限'],
                'secondary': ['值', '设置', '参数'],
                'exclude': ['电流', 'Ⅰ段', 'I段', '1段', '一段', 'Ⅱ段', 'II段', '2段', '二段', '反时限','相间']
            },
            '相间距离I段定值': {
                'must': ['相间', '距离'],
                'first': ['Ⅰ段', 'I段', '1段', '一段'],
                'primary': ['相间', '距离', '段'],
                'secondary': ['定值', '值', '设置'],
                'exclude': ['电流', '时间', 'Ⅱ段', 'II段', '2段', '二段', 'Ⅲ段', 'III段', '3段', '三段','接地']
            },
            '相间距离I段时间': {
                'must': ['相间', '距离'],
                'first': ['Ⅰ段', 'I段', '1段', '一段'],
                'primary': ['相间', '距离', '时间', '时限'],
                'secondary': ['值', '设置', '参数'],
                'exclude': ['电流', 'Ⅱ段', 'II段', '2段', '二段', 'Ⅲ段', 'III段', '3段', '三段', '反时限', '接地']
            },
            '相间距离II段定值': {
                'must': ['相间', '距离'],
                'first': ['Ⅱ段', 'II段', '2段', '二段'],
                'primary': ['相间', '距离', '段'],
                'secondary': ['定值', '值', '设置'],
                'exclude': ['电流', '时间', 'Ⅰ段', 'I段', '1段', '一段', 'Ⅲ段', 'III段', '3段', '三段','接地']
            },
            '相间距离II段时间': {
                'must': ['相间', '距离'],
                'first': ['Ⅱ段', 'II段', '2段', '二段'],
                'primary': ['相间', '距离', '时间', '时限'],
                'secondary': ['值', '设置', '参数'],
                'exclude': ['电流', 'Ⅰ段', 'I段', '1段', '一段', 'Ⅲ段', 'III段', '3段', '三段', '反时限','接地']
            },
            '相间距离III段定值': {
                'must': ['相间', '距离'],
                'first': ['Ⅲ段', 'III段', '3段', '三段'],
                'primary': ['相间', '距离', '段'],
                'secondary': ['定值', '值', '设置'],
                'exclude': ['电流', '时间', 'Ⅰ段', 'I段', '1段', '一段', 'Ⅱ段', 'II段', '2段', '二段','接地']
            },
            '相间距离III段时间': {
                'must': ['相间', '距离'],
                'first': ['Ⅲ段', 'III段', '3段', '三段'],
                'primary': ['相间', '距离', '时间', '时限'],
                'secondary': ['值', '设置', '参数'],
                'exclude': ['电流', 'Ⅰ段', 'I段', '1段', '一段', 'Ⅱ段', 'II段', '2段', '二段', '反时限','接地']
            },
            '零序过流I段定值': {
                'must': ['零序','过流'],
                'first': ['Ⅰ段', 'I段', '1段', '一段'],
                'primary': ['零序', '过流', '电流'],
                'secondary': ['定值', '值'],
                'exclude': ['时间', '时限', 'Ⅱ段', 'II段', '2段', '二段', 'Ⅲ段', 'III段', '3段', '三段','起动','电压']
            },
            '零序过流I段时间': {
                'must': ['零序'],
                'first': ['Ⅰ段', 'I段', '1段', '一段'],
                'primary': ['零序', '过流', '时间', '时限'],
                'secondary': ['定值', '值'],
                'exclude': ['电流', 'Ⅰ段', 'I段', '1段', '一段', 'Ⅲ段', 'III段', '3段', '三段', '反时限','电压']
            },
            '零序过流II段定值': {
                'must': ['零序','过流'],
                'first': ['Ⅱ段', 'II段', '2段', '二段'],
                'primary': ['零序', '过流', '电流'],
                'secondary': ['定值', '值'],
                'exclude': ['时间', '时限', 'Ⅰ段', 'I段', '1段', '一段', 'Ⅲ段', 'III段', '3段', '三段','起动','电压']
            },
            '零序过流II段时间': {
                'must': ['零序'],
                'first': ['Ⅱ段', 'II段', '2段', '二段'],
                'primary': ['零序', '过流', '时间', '时限'],
                'secondary': ['定值', '值'],
                'exclude': ['电流', 'Ⅰ段', 'I段', '1段', '一段', 'Ⅲ段', 'III段', '3段', '三段', '反时限','电压']
            },
            '零序过流III段定值': {
                'must': ['零序','过流'],
                'first': ['Ⅲ段', 'III段', '3段', '三段'],
                'primary': ['零序', '过流', '电流'],
                'secondary': ['定值', '值'],
                'exclude': ['时间', '时限', 'Ⅰ段', 'I段', '1段', '一段', 'Ⅱ段', 'II段', '2段', '二段','起动','电压']
            },
            '零序过流III段时间': {
                'must': ['零序'],
                'first': ['Ⅲ段', 'III段', '3段', '三段'],
                'primary': ['零序', '过流', '时间', '时限'],
                'secondary': ['定值', '值'],
                'exclude': ['电流', 'Ⅰ段', 'I段', '1段', '一段', 'Ⅱ段', 'II段', '2段', '二段', '反时限','电压']
            },

        },
    "transform": {
        "额定容量": {
            "must": ["额定", "容量"],
            "first": ["额定","容量"],
            "primary": [],
            "secondary": ["值", "参数", "额定值"],
            "exclude": ["电压", "电流", "CT", "变比", "过流", "零序", "段", "时间"]
        },
        "高压侧额定电压": {
            "must": ["额定电压", "高压侧"],
            "first": ["高压侧", "额定电压"],
            "primary": ["高压侧", "额定", "电压"],
            "secondary": ["值", "参数", "额定值"],
            "exclude": ["容量", "电流", "CT", "变比", "过流", "零序", "段", "时间"]
        },

        "高压侧CT变比": {
            "must": ["CT"],
            "first": ["高压侧", "CT变比","二次"],
            "primary": ["额定"],
            "secondary": ["值", "比值"],
            "exclude": ["容量", "电压", "电流", "过流", "零序", "段", "时间","一次"]
        },

        "复压过流I段电流定值": {
            "must": ["复压过流"],
            "first": ["Ⅰ段", "I段", "1段", "一段"],
            "primary": ["复压过流", "电流"],
            "secondary": ["定值", "值", "设置"],
            "exclude": ["时间", "时限", "Ⅱ段", "II段", "2段", "二段", "Ⅲ段", "III段", "3段", "三段","零序",'Ⅳ段','电压']
        },
        "复压过流I段1时限时间定值": {
            "must": ["复压过流"],
            "first": ["Ⅰ段1", "I段1", "1段1", "一段1"],
            "primary": ["复压过流", "时间", "时限"],
            "secondary": ["定值", "值", "设置"],
            "exclude": ["电流", "2时限", "3时限", "Ⅱ段", "III段", "反时限","零序",'Ⅳ段','电压']
        },
        "复压过流I段2时限时间定值": {
            "must": ["复压过流"],
            "first": ["Ⅰ段2", "I段2", "1段2", "一段2"],
            "primary": ["复压过流", "时间", "时限"],
            "secondary": ["定值", "值", "设置"],
            "exclude": ["电流", "1时限", "3时限", "Ⅱ段", "III段", "反时限","零序",'Ⅳ段','电压']
        },
        "复压过流I段3时限时间定值": {
            "must": ["复压过流"],
            "first": ["Ⅰ段3", "I段3", "1段3", "一段3"],
            "primary": ["复压过流", "时间", "时限"],
            "secondary": ["定值", "值", "设置"],
            "exclude": ["电流", "1时限", "2时限", "Ⅱ段", "III段", "反时限","零序",'Ⅳ段','电压']
        },
        "复压过流II段电流定值": {
            "must": ["复压过流"],
            "first": ["Ⅱ段", "II段", "2段", "二段"],
            "primary": ["复压过流", "电流"],
            "secondary": ["定值", "值", "设置"],
            "exclude": ["时间", "时限", "Ⅰ段", "I段", "1段", "一段", "Ⅲ段", "III段", "3段", "三段","零序",'Ⅳ段','电压']
        },
        "复压过流II段1时限时间定值": {
            "must": ["复压过流"],
            "first": ["II段1时限", "Ⅱ段1时限"],
            "primary": ["复压过流", "时间", "时限"],
            "secondary": ["定值", "值", "设置"],
            "exclude": ["电流", "2时限", "3时限", "Ⅰ段", "III段", "反时限","零序",'Ⅳ段','电压']
        },
        "复压过流II段2时限时间定值": {
            "must": ["复压过流"],
            "first": ["II段2时限", "Ⅱ段2时限"],
            "primary": ["复压过流", "时间", "时限"],
            "secondary": ["定值", "值", "设置"],
            "exclude": ["电流", "1时限", "3时限", "Ⅰ段", "III段", "反时限","零序",'Ⅳ段','电压']
        },
        "复压过流II段3时限时间定值": {
            "must": ["复压过流"],
            "first": ["II段3时限", "Ⅱ段3时限"],
            "primary": ["复压过流", "时间", "时限"],
            "secondary": ["定值", "值", "设置"],
            "exclude": ["电流", "1时限", "2时限", "Ⅰ段", "III段", "反时限","零序",'Ⅳ段','电压']
        },
        "复压过流III段电流定值": {
            "must": ["复压过流"],
            "first": ["Ⅲ段", "III段", "3段", "三段"],
            "primary": ["复压过流", "电流"],
            "secondary": ["定值", "值", "设置"],
            "exclude": ["时间", "时限", "Ⅰ段", "I段", "1段", "一段", "Ⅱ段", "II段", "2段", "二段","零序",'Ⅳ段','电压']
        },
        "复压过流III段1时限时间定值": {
            "must": ["复压过流"],
            "first": ["III段1时限", "Ⅲ段1时限"],
            "primary": ["复压过流", "时间", "时限"],
            "secondary": ["定值", "值", "设置"],
            "exclude": ["电流", "2时限", "3时限", "Ⅰ段", "II段", "反时限","零序",'Ⅳ段','电压']
        },
        "复压过流III段2时限时间定值": {
            "must": ["复压过流"],
            "first": ["III段2时限", "Ⅲ段2时限"],
            "primary": ["复压过流", "时间", "时限"],
            "secondary": ["定值", "值", "设置"],
            "exclude": ["电流", "1时限", "3时限", "Ⅰ段", "II段", "反时限","零序",'Ⅳ段','电压']
        },
        "复压过流III段3时限时间定值": {
            "must": ["复压过流"],
            "first": ["III段3时限", "Ⅲ段3时限"],
            "primary": ["复压过流", "时间", "时限"],
            "secondary": ["定值", "值", "设置"],
            "exclude": ["电流", "1时限", "2时限", "Ⅰ段", "II段", "反时限","零序",'Ⅳ段','电压']
        },
        "零序过流I段电流定值": {
            "must": ["零序","过流"],
            "first": ["Ⅰ段", "I段", "1段", "一段"],
            "primary": ["零序","过流", "电流"],
            "secondary": ["定值", "值", "设置"],
            "exclude": ["时间", "时限", "Ⅱ段", "II段", "2段", "二段", "Ⅲ段", "III段", "3段", "三段","高压侧","中压侧","低压侧","复压",'Ⅳ段','电压']
        },
        "零序过流I段1时限时间定值": {
            "must": ["零序","过流"],
            "first": ["I段1时限", "Ⅰ段1时限","1段1时限"],
            "primary": ["零序","过流", "时间", "时限"],
            "secondary": ["定值", "值", "设置"],
            "exclude": ["电流", "2时限", "3时限", "Ⅱ段", "III段", "反时限","高压侧","中压侧","低压侧","复压",'Ⅳ段','电压']
        },
        "零序过流I段2时限时间定值": {
            "must": ["零序","过流"],
            "first": ["I段2时限", "Ⅰ段2时限"],
            "primary": ["零序","过流", "时间", "时限"],
            "secondary": ["定值", "值", "设置"],
            "exclude": ["电流", "1时限", "3时限", "Ⅱ段", "III段", "反时限","高压侧","中压侧","低压侧","复压",'Ⅳ段','电压']
        },
        "零序过流I段3时限时间定值": {
            "must": ["零序","过流"],
            "first": ["I段3时限", "Ⅰ段3时限"],
            "primary": ["零序","过流", "时间", "时限"],
            "secondary": ["定值", "值", "设置"],
            "exclude": ["电流", "1时限", "2时限", "Ⅱ段", "III段", "反时限","高压侧","中压侧","低压侧","复压",'Ⅳ段','电压']
        },
        "零序过流II段电流定值": {
            "must": ["零序","过流"],
            "first": ["Ⅱ段", "II段", "2段", "二段"],
            "primary": ["零序","过流", "电流"],
            "secondary": ["定值", "值", "设置"],
            "exclude": ["时间", "时限", "Ⅰ段", "I段", "1段", "一段", "Ⅲ段", "III段", "3段", "三段","高压侧","中压侧","低压侧","复压",'Ⅳ段','电压']
        },
        "零序过流II段1时限时间定值": {
            "must": ["零序","过流"],
            "first": ["II段1时限", "Ⅱ段1时限"],
            "primary": ["零序","过流", "时间", "时限"],
            "secondary": ["定值", "值", "设置"],
            "exclude": ["电流", "2时限", "3时限", "Ⅰ段", "III段", "反时限","高压侧","中压侧","低压侧","复压",'Ⅳ段','电压']
        },
        "零序过流II段2时限时间定值": {
            "must": ["零序","过流"],
            "first": ["II段2时限", "Ⅱ段2时限"],
            "primary": ["零序","过流", "时间", "时限"],
            "secondary": ["定值", "值", "设置"],
            "exclude": ["电流", "1时限", "3时限", "Ⅰ段", "III段", "反时限","高压侧","中压侧","低压侧","复压",'Ⅳ段','电压']
        },
        "零序过流II段3时限时间定值": {
            "must": ["零序","过流"],
            "first": ["II段3时限", "Ⅱ段3时限"],
            "primary": ["零序","过流", "时间", "时限"],
            "secondary": ["定值", "值", "设置"],
            "exclude": ["电流", "1时限", "2时限", "Ⅰ段", "III段", "反时限","高压侧","中压侧","低压侧","复压",'Ⅳ段','电压']
        },
        "零序过流III段电流定值": {
            "must": ["零序","过流"],
            "first": ["Ⅲ段", "III段", "3段", "三段"],
            "primary": ["零序","过流", "电流"],
            "secondary": ["定值", "值", "设置"],
            "exclude": ["时间", "时限", "Ⅰ段", "I段", "1段", "一段", "Ⅱ段", "II段", "2段", "二段","高压侧","中压侧","低压侧","复压",'Ⅳ段','电压']
        },
        "零序过流III段1时限时间定值": {
            "must": ["零序","过流"],
            "first": ["III段1时限", "Ⅲ段1时限"],
            "primary": ["零序","过流", "时间", "时限"],
            "secondary": ["定值", "值", "设置"],
            "exclude": ["电流", "2时限", "3时限", "Ⅰ段", "II段", "反时限","高压侧","中压侧","低压侧","复压",'Ⅳ段','电压']
        },
        "零序过流III段2时限时间定值": {
            "must": ["零序","过流"],
            "first": ["III段2时限", "Ⅲ段2时限"],
            "primary": ["零序","过流", "时间", "时限"],
            "secondary": ["定值", "值", "设置"],
            "exclude": ["电流", "1时限", "3时限", "Ⅰ段", "II段", "反时限","高压侧","中压侧","低压侧","复压",'Ⅳ段','电压']
        },
        "零序过流III段3时限时间定值": {
            "must": ["零序","过流"],
            "first": ["III段3时限", "Ⅲ段3时限"],
            "primary": ["零序","过流", "时间", "时限"],
            "secondary": ["定值", "值", "设置"],
            "exclude": ["电流", "1时限", "2时限", "Ⅰ段", "II段", "反时限","高压侧","中压侧","低压侧","复压",'Ⅳ段','电压']
        },
        "差动动作电流定值": {
            "must": ["差动"],
            "first": ["动作电流", "电流","差流"],
            "primary": ["差动", "动作", "电流","起始"],
            "secondary": ["定值", "值", "设置"],
            "exclude": ["时间", "时限", "补偿", "系数", "距离", "过流", "段"]
        },

    },"busbar":{
                "母线差动启动电流定值": {
                    "must": ["差动"],
                    "first": ["启动电流", "母线", "差动"],
                    "primary": [],
                    "secondary": ["定值", "值", "设置"],
                    "exclude": ["时间", "时限", "段","失灵"]
                },
                "母联失灵电流定值": {
                    "must": ["失灵"],
                    "first": ["失灵", "母联"],
                    "primary": ["电流"],
                    "secondary": ["定值", "值", "设置"],
                    "exclude": ["时间", "时限", "过流", "段","差动","三相","零序"]
                },
                "三相失灵电流定值": {
                    "must": ["失灵"],
                    "first": ["失灵", "三相"],
                    "primary": ["电流"],
                    "secondary": ["定值", "值", "设置"],
                    "exclude": ["时间", "时限",  "过流", "段", "差动","母线","零序"]
                },
                "失灵零序电流定值": {
                    "must": ["失灵"],
                    "first": ["失灵", "零序"],
                    "primary": ["电流"],
                    "secondary": ["定值", "值", "设置"],
                    "exclude": ["时间", "时限", "过流", "段", "差动", "母线","三相"]
                },
        "母联失灵时间定值": {
            "must": ["母联"],
            "first": ["时间", "时限","母联"],
            "primary": ["失灵",],
            "secondary": ["定值", "值", "设置"],
            "exclude": ["电流", "1时限", "2时限", "Ⅰ段", "II段", "反时限"]
        },
        "失灵时间定值": {
            "must": ["失灵"],
            "first": [ "失灵"],
            "primary": ["时间","时限"],
            "secondary": ["定值", "值", "设置"],
            "exclude": ["电流", "Ⅰ段", "II段", "反时限","母联"]
        },
            }

        }



    def map_to_standard(self, extracted_param: str, context: str = "", mapped_std_params: set = None,device_type: str = "line_10") -> Optional[str]:
        if not extracted_param or extracted_param == "未识别到":
            return None

        # 清理参数名
        cleaned_param = self._clean_parameter_name(extracted_param)
        combined_text = cleaned_param
        if context:
            combined_text = self._clean_parameter_name(context) + cleaned_param

        print(f"关键词映射: '{extracted_param}' -> 清理后: '{cleaned_param}'")
        # 检查设备类型是否存在
        if device_type not in self.parameter_patterns:
            print(f"  警告: 未找到设备类型 '{device_type}' 的参数模式")
            return None
        device_patterns = self.parameter_patterns[device_type]
        # 计算每个标准参数的匹配分数
        scores = {}
        for std_name, patterns in device_patterns.items():
            # 如果该标准参数已经被映射过，跳过
            if mapped_std_params and std_name in mapped_std_params:
                continue

            score = self._calculate_match_score(combined_text, patterns)
            if score > 0:
                scores[std_name] = score
                print(f"  候选: {std_name} = {score}")

        # 选择得分最高的参数
        if scores:
            best_match = max(scores.items(), key=lambda x: x[1])
            best_std_name, best_score = best_match

            # 设置匹配阈值
            if best_score >= 4.0:  # 至少匹配2个主要关键词或等效分数
                print(f"  最佳匹配: {best_std_name} (得分: {best_score})")
                return best_std_name
            else:
                print(f"  匹配分数不足: {best_std_name} (得分: {best_score} < 4.0)")

        print(f"  无匹配参数")
        return None

    def _calculate_match_score(self, text: str, patterns: Dict) -> float:
        """计算文本与模式匹配的分数"""
        score = 0.0
        match = 0
        for keyword in patterns['must']:
            print(text, keyword,self._fuzzy_contains(text, keyword))
            if self._fuzzy_contains(text, keyword):
                match = 1
        if match == 0:
            return score
        # 关键第一词匹配（权重高）
        for keyword in patterns['first']:
            print(keyword,self._fuzzy_contains(text, keyword))
            if self._fuzzy_contains(text, keyword):
                score += 4.0  # 主要关键词得4分
                print(f'first{score}')
        # 主要关键词匹配（权重高）
        for keyword in patterns['primary']:
            print(keyword, self._fuzzy_contains(text, keyword))
            if self._fuzzy_contains(text, keyword):
                score += 2.0  # 主要关键词得2分
                print(f'primary{score}')

        # 次要关键词匹配（权重低）
        for keyword in patterns['secondary']:
            print(keyword, self._fuzzy_contains(text, keyword))
            if self._fuzzy_contains(text, keyword):
                score += 0.5  # 次要关键词得0.5分
                print(f'secondary{score}')
        # 排除关键词惩罚（如果包含排除关键词，大幅扣分）
        for keyword in patterns['exclude']:
            print(keyword, self._fuzzy_contains(text, keyword))
            if self._fuzzy_contains(text, keyword):
                score -= 5.0
                print(f'exclude{score}')

        # 确保分数不为负
        return max(score, 0.0)

    def _fuzzy_contains(self, text: str, keyword: str) -> bool:
        """模糊包含判断（修复段数互斥问题）"""
        # 1. 定义所有段数关键词集合（互斥组）
        segment_keywords = {'Ⅰ段', 'I段', '1段', '一段',
                            'Ⅱ段', 'II段', '2段', '二段',
                            'Ⅲ段', 'III段', '3段', '三段'}

        # 2. 段数关键词专属校验（核心修复）
        if keyword in segment_keywords:
            # 提取文本中实际存在的所有段数关键词
            existing_segments = [seg for seg in segment_keywords if seg in text]
            # 仅当文本中只存在目标段数（或无其他段数）时，才判定为匹配
            return keyword in existing_segments and len(existing_segments) == 1

        # 3. 原有正常匹配逻辑（非段数关键词沿用）
        if keyword in text:
            return True
        similarity = difflib.SequenceMatcher(None, text, keyword).ratio()
        if similarity > 0.85:
            return True
        common_errors = {
            'I段': ['Ⅰ段', 'l段', '丨段', '1段', '一段'],
            'II段': ['Ⅱ段', 'll段', '2段', '二段'],
            'III段': ['Ⅲ段', 'lll段', '3段', '三段'],
            '段': ['断'],
            '流': ['琉'],
            '电': ['龟'],
            '保': ['堡'],
            '护': ['户']
        }
        for correct, errors in common_errors.items():
            if keyword == correct and any(error in text for error in errors):
                return True
        return False

    def _strict_contains(self, text: str, keyword: str) -> bool:
        """严格包含判断（用于段数排他性校验）"""
        common_errors = {
            'I段': ['Ⅰ段', 'l段', '丨段', '1段', '一段'],
            'II段': ['Ⅱ段', 'll段', '2段', '二段'],
            'III段': ['Ⅲ段', 'lll段', '3段', '三段'],
            '段': ['断'],
            '流': ['琉'],
            '电': ['龟'],
            '保': ['堡'],
            '护': ['户']
        }
        return keyword in text or any(kw in text for kw in common_errors.get(keyword, []))

    def _clean_parameter_name(self, param: str) -> str:
        """清理参数名"""
        # 去除标点符号和空格
        cleaned = re.sub(r'[^\w\u4e00-\u9fffⅠⅡⅢ]', '', param)
        return cleaned


class SmartTableExtractor:
    """智能表格提取器 - 使用关键词匹配，支持防重复映射"""

    def __init__(self, parameter_mapper: KeywordBasedParameterMapper):
        self.mapper = parameter_mapper

    def _fix_ocr_char_confusion(self, text: str) -> str:
        """修复OCR常见的字符混淆问题"""
        if not text:
            return text
        # 常见混淆映射：键为错误字符，值为正确字符
        confusion_map = {
            'O': '0',  # 字母O → 数字0
            'o': '0',  # 小写o → 数字0
            'l': '1',  # 字母l → 数字1
        }
        # 替换混淆字符
        fixed_text = text
        for wrong_char, right_char in confusion_map.items():
            fixed_text = fixed_text.replace(wrong_char, right_char)
        return fixed_text
    def extract_from_tables(self, table_data: list, format_type: str,device_type:str) -> dict:
        """从表格数据中提取参数和数值 - 支持防重复映射"""
        results = {}
        # 维护已映射的标准参数集合
        mapped_std_params = set()

        for table_index, table in enumerate(table_data):
            if not isinstance(table, dict):
                continue

            print(f"处理表格 {table_index + 1}")

            # 构建完整的表格数据结构
            table_structure = self._build_complete_table_structure(table)
            if not table_structure:
                continue

            # 根据格式提取数据
            if format_type == 'format2':
                table_results = self._extract_format2_with_keywords(table_structure, mapped_std_params,device_type)
            else:
                table_results = self._extract_format1_with_keywords(table_structure, mapped_std_params,device_type)

            print("当前表格提取结果:", table_results)

            # 合并结果，并更新已映射参数集合
            for key, value in table_results.items():
                if key not in results or results[key] == "未识别到":
                    results[key] = value


        print("最终提取结果:", results)
        return results

    def _extract_format2_with_keywords(self, table_array: list, mapped_std_params: set,device_type:str) -> dict:
        """使用关键词匹配提取格式2数据 - 支持防重复映射"""
        results = {}

        if len(table_array) < 2:
            return results

        headers = table_array[0]
        data_rows = table_array[1:]

        print(f"表头: {headers}")

        # 智能查找列索引（支持多种表头表达）
        protection_idx = self._find_column_by_keywords(headers, ['保护功能', '功能类型', '保护类型'])
        name_idx = self._find_column_by_keywords(headers, ['名称', '定值名称', '参数名称'])
        value_idx = self._find_column_by_keywords(headers, ['定值', '整定值', '数值', '值'])

        print(f"保护功能列: {protection_idx}, 名称列: {name_idx}, 定值列: {value_idx}")

        if name_idx is None or value_idx is None:
            print("未找到必要的列")
            return results

        current_protection = ""

        for row_index, row in enumerate(data_rows):
            # 跳过空行
            if not any(row):
                continue

            # 更新保护功能
            if protection_idx is not None and row[protection_idx]:
                current_protection = row[protection_idx]

            # 提取名称和数值
            name = row[name_idx] if name_idx < len(row) else ""
            param_value = row[value_idx] if value_idx < len(row) else ""

            if name and param_value and name != param_value and self._is_valid_value(param_value):
                # 组合保护功能和名称
                if current_protection:
                    combined_name = current_protection + name
                else:
                    combined_name = name

                # 使用关键词匹配映射到标准参数名，传入已映射参数集合
                standardized_name = self.mapper.map_to_standard(combined_name, current_protection, mapped_std_params,device_type)

                if standardized_name and standardized_name not in mapped_std_params:
                    results[standardized_name] = param_value
                    mapped_std_params.add(standardized_name)
                    print(f"  提取结果: {standardized_name} = {param_value}")
                    # 注意：这里不在循环内更新mapped_std_params，而是在外层统一更新

        return results

    def _extract_format1_with_keywords(self, table_array: list, mapped_std_params: set,device_type:str) -> dict:
        """使用关键词匹配提取格式1数据 - 支持防重复映射"""
        results = {}

        if len(table_array) < 2:
            return results

        headers = table_array[0]
        data_rows = table_array[1:]

        # 智能查找列索引
        name_idx = self._find_column_by_keywords(headers, ['定值名称', '名称'])
        value_idx = self._find_column_by_keywords(headers, ['定值', '整定值', '数值','新定值'])

        if name_idx is None or value_idx is None:
            return results

        current_function = ""
        for row in data_rows:
            # 跳过空行
            if not any(row):
                continue

            # 提取参数名和参数值
            param_name = row[name_idx] if name_idx < len(row) else ""
            param_value = row[value_idx] if value_idx < len(row) else ""

            if param_name and param_value and param_name != param_value and self._is_valid_value(param_value):
                # 使用关键词匹配映射到标准参数名，传入已映射参数集合
                standardized_name = self.mapper.map_to_standard(param_name, current_function, mapped_std_params,device_type)

                if standardized_name and standardized_name not in mapped_std_params:
                    results[standardized_name] = param_value
                    mapped_std_params.add(standardized_name)  # 记录已映射的参数

        return results

    def _find_column_by_keywords(self, headers: list, keywords: list) -> int:
        """使用关键词匹配查找列索引"""
        best_match_index = -1
        best_match_score = 0

        for i, header in enumerate(headers):
            header_clean = re.sub(r'[^\w\u4e00-\u9fff]', '', header)
            score = self._calculate_header_match_score(header_clean, keywords)

            if score > best_match_score:
                best_match_score = score
                best_match_index = i

        return best_match_index if best_match_score > 0 else None

    def _calculate_header_match_score(self, header: str, keywords: list) -> float:
        """计算表头与关键词的匹配分数"""
        best_score = 0

        for keyword in keywords:
            # 完全匹配
            if keyword == header:
                return 2.0

            # 包含匹配
            if keyword in header:
                score = 1.0 + (len(keyword) / len(header)) * 0.5
                best_score = max(best_score, score)

            # 相似度匹配
            similarity = difflib.SequenceMatcher(None, header, keyword).ratio()
            if similarity > 0.7:
                best_score = max(best_score, similarity)

        return best_score

    def _build_complete_table_structure(self, table: dict) -> list:
        """构建完整的表格结构"""
        try:
            # 获取所有单元格
            all_cells = []

            # 处理表头
            header_cells = table.get('header', [])
            if not isinstance(header_cells, list):
                header_cells = [header_cells] if header_cells else []

            # 处理表体
            body_cells = table.get('body', [])
            if not isinstance(body_cells, list):
                body_cells = [body_cells] if body_cells else []

            all_cells = header_cells + body_cells

            # 按行列组织数据
            cell_dict = {}
            for cell in all_cells:
                if not isinstance(cell, dict):
                    continue

                row = cell.get('row_start', 0)
                col = cell.get('col_start', 0)
                text = cell.get('words', '').strip()

                if row not in cell_dict:
                    cell_dict[row] = {}
                cell_dict[row][col] = text

            # 转换为有序列表
            if not cell_dict:
                return []

            # 确定表格维度
            max_row = max(cell_dict.keys())
            max_col = max(max(row.keys()) for row in cell_dict.values())

            # 构建二维数组
            table_array = []
            for row in range(max_row + 1):
                row_data = []
                for col in range(max_col + 1):
                    row_data.append(cell_dict.get(row, {}).get(col, ''))
                table_array.append(row_data)
            print("表格结构:", table_array)
            return table_array

        except Exception as e:
            print(f"构建表格结构失败: {e}")
            return []

    def _is_valid_value(self, value: str) -> bool:
        """检查是否为有效的参数值"""
        if not value or value == "未识别到":
            return False
        fixed_value = self._fix_ocr_char_confusion(value)
        # 检查是否包含数字或特殊字符
        if re.search(r'[\d/]', fixed_value):
            return True

        return False

class BasicInfoExtractor:
    """基础信息提取器（CT变比、PT变比等）"""

    def __init__(self):
        self.basic_info_patterns = {
            'CT变比': r'CT变比[:：\s]*([\d/]+[A]?)',
            'PT变比': r'PT变比[:：\s]*([\d/\.]+[V]?)',
            '零序CT变比': r'零序CT变比[:：\s]*([\d/]+[A]?)',
        }

    def extract_basic_info(self, ocr_text: str) -> dict:
        """从OCR文本中提取基础信息"""
        results = {}

        print("提取基础信息...")
        for field, pattern in self.basic_info_patterns.items():
            matches = re.findall(pattern, ocr_text, re.IGNORECASE)
            if matches:
                # 取最后一个匹配（通常是最具体的）
                results[field] = matches[-1].strip()
                print(f"  找到 {field}: {results[field]}")
        print(results)
        return results


class EnhancedBaiduOCR:
    """增强版百度OCR识别 - 修复版本"""

    def __init__(self, enable_preprocess=True):
        self.token = self._fetch_token()
        if not self.token:
            raise ValueError("获取百度OCR令牌失败，请检查API_KEY和SECRET_KEY")

        self.enable_preprocess = enable_preprocess
        self.preprocessor = ImagePreprocessor() if enable_preprocess else None
        self.format_detector = FormatDetector()
        self.parameter_mapper = KeywordBasedParameterMapper()
        self.table_extractor = SmartTableExtractor(self.parameter_mapper)
        self.basic_info_extractor = BasicInfoExtractor()

    def _fetch_token(self):
        """获取百度OCR访问令牌"""
        params = {
            'grant_type': 'client_credentials',
            'client_id': API_KEY,
            'client_secret': SECRET_KEY
        }
        post_data = urlencode(params)
        if IS_PY3:
            post_data = post_data.encode('utf-8')

        try:
            req = Request(TOKEN_URL, post_data)
            with urlopen(req, timeout=10) as f:
                result_str = f.read()
            if IS_PY3:
                result_str = result_str.decode()

            result = json.loads(result_str)
            if 'access_token' in result:
                return result['access_token']
            else:
                print(f"获取令牌失败: {result}")
                return None
        except URLError as e:
            print(f"网络错误: {e}")
            return None

    def read_image_file(self, image_path):
        """读取本地图片文件并转换为base64编码"""
        try:
            with open(image_path, 'rb') as f:
                image_data = f.read()
            return base64.b64encode(image_data).decode('utf-8')
        except Exception as e:
            print(f"读取图片失败: {e}")
            return None

    def general_ocr(self, image_path, use_preprocess=None):
        """通用文字识别"""
        if use_preprocess is None:
            use_preprocess = self.enable_preprocess

        if use_preprocess and self.preprocessor:
            image_base64 = self.preprocessor.preprocess_for_ocr(image_path)
        else:
            image_base64 = self.read_image_file(image_path)

        if not image_base64:
            return None

        url = f"{OCR_ACCURATE_URL}?access_token={self.token}"
        data = urlencode({
            'image': image_base64,
            'language_type': 'CHN_ENG'
        })

        try:
            req = Request(url, data.encode('utf-8'))
            req.add_header('Content-Type', 'application/x-www-form-urlencoded')
            with urlopen(req, timeout=30) as f:
                result_str = f.read()
            if IS_PY3:
                result_str = result_str.decode()
            return json.loads(result_str)
        except URLError as e:
            print(f"OCR识别请求失败: {e}")
            return None

    def table_ocr(self, image_path, use_preprocess=None):
        """表格识别"""
        if use_preprocess is None:
            use_preprocess = self.enable_preprocess

        if use_preprocess and self.preprocessor:
            image_base64 = self.preprocessor.preprocess_for_table(image_path)
        else:
            image_base64 = self.read_image_file(image_path)

        if not image_base64:
            return None

        url = f"{OCR_FORM_URL}?access_token={self.token}"
        data = urlencode({
            'image': image_base64,
            'request_type': 'excel'
        })

        try:
            req = Request(url, data.encode('utf-8'))
            req.add_header('Content-Type', 'application/x-www-form-urlencoded')
            with urlopen(req, timeout=30) as f:
                result_str = f.read()
            if IS_PY3:
                result_str = result_str.decode()
            return json.loads(result_str)
        except URLError as e:
            print(f"表格识别请求失败: {e}")
            return None

    def extract_device_model(self, image_path, pattern="装置型号"):
        """提取装置型号"""
        ocr_result = self.general_ocr(image_path)
        if not ocr_result or 'words_result' not in ocr_result:
            return None
        device_model = None
        found_device_label = False
        model_pattern = re.compile(r'[A-Za-z0-9\-]+')
        label_pattern = re.compile(rf'{pattern}[:：\s]*')

        for words in ocr_result['words_result']:
            text = words['words'].strip()
            if not found_device_label:
                if label_pattern.search(text):
                    found_device_label = True
                    remaining_text = label_pattern.sub('', text).strip()
                    match = model_pattern.search(remaining_text)
                    if match:
                        device_model = match.group().strip()
                        return device_model
            else:
                match = model_pattern.search(text)
                if match:
                    device_model = match.group().strip()
                    break

        return device_model

    def extract_protection_settings(self, image_path, target_params=None,device_type='line_10'):
        all_results = {}

        # 获取表格识别结果
        table_result = self.table_ocr(image_path)
        # 获取通用OCR结果用于基础信息提取
        ocr_text_result = self.general_ocr(image_path)
        ocr_text = self._get_full_text(ocr_text_result) if ocr_text_result else ""
        print(ocr_text)
        # 1. 从表格中提取保护定值
        if table_result and 'tables_result' in table_result:
            # 检测格式
            format_type = self.format_detector.detect_format(ocr_text, table_result['tables_result'])
            print(f"检测到定值单格式: {format_type}")

            # 提取表格数据
            table_settings = self.table_extractor.extract_from_tables(
                table_result['tables_result'],
                format_type,device_type
            )
            all_results.update(table_settings)

        # # 2. 从通用文本中提取基础信息
        if ocr_text:
            basic_info = self.basic_info_extractor.extract_basic_info(ocr_text)
            all_results.update(basic_info)

        # 3. 数值清洗
        cleaned_results = {}
        for param, value in all_results.items():
            cleaned_value = self.clean_value_text(value)
            cleaned_results[param] = cleaned_value if cleaned_value else value

        # 4. 如果指定了目标参数，进行过滤和映射
        if target_params:
            filtered_results = {}
            for target_param in target_params:
                filtered_results[target_param] = cleaned_results.get(target_param, "未识别到")
            return filtered_results

        return cleaned_results

    def extract_specified_fields_from_table(self, image_path, target_fields):
        """
        兼容旧接口：提取指定字段
        """
        return self.extract_protection_settings(image_path, target_fields)

    def _get_full_text(self, ocr_result):
        """从OCR结果中提取完整文本"""
        if not ocr_result or 'words_result' not in ocr_result:
            return ""

        texts = []
        for words in ocr_result['words_result']:
            if 'words' in words:
                texts.append(words['words'])

        return '\n'.join(texts)

    def _find_matching_parameter_value(self, target_param, extracted_params):
        """在提取的参数中查找匹配的目标参数值"""
        # 先尝试精确匹配
        if target_param in extracted_params:
            return extracted_params[target_param]

        # 尝试模糊匹配
        for extracted_param, value in extracted_params.items():
            if self._fuzzy_match_parameters(target_param, extracted_param):
                return value

        return "未识别到"

    def _fuzzy_match_parameters(self, param1, param2, threshold=0.7):
        """参数模糊匹配"""
        # 简单包含匹配
        if param1 in param2 or param2 in param1:
            return True

        # 使用difflib进行相似度匹配
        similarity = difflib.SequenceMatcher(None, param1, param2).ratio()
        return similarity >= threshold

    def clean_value_text(self, value_text):
        """
        数值清洗函数 - 修复版本
        """
        if not value_text or value_text == "未识别到":
            return ""
        # 步骤1：修复OCR常见字符混淆（先于其他处理，避免误判）
        confusion_map = {'O': '0', 'o': '0', 'l': '1', 'I': '1'}
        fixed_text = value_text
        for wrong, right in confusion_map.items():
            fixed_text = fixed_text.replace(wrong, right)
        # 步骤1：去除所有字母（保留斜杠和数字相关符号）
        value_without_letters = re.sub(r'[A-Za-z]', '', fixed_text)

        # 步骤2：去除干扰符号（保留小数点和斜杠）
        cleaned = re.sub(r'[-—_ ,，\s]', '', value_without_letters)

        # 步骤3：处理斜杠（保留完整变比格式）
        if ('/' in cleaned or '\\' in cleaned):
            # 分割并取最后一部分（兼容/和\）
            parts = re.split(r'[/\\]', cleaned)
            cleaned = parts[-1] if parts else cleaned

        # 步骤4：仅保留数字、小数点和斜杠
        final_cleaned = re.sub(r'[^\d.]', '', cleaned)

        # 处理多个小数点（仅保留第一个）
        if final_cleaned.count('.') > 1:
            first_dot = final_cleaned.index('.')
            final_cleaned = final_cleaned[:first_dot + 1] + final_cleaned[first_dot + 1:].replace('.', '')

        return final_cleaned.strip()


# # 全局OCR实例（可复用）
_global_ocr_instance = None


def get_baidu_ocr_instance():
    """获取全局OCR实例"""
    global _global_ocr_instance
    if _global_ocr_instance is None:
        _global_ocr_instance = EnhancedBaiduOCR()
    return _global_ocr_instance


# 使用示例
# if __name__ == "__main__":
#     # 初始化OCR
#     ocr = get_baidu_ocr_instance()
#
#     # 测试提取
#     image_path = "1.png"  # 替换为您的图片路径
#
#
#
#
#     # 提取指定参数
#     print("\n指定参数:")
#     target_params = ["过流I段电流定值", "过流I段时限定值", "过流II段电流定值", "过流II段时限定值",
#                      "过流III段电流定值", "过流III段时限定值", "零序过流I段电流定值", "零序过流I段时限定值", "零序过流II段电流定值", "零序过流II段时限定值",
#                      "零序过流III段电流定值", "零序过流III段时限定值"]
#     specified_settings = ocr.extract_protection_settings(image_path, target_params)
#     for param, value in specified_settings.items():
#         print(f"  {param}: {value}")
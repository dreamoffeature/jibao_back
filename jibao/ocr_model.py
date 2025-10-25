# # coding=utf-8
# import sys
# import json
# import base64
# import re
# from urllib.request import urlopen, Request
# from urllib.error import URLError
# from urllib.parse import urlencode
#
# # 兼容Python2和Python3
# IS_PY3 = sys.version_info.major == 3
#
# # 忽略HTTPS证书验证
# import ssl
#
# ssl._create_default_https_context = ssl._create_unverified_context
#
# # 百度OCR配置（请替换为你的实际密钥）
# API_KEY = 'IlquMzBntv5vGDv6AIesxzWQ'
# SECRET_KEY = '5F6xm3QO35FgP4mkoZQTm7R5njHCuhKp'
#
# # 百度OCR接口地址
# TOKEN_URL = 'https://aip.baidubce.com/oauth/2.0/token'
# OCR_ACCURATE_URL = "https://aip.baidubce.com/rest/2.0/ocr/v1/accurate_basic"  # 高精度通用识别
# OCR_FORM_URL = "https://aip.baidubce.com/rest/2.0/ocr/v1/table"  # 表格识别（可选）
#
#
# class BaiduOCR:
#     def __init__(self):
#         self.token = self.fetch_token()
#         if not self.token:
#             raise ValueError("获取百度OCR令牌失败，请检查API_KEY和SECRET_KEY")
#
#     def fetch_token(self):
#         """获取百度OCR访问令牌"""
#         params = {
#             'grant_type': 'client_credentials',
#             'client_id': API_KEY,
#             'client_secret': SECRET_KEY
#         }
#         post_data = urlencode(params)
#         if IS_PY3:
#             post_data = post_data.encode('utf-8')
#
#         try:
#             req = Request(TOKEN_URL, post_data)
#             with urlopen(req, timeout=10) as f:
#                 result_str = f.read()
#             if IS_PY3:
#                 result_str = result_str.decode()
#
#             result = json.loads(result_str)
#             if 'access_token' in result:
#                 return result['access_token']
#             else:
#                 print(f"获取令牌失败: {result}")
#                 return None
#         except URLError as e:
#             print(f"网络错误: {e}")
#             return None
#
#     def read_image_file(self, image_path):
#         """读取本地图片文件并转换为base64编码"""
#         try:
#             with open(image_path, 'rb') as f:
#                 image_data = f.read()
#             return base64.b64encode(image_data).decode('utf-8')
#         except Exception as e:
#             print(f"读取图片失败: {e}")
#             return None
#
#     def general_ocr(self, image_path):
#         """通用文字识别（高精度版）"""
#         image_base64 = self.read_image_file(image_path)
#         if not image_base64:
#             return None
#
#         url = f"{OCR_ACCURATE_URL}?access_token={self.token}"
#         data = urlencode({
#             'image': image_base64,
#             'language_type': 'CHN_ENG'  # 中英文混合识别
#         })
#
#         try:
#             req = Request(url, data.encode('utf-8'))
#             req.add_header('Content-Type', 'application/x-www-form-urlencoded')
#             with urlopen(req, timeout=30) as f:
#                 result_str = f.read()
#             if IS_PY3:
#                 result_str = result_str.decode()
#             return json.loads(result_str)
#         except URLError as e:
#             print(f"OCR识别请求失败: {e}")
#             return None
#
#     def extract_device_model_basic_info(self, image_path,pattern):
#         ocr_result = self.general_ocr(image_path)
#         if not ocr_result or 'words_result' not in ocr_result:
#             return None
#
#         device_model = None
#         found_device_label = False
#         # 增强正则：支持匹配型号前后的冒号、空格等符号
#         model_pattern = re.compile(r'[A-Za-z0-9\-]+')
#         # 匹配“装置型号”及后续可能的符号（冒号、空格等）
#         label_pattern = re.compile(rf'{pattern}[:：\s]*')
#
#         for words in ocr_result['words_result']:
#             text = words['words'].strip()
#             if not found_device_label:
#                 # 检查是否包含“装置型号”及后续符号
#                 if label_pattern.search(text):
#                     found_device_label = True
#                     # 移除“装置型号”及符号，提取剩余文本
#                     remaining_text = label_pattern.sub('', text).strip()
#                     # 从剩余文本中匹配型号
#                     match = model_pattern.search(remaining_text)
#                     if match:
#                         device_model = match.group().strip()
#                         return device_model
#             else:
#                 # 匹配下一行的型号
#                 match = model_pattern.search(text)
#                 if match:
#                     device_model = match.group().strip()
#                     break
#
#         return device_model
#
#     def table_ocr(self, image_path):
#         """表格识别（返回结构化表格数据）"""
#         image_base64 = self.read_image_file(image_path)
#         if not image_base64:
#             return None
#
#         url = f"{OCR_FORM_URL}?access_token={self.token}"
#         data = urlencode({
#             'image': image_base64,
#             'request_type': 'excel'
#         })
#
#         try:
#             req = Request(url, data.encode('utf-8'))
#             req.add_header('Content-Type', 'application/x-www-form-urlencoded')
#             with urlopen(req, timeout=30) as f:
#                 result_str = f.read()
#             if IS_PY3:
#                 result_str = result_str.decode()
#             return json.loads(result_str)
#         except URLError as e:
#             print(f"表格识别请求失败: {e}")
#             return None
#
#
#
#     def extract_specified_fields_from_table(self, image_path, target_fields):
#         """
#         提取表格中的指定字段，优先处理表头，保留数值清洗逻辑
#         """
#         table_result = self.table_ocr(image_path)
#         if not table_result or 'tables_result' not in table_result:
#             return {}
#
#         tables = table_result.get('tables_result', [])
#         if not isinstance(tables, list) or len(tables) == 0:
#             return {}
#
#         extracted = {}
#
#         for table in tables:
#             if not isinstance(table, dict):
#                 continue
#             body = table.get('body', [])
#             body_cells = body if isinstance(body, list) else [body]
#
#             for target_field in target_fields:
#                 if target_field in extracted:
#                     continue
#
#                 # 遍历表体单元格
#                 field_cell = None
#                 for cell in body_cells:
#                     if not isinstance(cell, dict):
#                         continue
#                     cell_text = cell.get('words', '').strip()
#                     if target_field in cell_text:
#                         field_cell = cell
#                         break
#
#                 if not field_cell:
#                     continue
#
#                 # 提取表体字段对应的值（右侧相邻单元格）
#                 value_text = self._get_right_value(field_cell, header_cells + body_cells)
#                 # 清洗数值
#                 cleaned_value = self.clean_value_text(value_text)
#                 extracted[target_field] = cleaned_value if cleaned_value else value_text
#
#         return extracted
#
#     def _get_right_value(self, field_cell, all_cells):
#         """辅助函数：获取字段单元格右侧相邻的值"""
#         field_row = field_cell.get('row_start', 0)
#         field_col = field_cell.get('col_start', 0)
#         value_text = "未识别到"
#
#         # 按列号排序，优先取右侧相邻列
#         sorted_cells = sorted(
#             [c for c in all_cells if isinstance(c, dict)],
#             key=lambda x: x.get('col_start', 0)
#         )
#
#         for value_cell in sorted_cells:
#             val_row = value_cell.get('row_start', 0)
#             val_col = value_cell.get('col_start', 0)
#             # 同 row + 列号大于字段列（右侧）
#             if val_row == field_row and val_col > field_col:
#                 value_text = value_cell.get('words', '').strip()
#                 # 过滤无效值
#                 if value_text in ['-', '—', '无', 'NULL', 'null', '']:
#                     continue
#                 break
#
#         return value_text
#
#     def clean_value_text(self, value_text):
#         """
#         数值清洗函数（保留核心逻辑）：
#         - 去除字母和无效符号
#         - 提取斜杠后的数字（如"100/5" → "5"）
#         - 保留整数、小数（如"2.5A" → "2.5"，"3-5" → "35"）
#         """
#         if not value_text or value_text == "未识别到":
#             return ""
#
#         # 步骤1：去除所有字母（a-z, A-Z）
#         value_without_letters = re.sub(r'[A-Za-z]', '', value_text)
#
#         # 步骤2：去除横线、空格等干扰符号（保留小数点和斜杠）
#         cleaned = re.sub(r'[-—_ ,，\s]', '', value_without_letters)
#
#         # 步骤3：处理斜杠（仅保留斜杠后的数字，如"100/5" → "5"）
#         if '/' in cleaned or '\\' in cleaned:
#             # 分割并取最后一部分（兼容/和\）
#             parts = re.split(r'[/\\]', cleaned)
#             cleaned = parts[-1] if parts else cleaned
#
#         # 步骤4：仅保留数字和小数点（过滤其他特殊符号）
#         final_cleaned = re.sub(r'[^\d.]', '', cleaned)
#
#         # 处理多个小数点（仅保留第一个）
#         if final_cleaned.count('.') > 1:
#             first_dot = final_cleaned.index('.')
#             final_cleaned = final_cleaned[:first_dot + 1] + final_cleaned[first_dot + 1:].replace('.', '')
#
#         return final_cleaned.strip()
#

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
                'first': ['Ⅰ段', 'I段', '1段', '一段'],
                'primary': ['过流', '电流'],
                'secondary': ['定值', '值', '设置'],
                'exclude': ['时间', '时限', 'Ⅱ段', 'Ⅱ段', '2段', '二段', 'Ⅲ段', 'III段', '3段', '三段']
            },
            '过流I段时限定值': {
                'must': ['过流'],
                'first': ['Ⅰ段', 'I段', '1段', '一段'],
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
                'exclude': ['电流', 'Ⅱ段', 'II段', '2段', '二段', 'Ⅱ段', 'II段', '2段', '二段', '反时限']
            },
            # 零序保护
            '零序过流I段电流定值': {
                'must': ['零序'],
                'first': ['Ⅰ段', 'I段', '1段', '一段'],
                'primary': ['零序', '过流', '电流'],
                'secondary': ['定值', '值'],
                'exclude': ['时间', '时限', 'Ⅱ段', 'II段', '2段', '二段', 'Ⅲ段', 'III段', '3段', '三段']
            },
            '零序过流I段时限定值': {
                'must': ['零序'],
                'first': ['Ⅰ段', 'I段', '1段', '一段'],
                'primary': ['零序', '过流', '时间', '时限'],
                'secondary': ['定值', '值'],
                'exclude': ['电流', 'Ⅰ段', 'I段', '1段', '一段', 'Ⅲ段', 'III段', '3段', '三段', '反时限']
            },
            '零序过流II段电流定值': {
                'must': ['零序'],
                'first': ['Ⅱ段', 'II段', '2段', '二段'],
                'primary': ['零序', '过流', '电流'],
                'secondary': ['定值', '值'],
                'exclude': ['时间', '时限', 'Ⅰ段', 'I段', '1段', '一段', 'Ⅲ段', 'III段', '3段', '三段']
            },
            '零序过流II段时限定值': {
                'must': ['零序'],
                'first': ['Ⅱ段', 'II段', '2段', '二段'],
                'primary': ['零序', '过流', '时间', '时限'],
                'secondary': ['定值', '值'],
                'exclude': ['电流', 'Ⅰ段', 'I段', '1段', '一段', 'Ⅲ段', 'III段', '3段', '三段', '反时限']
            },
            '零序过流III段电流定值': {
                'must': ['零序'],
                'first': ['Ⅲ段', 'III段', '3段', '三段'],
                'primary': ['零序', '过流', '电流'],
                'secondary': ['定值', '值'],
                'exclude': ['时间', '时限', 'Ⅰ段', 'I段', '1段', '一段', 'Ⅱ段', 'II段', '2段', '二段']
            },
            '零序过流III段时限定值': {
                'must': ['零序'],
                'first': ['Ⅲ段', 'III段', '3段', '三段'],
                'primary': ['零序', '过流', '时间', '时限'],
                'secondary': ['定值', '值'],
                'exclude': ['电流', 'Ⅰ段', 'I段', '1段', '一段', 'Ⅱ段', 'II段', '2段', '二段', '反时限']
            }
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
            if best_score >= 3.0:  # 至少匹配2个主要关键词或等效分数
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
            if self._fuzzy_contains(text, keyword):
                match = 1
        if match == 0:
            return score
        # 关键第一词匹配（权重高）
        for keyword in patterns['first']:
            if self._fuzzy_contains(text, keyword):
                score += 4.0  # 主要关键词得4分
        # 主要关键词匹配（权重高）
        for keyword in patterns['primary']:
            if self._fuzzy_contains(text, keyword):
                score += 2.0  # 主要关键词得2分

        # 次要关键词匹配（权重低）
        for keyword in patterns['secondary']:
            if self._fuzzy_contains(text, keyword):
                score += 0.5  # 次要关键词得0.5分

        # 排除关键词惩罚（如果包含排除关键词，大幅扣分）
        for keyword in patterns['exclude']:
            if self._fuzzy_contains(text, keyword):
                score -= 5.0  # 排除关键词扣3分

        # 确保分数不为负
        return max(score, 0.0)

    def _fuzzy_contains(self, text: str, keyword: str) -> bool:
        """模糊包含判断"""
        # 完全包含
        if keyword in text:
            return True

        # 相似度匹配（处理OCR识别错误）
        similarity = difflib.SequenceMatcher(None, text, keyword).ratio()
        if similarity > 0.8:  # 相似度阈值
            return True

        # 处理常见的OCR识别错误
        common_errors = {
            'I': ['Ⅰ', 'l', '丨', '1'],
            'II': ['Ⅱ', 'll', '2'],
            'III': ['Ⅲ', 'lll', '3'],
            '段': ['断'],
            '流': ['琉'],
            '电': ['龟'],
            '保': ['堡'],
            '护': ['户']
        }

        # 检查常见错误替换
        for correct, errors in common_errors.items():
            for error in errors:
                if error in text and correct == keyword:
                    return True
                if keyword in text.replace(error, correct):
                    return True

        return False

    def _clean_parameter_name(self, param: str) -> str:
        """清理参数名"""
        # 去除标点符号和空格
        cleaned = re.sub(r'[^\w\u4e00-\u9fffⅠⅡⅢ]', '', param)
        return cleaned


class SmartTableExtractor:
    """智能表格提取器 - 使用关键词匹配，支持防重复映射"""

    def __init__(self, parameter_mapper: KeywordBasedParameterMapper):
        self.mapper = parameter_mapper

    def extract_from_tables(self, table_data: list, format_type: str) -> dict:
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
                table_results = self._extract_format2_with_keywords(table_structure, mapped_std_params)
            else:
                table_results = self._extract_format1_with_keywords(table_structure, mapped_std_params)

            print("当前表格提取结果:", table_results)

            # 合并结果，并更新已映射参数集合
            for key, value in table_results.items():
                if key not in results or results[key] == "未识别到":
                    results[key] = value


        print("最终提取结果:", results)
        return results

    def _extract_format2_with_keywords(self, table_array: list, mapped_std_params: set) -> dict:
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
                standardized_name = self.mapper.map_to_standard(combined_name, current_protection, mapped_std_params)

                if standardized_name and standardized_name not in mapped_std_params:
                    results[standardized_name] = param_value
                    mapped_std_params.add(standardized_name)
                    print(f"  提取结果: {standardized_name} = {param_value}")
                    # 注意：这里不在循环内更新mapped_std_params，而是在外层统一更新

        return results

    def _extract_format1_with_keywords(self, table_array: list, mapped_std_params: set) -> dict:
        """使用关键词匹配提取格式1数据 - 支持防重复映射"""
        results = {}

        if len(table_array) < 2:
            return results

        headers = table_array[0]
        data_rows = table_array[1:]

        # 智能查找列索引
        name_idx = self._find_column_by_keywords(headers, ['定值名称', '名称'])
        value_idx = self._find_column_by_keywords(headers, ['定值', '整定值', '数值'])

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
                standardized_name = self.mapper.map_to_standard(param_name, current_function, mapped_std_params)

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

        # 检查是否包含数字或特殊字符
        if re.search(r'[\d/]', value):
            return True

        return False

class BasicInfoExtractor:
    """基础信息提取器（CT变比、PT变比等）"""

    def __init__(self):
        self.basic_info_patterns = {
            'CT变比': r'CT变比[:：\s]*([\d/]+[A]?)',
            'PT变比': r'PT变比[:：\s]*([\d/\.]+[V]?)',
            '零序CT变比': r'零序CT变比[:：\s]*([\d/]+[A]?)',
            '定值单编号': r'定值单编号[:：\s]*([\w\d-]+)',
            '被保护设备': r'被保护设备[:：\s]*([^\s]+)',
            '装置型号': r'装置型号[:：\s]*([A-Za-z0-9\-]+)'
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

    def extract_protection_settings(self, image_path, target_params=None):
        all_results = {}

        # 获取表格识别结果
        table_result = self.table_ocr(image_path)
        # 获取通用OCR结果用于基础信息提取
        ocr_text_result = self.general_ocr(image_path)
        ocr_text = self._get_full_text(ocr_text_result) if ocr_text_result else ""

        # 1. 从表格中提取保护定值
        if table_result and 'tables_result' in table_result:
            # 检测格式
            format_type = self.format_detector.detect_format(ocr_text, table_result['tables_result'])
            print(f"检测到定值单格式: {format_type}")

            # 提取表格数据
            table_settings = self.table_extractor.extract_from_tables(
                table_result['tables_result'],
                format_type
            )
            all_results.update(table_settings)

        # # 2. 从通用文本中提取基础信息
        # if ocr_text:
        #     basic_info = self.basic_info_extractor.extract_basic_info(ocr_text)
        #     all_results.update(basic_info)

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

        # 步骤1：去除所有字母（保留斜杠和数字相关符号）
        value_without_letters = re.sub(r'[A-Za-z]', '', value_text)

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
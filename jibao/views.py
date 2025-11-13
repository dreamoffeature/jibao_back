import datetime
import os
from collections import defaultdict

import requests
from django.utils import timezone
from DjangoProject2 import settings
from django.http import JsonResponse
from rest_framework.views import APIView
from .models import Task, TaskImage, DeviceTemplate, ProtectTemplate
from.serializers import *
from .ocr_model import get_baidu_ocr_instance
ocr = get_baidu_ocr_instance()  # 使用增强版OCR


class CreateTaskBaseView(APIView):
    """创建任务基础信息接口（限制每人最多10条，超出删除最早一条）"""

    def post(self, request):
        openid = request.data.get('openid')
        task_name = request.data.get('task_name')
        if not openid or not task_name:
            return Response({
                "code": 400,
                "message": "openid和task_name是必填项"
            }, status=status.HTTP_400_BAD_REQUEST)

        # 关键逻辑：查询用户现有任务数，超出10条则删除最早的
        user_tasks = Task.objects.filter(user_openid=openid).order_by('create_time')  # 按创建时间升序（最早在前）
        if user_tasks.count() >= 10:
            oldest_task = user_tasks.first()  # 获取最早创建的任务
            # 先删除该任务关联的所有图片（物理文件+数据库记录）
            task_images = TaskImage.objects.filter(task=oldest_task)
            for img in task_images:
                if img.image and hasattr(img.image, 'path') and os.path.exists(img.image.path):
                    os.remove(img.image.path)
                img.delete()
            # 再删除任务本身
            oldest_task.delete()

        # 正常创建新任务
        task = Task.objects.create(
            user_openid=openid,
            task_name=task_name
        )
        serializer = TaskSerializer(task)
        return Response({
            "code": 200,
            "data": serializer.data
        }, status=status.HTTP_201_CREATED)


class UploadTaskImageView(APIView):
    """上传单张图片接口"""

    def post(self, request):
        task_id = request.data.get('task_id')
        try:
            task = Task.objects.get(id=task_id)
        except Task.DoesNotExist:
            return Response({
                "code": 404,
                "message": "任务不存在"
            }, status=status.HTTP_404_NOT_FOUND)
        image_file = request.FILES.get('image')
        if not image_file:
            return Response({
                "code": 400,
                "message": "请提供图片文件"
            }, status=status.HTTP_400_BAD_REQUEST)

        task_image = TaskImage.objects.create(
            task=task,
            image=image_file
        )
        serializer = TaskImageSerializer(task_image)
        return Response({
            "code": 200,
            "message": "图片上传成功",
            "data": serializer.data
        }, status=status.HTTP_201_CREATED)


class RecentTasksView(APIView):
    """获取用户最近任务接口"""

    def get(self, request):
        # 从请求头获取openid
        openid = request.headers.get('openid')
        if not openid:
            return Response({
                "code": 400,
                "message": "缺少openid参数"
            })

        try:
            # 获取最近7天的任务，按创建时间倒序排列，最多返回5条
            seven_days_ago = timezone.now() - datetime.timedelta(days=7)
            recent_tasks = Task.objects.filter(
                user_openid=openid,
                create_time__gte=seven_days_ago
            ).order_by('-create_time')[:5]

            # 格式化任务数据
            task_list = []
            for task in recent_tasks:
                # 获取任务的第一张图片作为展示图
                first_image = task.images.first()
                image_url = first_image.image.url if first_image else ""
                print(task.status)
                task_list.append({
                    "id": task.id,
                    "taskName": task.task_name,
                    "deviceModel": task.device_model,
                    "time": task.create_time.strftime("%Y-%m-%d %H:%M"),
                    "type": task.status ,
                    "typeName": "已完成" if task.status == "completed" else ("处理中" if task.status == "processing" else "失败"),
                    "location": f"{len(task.images.all())}张图片",
                    "imageUrl": image_url
                })
            return Response({
                "code": 200,
                "data": {
                    "taskList": task_list
                }
            })

        except Exception as e:
            return Response({
                "code": 500,
                "message": f"服务器错误: {str(e)}"
            })


class TaskDetailView(APIView):
    def get(self, request, task_id):
        openid = request.headers.get('openid')
        try:
            # 获取任务及关联的图片（使用select_related优化查询）
            task = Task.objects.select_related().get(id=task_id, user_openid=openid)
        except Task.DoesNotExist:
            return Response({"code": 404, "message": "任务不存在"}, status=404)

        # 查询该任务的所有图片（按上传时间排序）
        task_images = TaskImage.objects.filter(
            task=task
        ).order_by('upload_time')
        try:
            template = ProtectTemplate.objects.filter(protect_type=task.device_type).first()
        except ProtectTemplate.DoesNotExist:
            return Response({"code": 400, "message": f"未找到{task.device_type}对应的装置模板"}, status=400)

        # 同时记录参数所属层级（可选，用于前端展示层级关系）
        param_with_level = []
        required_params=[]
        protection_dict = template.protection_structure
        for level_name, param_list in protection_dict.items():
            # 跳过空列表，避免无效参数
            if not param_list or not isinstance(param_list, list):
                continue
            # 提取内层参数，添加到总列表
            required_params.extend(param_list)
            # 记录参数与层级的对应关系（可选，如需前端按层级显示可保留）
            param_with_level.extend([{"level": level_name, "param": p} for p in param_list])
        # 构造图片列表数据（包含ID、URL、上传时间）
        images = []
        for img in task_images:
            images.append({
                "id": img.id,  # 图片ID（用于删除等操作）
                "imageUrl": request.build_absolute_uri(img.image.url),  # 完整图片URL
                "uploadTime": img.upload_time.strftime("%Y-%m-%d %H:%M:%S")  # 上传时间
            })

        return Response({
            "code": 200,
            "data": {
                "task_name": task.task_name,
                "status": task.status,
                "images": images,  # 补充图片列表
                "create_time": task.create_time.strftime("%Y-%m-%d %H:%M:%S"),
                "ocr_result": task.ocr_result,
                "test_files": [],
                "device_model":task.device_model,
                "device_type":task.device_type,
                "param_with_level":param_with_level
            }
        })
class TaskHistoryView(APIView):
    """查看用户历史任务"""

    def get(self, request):
        # 1. 获取小程序传递的OpenID（从请求头或参数中获取）
        openid = request.GET.get("openid")
        if not openid:
            return JsonResponse({"code": 400, "message": "缺少openid参数"})

        # 2. 查询该用户的所有任务（按创建时间倒序）
        tasks = Task.objects.filter(user_openid=openid).order_by("-create_time")

        # 3. 转换为前端需要的格式
        task_list = [task.to_dict() for task in tasks]
        return JsonResponse({
            "code": 200,
            "data": {
                "task_count": tasks.count(),
                "task_list": task_list
            }
        })


class DeleteTaskImageView(APIView):
    """删除任务图片接口"""

    def post(self, request, image_id):
        openid = request.headers.get('openid')
        if not openid:
            return Response({"code": 401, "message": "未登录"}, status=status.HTTP_401_UNAUTHORIZED)

        try:
            # 确保图片属于当前用户的任务
            image = TaskImage.objects.select_related('task').get(
                id=image_id,
                task__user_openid=openid
            )
        except TaskImage.DoesNotExist:
            return Response({"code": 404, "message": "图片不存在"}, status=status.HTTP_404_NOT_FOUND)

        # 物理删除图片文件
        if image.image and hasattr(image.image, 'path') and os.path.exists(image.image.path):
            os.remove(image.image.path)

        # 数据库删除记录
        image.delete()
        return Response({"code": 200, "message": "图片删除成功"})


class EditOcrResultView(APIView):
    """编辑OCR识别结果接口"""

    def post(self, request, task_id):
        openid = request.headers.get('openid')
        new_ocr_result = request.data.get('ocr_result')  # 前端提交的新OCR结果列表

        if not openid:
            return Response({"code": 401, "message": "未登录"}, status=status.HTTP_401_UNAUTHORIZED)

        if not isinstance(new_ocr_result, list):
            return Response({"code": 400, "message": "OCR结果格式错误（需为数组）"}, status=status.HTTP_400_BAD_REQUEST)

        try:
            task = Task.objects.get(id=task_id, user_openid=openid)
        except Task.DoesNotExist:
            return Response({"code": 404, "message": "任务不存在"}, status=status.HTTP_404_NOT_FOUND)

        # 保存编辑后的OCR结果（格式：[{fieldName, fieldValue, fieldUnit}, ...]）
        task.ocr_result = new_ocr_result
        task.save()

        return Response({
            "code": 200,
            "message": "OCR结果更新成功",
            "data": {"ocr_result": task.ocr_result}
        })


class WxLoginView(APIView):
    """微信小程序登录，获取OpenID"""

    def post(self, request):
        # 1. 获取小程序传递的code
        code = request.data.get("code")
        if not code:
            return JsonResponse({"code": 400, "message": "缺少code参数"})

        # 2. 调用微信官方接口换取OpenID
        # 微信接口地址：https://api.weixin.qq.com/sns/jscode2session
        params = {
            "appid": settings.WX_APPID,  # 你的小程序AppID
            "secret": settings.WX_SECRET,  # 你的小程序AppSecret
            "js_code": code,
            "grant_type": "authorization_code"
        }
        response = requests.get("https://api.weixin.qq.com/sns/jscode2session", params=params)
        result = response.json()
        print(result)
        # 3. 处理返回结果（成功包含openid，失败包含errcode）
        if "errcode" in result:
            return JsonResponse({
                "code": 400,
                "message": f"登录失败：{result.get('errmsg')}"
            })

        # 4. 返回OpenID给小程序（无需存储用户信息，直接用OpenID关联任务）
        return JsonResponse({
            "code": 200,
            "data": {
                "openid": result.get("openid"),
                "session_key": result.get("session_key")  # 可选，如需解密用户信息可保留
            }
        })


class RecognizeDeviceModelView(APIView):
    """识别装置型号，并从 DeviceTemplate 解析保护层级+对应保护列表（适配最新结构）"""

    def post(self, request, task_id):
        openid = request.headers.get('openid')
        # 1. 权限校验：确认任务归属
        try:
            task = Task.objects.get(id=task_id, user_openid=openid)
        except Task.DoesNotExist:
            return Response({"code": 404, "message": "任务不存在或无权限"}, status=404)

        # 2. 识别并存储装置型号（处理无图片场景）
        device_model = None
        if task.images.exists():
            device_model = ocr.extract_device_model(task.images.first().image.path,'装置型号')
        task.device_model = device_model or "未知型号"


        # 3. 初始化返回核心数据（protect_type=保护层级，sub_protect_type=保护列表）
        protect_fault_relation = []  # 前端最终接收的列表
        template_found = True        # 标记是否找到匹配的装置模板
        if device_model:
            try:
                # 1. 获取所有 DeviceTemplate 的 device_model 列表
                all_device_models = DeviceTemplate.objects.values_list('device_model', flat=True)

                # 2. 筛选出被 task.device_model 包含的子串
                matching_models = [dm for dm in all_device_models if dm in task.device_model]
                if matching_models:
                    # 3. 查找匹配的记录
                    template = DeviceTemplate.objects.filter(device_model__in=matching_models).first()
                    protect_fault_relation = template.protection_structure  # 简化后的结构：[]
                    task.device_type = template.device_type
                else:
                    # 未识别到装置型号：返回空列表+提示
                    template_found = False
                    protect = ProtectTemplate.objects.get(protect_type=task.device_type)
                    protect_fault_relation = list(protect.protection_structure.keys())
            except DeviceTemplate.DoesNotExist:
                # 未找到模板：返回默认的“保护层级-保护列表”映射
                template_found = False
                protect_fault_relation = []

            except Exception as e:
                # 其他异常（如结构解析错误）
                print(f"模板解析异常: {str(e)}")
                return Response({"code": 500, "message": "装置保护结构解析失败"}, status=500)
        else:
            # 未识别到装置型号：返回空列表+提示
            template_found = False
            protect_fault_relation = []
        task.save()
        # 4. 构建最终返回结果
        response = {
            "code": 200,
            "data": {
                "device_model": device_model,          # 识别的装置型号
                "device_type":task.device_type,
                "protect_fault_relation": protect_fault_relation,  # 核心映射关系
                "template_found": template_found       # 模板匹配状态（前端可提示）
            }
        }

        # 未找到模板时添加提示信息
        if not template_found:
            response["message"] = f"未找到「{device_model or '未知型号'}」的装置模板，已使用默认保护配置"

        return Response(response)


class ReloadModelConfigView(APIView):
    """根据用户手动输入的装置型号重新加载保护/故障类型配置"""

    def post(self, request, task_id):
        openid = request.headers.get('openid')
        device_model = request.data.get('device_model', '').strip()
        device_type = request.data.get('device_type', '').strip()
        try:
            task = Task.objects.get(id=task_id, user_openid=openid)
        except Task.DoesNotExist:
            return Response({"code": 404, "message": "任务不存在"}, status=404)

        # 更新任务表中的装置型号（用户手动修改后）
        task.device_model = device_model
        task.device_type = device_type
        task.save()

        if device_type:
            try:
                protect = ProtectTemplate.objects.get(protect_type=device_type)
            except ProtectTemplate.DoesNotExist:
                return Response({"code": 500, "message": "保护模板表不存在该保护类型"}, status=500)
            protect_fault_relation = list(protect.protection_structure.keys() )
        else:
            protect_fault_relation = []


        return Response({
            "code": 200,
            "data": {
                'device_type': device_type,
                "protect_fault_relation": protect_fault_relation,
            }
        })


class SmartRecognitionStrategy:
    """
    智能识别策略 - 根据不同参数类型采用不同策略
    """

    def __init__(self, required_params):
        self.required_params = required_params
        self.param_categories = self._categorize_parameters(required_params)

    def _categorize_parameters(self, params):
        """将参数按类型分类"""
        categories = {
            'current_settings': [],  # 电流相关定值
            'time_settings': [],  # 时间相关定值
            'voltage_settings': [],  # 电压相关定值
            'ratio_settings': [],  # 变比相关
            'other_settings': []  # 其他参数
        }

        current_keywords = ['电流', '过流', '零序']
        time_keywords = ['时间', '时限', '延时']
        voltage_keywords = ['电压', '过压', '低压']
        ratio_keywords = ['变比', '变化', 'CT', 'PT']

        for param in params:
            param_lower = param.lower()

            if any(keyword in param_lower for keyword in current_keywords):
                categories['current_settings'].append(param)
            elif any(keyword in param_lower for keyword in time_keywords):
                categories['time_settings'].append(param)
            elif any(keyword in param_lower for keyword in voltage_keywords):
                categories['voltage_settings'].append(param)
            elif any(keyword in param_lower for keyword in ratio_keywords):
                categories['ratio_settings'].append(param)
            else:
                categories['other_settings'].append(param)

        return categories

    def get_priority_recognition_order(self):
        """获取优先识别顺序"""
        # 通常变比和基础参数在表格开头，先识别这些可以提高效率
        priority_order = (
                self.param_categories['ratio_settings'] +
                self.param_categories['current_settings'] +
                self.param_categories['voltage_settings'] +
                self.param_categories['time_settings'] +
                self.param_categories['other_settings']
        )
        return priority_order


class TaskOcrRecognitionView(APIView):
    def post(self, request, task_id):
        # 1. 权限校验
        openid = request.headers.get("openid")
        if not openid:
            return Response({"code": 400, "message": "缺少openid请求头"}, status=400)

        try:
            task = Task.objects.get(id=task_id, user_openid=openid)
        except Task.DoesNotExist:
            return Response({"code": 404, "message": "任务不存在或无权限访问"}, status=404)

        # 2. 获取请求参数
        device_type = request.data.get("device_type")
        image_ids = request.data.get("image_ids", [])

        if not device_type:
            return Response({"code": 400, "message": "保护类型不能为空"}, status=400)

        # 3. 获取需识别的图片列表
        try:
            if image_ids:
                task_images = task.images.filter(id__in=image_ids).order_by("upload_time")
            else:
                task_images = task.images.all().order_by("upload_time")

            if not task_images.exists():
                return Response({"code": 400, "message": "未找到指定的定值单图片"}, status=400)
        except Exception as e:
            return Response({"code": 500, "message": f"获取图片失败：{str(e)}"}, status=500)

        # 4. 从模板获取需识别的参数列表
        try:
            template = ProtectTemplate.objects.filter(protect_type=device_type).first()
            if not template:
                return Response({"code": 400, "message": f"未找到{device_type}对应的装置模板"}, status=400)
            if not template.protection_structure:
                return Response({"code": 400, "message": f"装置模板{device_type}的定值参数配置为空"}, status=400)

            required_params = []
            protection_dict = template.protection_structure
            for level_name, param_list in protection_dict.items():
                # 跳过空列表，避免无效参数
                if not param_list or not isinstance(param_list, list):
                    continue
                # 提取内层参数，添加到总列表
                required_params.extend(param_list)
            # 校验参数有效性（排除空字符串、None等）
            required_params = [p for p in required_params if p and isinstance(p, str)]
            if not required_params:
                return Response({"code": 400, "message": f"装置模板{device_type}的定值参数存在无效值或无参数"},
                                status=400)

            print(f"最终需识别的参数列表: {required_params}")
        except Exception as e:
            return Response({"code": 500, "message": f"解析模板失败：{str(e)}"}, status=500)

        # 5. 使用增强版OCR识别
        try:
            merged_results = defaultdict(list)
            image_recognition_details = []

            # 记录已识别的参数
            recognized_params = set()

            for img_index, img in enumerate(task_images):
                # 如果所有参数都已识别，跳过剩余图片
                if len(recognized_params) == len(required_params):
                    img_details = {
                        "image_id": img.id,
                        "image_name": img.image.name.split("/")[-1],
                        "recognized_params": [{
                            "param_name": param,
                            "param_value": "跳过(已识别)",
                            "recognition_status": "skipped"
                        } for param in required_params]
                    }
                    image_recognition_details.append(img_details)
                    continue
                img_path = img.image.path

                # 只识别尚未识别的参数
                remaining_params = [p for p in required_params if p not in recognized_params]
                table_result = ocr.extract_protection_settings(img_path, remaining_params,device_type)
                # 记录单张图片的识别结果
                img_details = {
                    "image_id": img.id,
                    "image_name": img.image.name.split("/")[-1],
                    "recognized_params": []
                }
                # 处理识别结果
                newly_recognized = []
                for param in required_params:
                    if param in recognized_params:
                        img_details["recognized_params"].append({
                            "param_name": param,
                            "param_value": "已识别(前序图片)",
                            "recognition_status": "already_done"
                        })
                    else:
                        value = table_result.get(param, "未识别到")
                        merged_results[param].append(value)

                        status = "success" if value != "未识别到" else "fail"
                        img_details["recognized_params"].append({
                            "param_name": param,
                            "param_value": value,
                            "recognition_status": status
                        })

                        if value != "未识别到":
                            newly_recognized.append(param)

                # 更新已识别参数
                recognized_params.update(newly_recognized)
                image_recognition_details.append(img_details)

                print(f"图片 {img_index + 1}: 新识别参数 {newly_recognized}")
        except Exception as e:
            print('33333')
            return Response({"code": 500, "message": f"OCR识别失败：{str(e)}"}, status=500)

        # 6. 生成最终结果
        final_params = {}
        recognition_stats = {
            "total_params": len(required_params),
            "recognized_params": 0,
            "unrecognized_params": 0
        }

        for param in required_params:
            values = merged_results[param]
            valid_values = [v for v in values if v and v != "未识别到"]
            if valid_values:
                final_params[param] = valid_values[0]
                recognition_stats["recognized_params"] += 1
            else:
                final_params[param] = "未识别到"
                recognition_stats["unrecognized_params"] += 1

        # 7. 更新任务状态
        try:
            task.ocr_result = final_params
            task.save()
            print(
                f"✅ 任务{task_id}识别完成: {recognition_stats['recognized_params']}/{recognition_stats['total_params']}")
        except Exception as e:
            print(f"警告: 任务保存失败: {e}")
            return Response({"code": 500, "message": f"OCR结果写入数据库失败：{str(e)}"}, status=500)

        # 8. 返回结果
        return Response({
            "code": 200,
            "message": f"已完成{len(task_images)}张图片的识别",
        })


from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .models import Task
from .serializers import SaveOcrResultSerializer  # 假设已存在该序列化器


class SaveOcrResultView(APIView):
    """保存OCR结果到Task表的ocr_result字段（支持参数新增、更新和删除）"""

    def post(self, request, task_id):
        # 1. 验证用户身份
        openid = request.headers.get('openid')
        if not openid:
            return Response(
                {"code": 401, "message": "缺少用户标识（openid）"},
                status=status.HTTP_401_UNAUTHORIZED
            )

        # 2. 验证任务存在且归属当前用户
        try:
            task = Task.objects.get(id=task_id, user_openid=openid)
        except Task.DoesNotExist:
            return Response(
                {"code": 404, "message": "任务不存在或无权限访问"},
                status=status.HTTP_404_NOT_FOUND
            )

        # 3. 验证请求数据（支持params新增/更新，delete_keys删除）
        # 扩展序列化器支持delete_keys字段（允许为空）
        class ExtendedSaveOcrResultSerializer(SaveOcrResultSerializer):
            delete_keys = serializers.ListField(
                child=serializers.CharField(),
                required=False,
                default=[],
                help_text="需要删除的参数键名列表"
            )

        serializer = ExtendedSaveOcrResultSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(
                {"code": 400, "message": "数据格式错误", "errors": serializer.errors},
                status=status.HTTP_400_BAD_REQUEST
            )

        # 4. 处理参数（先删除再更新/新增）
        new_params = serializer.validated_data.get("params", {})
        delete_keys = serializer.validated_data.get("delete_keys", [])

        # 初始化现有结果（确保是字典）
        current_ocr_result = task.ocr_result or {}

        # 4.1 执行删除操作
        for key in delete_keys:
            if key in current_ocr_result:
                del current_ocr_result[key]
        print(new_params)
        # 4.2 执行更新/新增操作（新参数覆盖旧参数，新增参数追加）
        current_ocr_result.update(new_params)

        # 5. 保存更新
        task.ocr_result = current_ocr_result
        task.save(update_fields=["ocr_result"])

        # 6. 返回响应（包含删除和更新后的完整结果）
        return Response({
            "code": 200,
            "message": "OCR结果更新成功",
            "data": {
                "task_id": task.id,
                "deleted_keys": delete_keys,  # 返回本次删除的键
                "updated_params": new_params,  # 返回本次更新的参数
                "ocr_result": task.ocr_result  # 返回更新后的完整参数列表
            }
        })
from rest_framework import serializers
from.models import Task, TaskImage

class TaskSerializer(serializers.ModelSerializer):
    class Meta:
        model = Task
        fields = ['id', 'user_openid', 'task_name', 'create_time']

class TaskImageSerializer(serializers.ModelSerializer):
    class Meta:
        model = TaskImage
        fields = ['id', 'task', 'image', 'upload_time']

class SaveOcrResultSerializer(serializers.Serializer):
    """验证平级参数格式（键值对）"""
    # 直接接收平级参数字典（无需保护层级/类型字段，直接合并到ocr_result）
    params = serializers.DictField(
        child=serializers.CharField(allow_blank=True),  # 允许空字符串（如"未识别到"）
        allow_empty=False  # 不允许空字典提交
    )
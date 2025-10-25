from django.db import models


# 1. 定值单图片表（与任务一对多）
class TaskImage(models.Model):
    task = models.ForeignKey('Task', on_delete=models.CASCADE, related_name='images', verbose_name="关联任务")
    image = models.ImageField(upload_to="original_images/%Y%m%d/", verbose_name="定值单图片")
    upload_time = models.DateTimeField(auto_now_add=True, verbose_name="上传时间")
    # 可选：记录单张图片的OCR状态（如部分识别失败可单独标记）
    ocr_status = models.CharField(max_length=20, choices=[
        ("pending", "待识别"),
        ("success", "识别成功"),
        ("failed", "识别失败")
    ], default="pending", verbose_name="单图OCR状态")

    def to_dict(self):
        return {
            "id": self.id,
            "image_url": self.image.url,
            "upload_time": self.upload_time.strftime("%Y-%m-%d %H:%M:%S"),
            "ocr_status": self.ocr_status
        }

# 2. 任务表（修改后）
class Task(models.Model):
    id = models.AutoField(primary_key=True)
    user_openid = models.CharField(max_length=100, verbose_name="用户OpenID")
    task_name = models.CharField(max_length=100, verbose_name="任务名称")
    create_time = models.DateTimeField(auto_now_add=True, verbose_name="创建时间")
    ocr_result = models.JSONField(blank=True, null=True, verbose_name="合并后的OCR识别结果")  # 多图识别结果合并
    pts_file = models.FileField(upload_to="pts_files/%Y%m%d/", blank=True, null=True, verbose_name="生成的PTS文件")
    status = models.CharField(max_length=20, choices=[
        ("processing", "处理中"),
        ("completed", "已完成"),
        ("failed", "失败")
    ], default="processing", verbose_name="任务状态")
    device_model = models.CharField(max_length=20, choices=[],blank=True, null=True,verbose_name="装置型号")



class DeviceTemplate(models.Model):
    device_model = models.CharField(max_length=50, unique=True, verbose_name="装置型号")
    manufacture = models.CharField(max_length=50,verbose_name="生产厂家")
    device_type = models.CharField(max_length=50,choices=[
        ("line", "线路保护"),
        ("line_10", "10kv线路保护"),
        ("transform", "主变保护"),
        ("busbar", "母差保护")
    ], default="line",verbose_name="装置类型")
    # JSON格式存储保护层级与校验项，结构见下方示例
    protection_structure = models.JSONField(verbose_name="保护结构与校验项")
    params = models.JSONField(verbose_name="定值参数", null=True, blank=True)
    create_time = models.DateTimeField(auto_now_add=True, verbose_name="创建时间")
    def __str__(self):
        return self.device_model

class ProtectTemplate(models.Model):
    protect_type = models.CharField(max_length=50,choices=[
        ("line", "线路保护"),
        ("line_10","10kv线路保护"),
        ("transform", "主变保护"),
        ("busbar", "母差保护")
    ], default="line",verbose_name="保护类型")
    protection_structure = models.JSONField(verbose_name="保护及其参数")
    def __str__(self):
        return self.protect_type
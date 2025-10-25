"""
URL configuration for DjangoProject2 project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from .views import *

urlpatterns = [
    path('api/tasks/create-base/', CreateTaskBaseView.as_view(), name='create_task_base'),
    path('api/tasks/upload-image/', UploadTaskImageView.as_view(), name='upload_task_image'),
    path('api/wx/login/', WxLoginView.as_view(), name='wx_login'),
    # 最近任务接口
    path('api/tasks/recent/', RecentTasksView.as_view(), name='recent_tasks'),
    # OCR识别接口：POST /jibao/api/tasks/{taskId}/ocr/
    path('api/tasks/<int:task_id>/ocr/', TaskOcrRecognitionView.as_view(), name="task_ocr_recognition"),
    # 任务详情接口
    path('api/tasks/<int:task_id>/detail/', TaskDetailView.as_view(), name='task_detail'),
    # 删除图片接口：POST /jibao/api/task-images/{image_id}/delete/
    path('api/task-images/<int:image_id>/delete/', DeleteTaskImageView.as_view(), name='delete_task_image'),
    # 识别装置型号接口：POST /jibao/api/tasks/{task_id}/recognize-model/
    path('api/tasks/<int:task_id>/recognize-model/', RecognizeDeviceModelView.as_view(), name='recognize_device_model'),
    # 重新加载配置接口：POST /jibao/api/tasks/{task_id}/reload-model-config/
    path('api/tasks/<int:task_id>/reload-model-config/', ReloadModelConfigView.as_view(), name='reload_model_config'),
    # 编辑OCR结果接口：POST /jibao/api/tasks/{task_id}/edit-ocr/
    path('api/tasks/<int:task_id>/edit-ocr/', EditOcrResultView.as_view(), name='edit_ocr_result'),
    # 保存OCR结果接口：POST /api/tasks/{task_id}/save-ocr/
    path("api/tasks/<int:task_id>/save-ocr/", SaveOcrResultView.as_view(), name="save-ocr-result"),
]

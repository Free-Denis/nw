from django.urls import path
from core import views
from core import views_ai

urlpatterns = [
    path('', views.index, name='index'),
    path('upload_excel/', views.upload_excel, name='upload_excel'),
    path('app_state/', views.app_state, name='app_state'),
    path('save_settings/', views.save_settings, name='save_settings'),
    path('list_models/', views.list_models, name='list_models'),
    path('start_analysis/', views.start_analysis, name='start_analysis'),
    path('analysis_progress/', views.analysis_progress, name='analysis_progress'),
    path('lock_message/', views.lock_message, name='lock_message'),
    path('regenerate_text/', views.regenerate_text, name='regenerate_text'),
    path('regenerate_image/', views.regenerate_image, name='regenerate_image'),
    path('download_rated/', views.download_rated, name='download_rated'),
    path('news_image/', views.news_image, name='news_image'),
    # Prompt Laboratory
    path('lab/start/', views.lab_start, name='lab_start'),
    path('lab/upload_excel/', views.lab_upload_excel, name='lab_upload_excel'),
    path('lab/progress/', views.lab_progress, name='lab_progress'),
    path('lab/answer_questions/', views.lab_answer_questions, name='lab_answer_questions'),
    path('lab/iterate/', views.lab_iterate, name='lab_iterate'),
    path('lab/save_profile/', views.lab_save_profile, name='lab_save_profile'),
    path('lab/cancel/', views.lab_cancel, name='lab_cancel'),
    # Audit hypotheses
    path('hypo/start/', views_ai.hypo_start, name='hypo_start'),
    path('hypo/progress/', views_ai.hypo_progress, name='hypo_progress'),
    path('hypo/cancel/', views_ai.hypo_cancel, name='hypo_cancel'),
    # Anonymization
    path('anon/run/', views_ai.anon_run, name='anon_run'),
    path('anon/excel_start/', views_ai.anon_excel_start, name='anon_excel_start'),
    path('anon/excel_progress/', views_ai.anon_excel_progress, name='anon_excel_progress'),
    path('anon/excel_cancel/', views_ai.anon_excel_cancel, name='anon_excel_cancel'),
]

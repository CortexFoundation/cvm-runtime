from django.urls import re_path

from . import consumers

websocket_urlpatterns = [
    re_path(
        r'ws/web/mrt/execute/(?P<room_name>\w+)/$',
        consumers.MRTExecuteConsumer.as_asgi()),
    re_path(
        r'ws/web/yaml/init/(?P<room_name>\w+)/$',
        consumers.YAMLInitConsumer.as_asgi()),
    re_path(
        r'ws/web/yaml/update/(?P<room_name>\w+)/$',
        consumers.YAMLUpdateConsumer.as_asgi()),
    re_path(
        r'ws/web/yaml/clear/(?P<room_name>\w+)/$',
        consumers.YAMLClearConsumer.as_asgi()),
    re_path(
        r'ws/web/yaml/collect/(?P<room_name>\w+)/$',
        consumers.YAMLCollectConsumer.as_asgi()),
]

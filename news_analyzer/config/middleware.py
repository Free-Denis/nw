class JupyterHubProxyMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        script_name = os.environ.get('FORCE_SCRIPT_NAME', '')
        if script_name:
            request.META['SCRIPT_NAME'] = script_name
        return self.get_response(request)

import os

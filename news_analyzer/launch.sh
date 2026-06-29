#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

VENV_PYTHON="$SCRIPT_DIR/venv/bin/python"
DJANGO_PORT=8000

check_port() {
    local port=$1
    if ss -tuln | grep -q ":$port " ; then return 0; else return 1; fi
}

cleanup() {
    echo ""
    echo "Остановка Django..."
    [ ! -z "$DJANGO_PID" ] && kill $DJANGO_PID 2>/dev/null
    exit 0
}
trap cleanup SIGINT SIGTERM

echo "Запуск NewsAnalyzer..."

if [ ! -f "$VENV_PYTHON" ]; then
    echo "Python не найден в venv. Запустите setup.sh"
    exit 1
fi

# Переменные окружения для GigaChat API
export GIGACHAT_API_URL=${GIGACHAT_API_URL:-""}
export JPY_API_TOKEN=${JPY_API_TOKEN:-""}

export JUPYTERHUB_USER=${JUPYTERHUB_USER:-"user"}

if check_port $DJANGO_PORT; then
    echo "Django уже запущен (порт $DJANGO_PORT занят)"
else
    echo "Запуск Django..."
    $VENV_PYTHON manage.py runserver 0.0.0.0:$DJANGO_PORT > django.log 2>&1 &
    DJANGO_PID=$!

    echo "Ожидание Django (порт $DJANGO_PORT)..."
    timeout=30
    count=0
    while [ $count -lt $timeout ]; do
        if check_port $DJANGO_PORT; then
            echo ""
            echo "Система готова!"
            echo "Ссылка: https://jupyterhub-datalab.apps.prom-datalab.ca.sbrf.ru/user/${JUPYTERHUB_USER}/vscode/proxy/8000/"
            break
        fi
        if ! ps -p $DJANGO_PID > /dev/null; then
            echo ""
            echo "Django упал! Лог:"
            tail -n 30 django.log
            cleanup
            exit 1
        fi
        sleep 2
        ((count=count+2))
        echo -n "."
    done

    if ! check_port $DJANGO_PORT; then
        echo ""
        echo "Django не ответил. Лог:"
        tail -n 30 django.log
        cleanup
        exit 1
    fi
fi

wait


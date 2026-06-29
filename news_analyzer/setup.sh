#!/bin/bash

echo "Настройка окружения NewsAnalyzer..."

if ! command -v python &> /dev/null; then
    echo "Python3 не найден"
    exit 1
fi

echo "Введите токен SberOSC (или Enter, если зависимости установлены):"
read -s -p "Токен: " TOKEN
echo ""

# Venv
if [ ! -d "venv" ]; then
    echo "Создание виртуального окружения..."
    python -m venv venv
fi
source venv/bin/activate

# UV
echo "Установка UV..."
if [ ! -z "$TOKEN" ]; then
    pip install uv --index-url "https://token:$TOKEN@sberosc.ca.sbrf.ru/repo/pypi/simple" --trusted-host sberosc.ca.sbrf.ru
else
    pip install uv
fi

mkdir -p ~/.configs/uv
cat > ~/.configs/uv/uv.toml << EOL
allow-insecure-host = [
    "sberosc.ca.sbrf.ru"
]
[[index]]
name = "sberosc"
url = "https://sberosc.ca.sbrf.ru/repo/pypi/simple"
default = true
EOL

if [ ! -z "$TOKEN" ]; then
    uv auth login sberosc.ca.sbrf.ru --token $TOKEN
fi

# Зависимости
echo "Установка Python-зависимостей..."
if [ ! -z "$TOKEN" ]; then
    uv pip install -r requirements.txt \
        --index-url "https://token:$TOKEN@sberosc.ca.sbrf.ru/repo/pypi/simple" \
        --trusted-host sberosc.ca.sbrf.ru
else
    uv pip install -r requirements.txt
fi

mkdir -p storage

echo ""
echo "Настройка завершена!"
echo "Для запуска: source venv/bin/activate && sh launch.sh"

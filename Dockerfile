FROM python:3.11.9-slim

WORKDIR /app

COPY sg_lang.py autogen_graphflow.py ./

RUN pip install --no-cache-dir \
    "sglang[all]" \
    autogen-agentchat \
    "autogen-ext[openai]"

RUN python -m sglang.launch_server --model-path meta-llama/Meta-Llama-3-8B-Instruct --port 30000

CMD ["python", "sg_lang.py"]

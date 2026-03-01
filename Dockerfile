FROM python:3.11.9-slim

WORKDIR /app

COPY sg_lang.py autogen_graphflow.py ./

RUN pip install --no-cache-dir \
    sglang \
    autogen-agentchat \
    "autogen-ext[openai]"

CMD ["python", "sg_lang.py"]

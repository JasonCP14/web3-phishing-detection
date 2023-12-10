FROM python:3.11.7-slim

RUN apt-get update
RUN apt-get install -y python3 python3-venv python3-pip

WORKDIR /usr/src/app

ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

COPY . ./
RUN pip install -r requirements.txt

# Expose the port for Flask app
EXPOSE 5000

# Run the Flask app
CMD ["python3", "app.py"]

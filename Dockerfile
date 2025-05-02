# pull python base image
FROM python:3.10

# set the working directory
WORKDIR /app

COPY requirements.txt .



# update pip
RUN pip install --upgrade pip

# install dependencies
RUN pip install -r requirements.txt



# copy application files
COPY xgboost-model.pkl /app/
COPY app.py /app/



# expose port for application
EXPOSE 7860

# start fastapi application
CMD ["python", "app.py"]
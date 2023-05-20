FROM python:3.8.13 as base

ARG UID
ARG GID

# Add a non-root user
RUN addgroup --gid $GID app
RUN adduser --disabled-login --geco '' --uid $UID --gid $GID app
RUN mkdir -p ./src/feedback ./src/static/uploads
RUN touch ./src/feedback/feedback


# Append /home/app/.local/bin/ to PATH variable because
# gunicorn is installed there.
ENV PYTHONPATH=$PYTHONPATH:/src
ENV PATH=$PATH:/home/app/.local/bin

#COPY --chown=app:app ./ /src
COPY ./ /src

RUN chown -R app:app /src
RUN chmod u+rw,g=r -R /src

USER app

WORKDIR /src

##Changing user APP location to avoid permissions erros

RUN pip install --upgrade pip && pip install -r requirements.txt

FROM base as test
RUN ["pytest", "-v", "/src/tests"]

FROM base as build
ENTRYPOINT ["gunicorn", "--workers=8", "--bind", "0.0.0.0:5000", "app:app"]
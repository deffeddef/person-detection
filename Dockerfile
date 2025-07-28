# Use a specific version of Python for reproducibility
FROM python:3.10.13-slim

# Set the working directory in the container
WORKDIR /app

# Create a non-root user to run the application
RUN useradd --create-home appuser
USER appuser

# Copy requirements and install dependencies
# This is done in a separate step to leverage Docker layer caching.
COPY --chown=appuser:appuser requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the models directory
# The models are large and change infrequently, so they get their own layer.
COPY --chown=appuser:appuser models/ ./models/

# Copy the application source code
COPY --chown=appuser:appuser src/ ./src/

# Set the entrypoint to run the prediction script as a module.
# This ensures relative imports work correctly.
# The default command is set to --help for user convenience.
ENTRYPOINT ["python", "-m", "src.predict"]
CMD ["--help"]
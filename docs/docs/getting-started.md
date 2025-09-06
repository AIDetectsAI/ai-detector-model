# Getting started

This article provides information on how to quickly set up and start working with the **AI Detector Model** project. It covers environment setup, dependency installation, preparing datasets, code formatting and running the initial scripts to ensure everything is working correctly.


## 👟 Quickstart

This section will guide you through setting up the project environment, installing dependencies, and preparing for development.

---

### 1️⃣ Create Python environment

Make sure you have Python 3.11.9 installed. Then create a virtual environment using the provided Makefile:

```bash
make create_environment
```

This will create a new Pipenv environment configured with the correct Python version.
You can activate it with:
```bash
pipenv shell
```

### 2️⃣ Install dependencies

Install all required Python packages:

```bash
make requirements          # Install globally
pipenv run make requirements  # Install inside pipenv environment
```

### 3️⃣ Environment setup (Optional)

This project uses a `.env` file for configuration. Create it from the template:

```bash
cp .env.example .env
```

> Currently .env is only used for dataset download

### 4️⃣ Prepare dataset

Download and process the dataset using the provided script:
```bash
make data
```

This will ensure that the raw, interim, and processed datasets are available in the correct folders.

### 5️⃣ Build and serve local documentation (Optional)

You can generate the project documentation and serve it locally for browsing:

```bash
make build_docs       # Build documentation
make serve_docs       # Serve docs at localhost:9000
```

The local docs are useful for quickly checking usage examples, model details, and project structure.

## 📝 Code formatting

This project follows standard Python code formatting and style conventions. To ensure consistency, the following tools are used:

- **Black** – automatically formats Python code to a uniform style  
- **isort** – sorts imports in a consistent order  
- **flake8** – checks for style violations and potential errors  

Before submitting any changes or pull requests, make sure to run:

```bash
make format   # Format code automatically
make lint     # Check code style and errors
```

Following these steps helps maintain clean, readable, and consistent code throughout the project.


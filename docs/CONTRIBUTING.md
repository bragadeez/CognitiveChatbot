# Contributing to Cognitive Chatbot

We welcome contributions to the Cognitive Chatbot project! Please read the guidelines below to help keep the codebase maintainable and robust.

## How to Contribute

### 1. Reporting Issues
* Search existing issues to ensure your bug or feature request has not already been reported.
* Create a detailed issue including clear reproduction steps, environment info, and actual vs. expected behavior.

### 2. Code Contributions
* Fork the repository and create your branch from the `main` branch: `git checkout -b feature/your-feature-name`.
* Ensure code compliance with project styling guidelines:
  * **Backend**: Follow PEP 8 styles for Python. Use FastAPI endpoints and Pydantic schemas.
  * **Frontend**: Follow React best practices, strict TypeScript mappings, and write clean, modular CSS inside `src/styles/`.
* Write clean, self-documenting code and update the documentation if interfaces change.
* Confirm that local tests and production builds run successfully before submitting:
  ```bash
  cd Implementation/frontend
  npm run build
  ```

### 3. Submitting Pull Requests
* Open a pull request targeting the `main` branch.
* Describe the exact changes made, their purpose, and links to any related issues.
* Request a review from the repository maintainers.

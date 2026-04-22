# Release Note Generator

A modular pipeline for automatically generating release notes from software development artifacts using self-hosted large language models (LLMs).

The pipeline extracts information from commits, issues, and pull requests, processes the data, and generates structured release notes in Markdown format.

The architecture separates data extraction, preprocessing, context integration, LLM-based generation, and postprocessing. Different LLM models or generation strategies can be used within the pipeline.

Output sections are generated based on the input data and can include categories such as Added, Changed, Fixed, Removed, and Security.

## Example

### Input:

[COMMITS] Add user management API

[COMMITS] Fix memory leak in auth module

[ISSUES] User cannot reset password

[PULL_REQUESTS] Implement role-based access control

### Generated Release Notes:

**Added**
- User management API.
- Role-based access control.

**Fixed**
- Memory leak in the authentication module.
- Issue preventing users from resetting passwords.

## Setup
```
pip install -r requirements.txt
```
```
python ds_main.py
```
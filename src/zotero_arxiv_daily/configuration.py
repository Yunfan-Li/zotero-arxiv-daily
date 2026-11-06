"""Opt-in migration of the pre-Hydra repository environment, without logging it."""
import os
import re

from omegaconf import open_dict


def _boolean(name, default=False):
    value = os.environ.get(name, '').strip().lower()
    if not value:
        return default
    if value in ('true', '1', 'yes'):
        return True
    if value in ('false', '0', 'no'):
        return False
    raise ValueError(f'{name} must be true or false (value withheld).')


def migrate_legacy_config(config):
    # Explicit CUSTOM_CONFIG files retain the upstream API-only defaults unless
    # they deliberately opt into legacy migration.
    if not config.get('legacy_env', False):
        return config
    query = os.environ.get('ARXIV_QUERY', '').strip()
    categories = [part.strip() for part in query.split('+')]
    if not query or any(not re.fullmatch(r'[A-Za-z][A-Za-z0-9.-]*(?:\.\*)?', c) for c in categories):
        raise ValueError('ARXIV_QUERY must contain arXiv categories joined with + (value withheld).')
    try:
        max_papers = int(os.environ.get('MAX_PAPER_NUM', '').strip() or '100')
        port = int(config.email.smtp_port)
    except (TypeError, ValueError):
        raise ValueError('SMTP_PORT and MAX_PAPER_NUM must be integers (values withheld).') from None
    if max_papers < 1 or not 1 <= port <= 65535:
        raise ValueError('SMTP_PORT or MAX_PAPER_NUM is out of range (values withheld).')
    with open_dict(config):
        config.source.arxiv.category = list(dict.fromkeys(categories))
        config.email.smtp_port = port
        config.executor.max_paper_num = max_papers
        config.executor.debug = _boolean('DEBUG')
        config.executor.send_empty = _boolean('SEND_EMPTY')
        config.llm.backend = 'api' if _boolean('USE_LLM_API') else 'local'
        config.llm.language = os.environ.get('LANGUAGE', '').strip() or 'English'
        # Retain the old gitignore rules verbatim rather than silently changing
        # their negation semantics into the new glob-based ignore_path setting.
        config.zotero.legacy_ignore = os.environ.get('ZOTERO_IGNORE', '')
        if config.llm.backend == 'api':
            if not config.llm.api.key:
                raise ValueError('USE_LLM_API requires OPENAI_API_KEY.')
            if not config.llm.api.base_url:
                config.llm.api.base_url = None
            if not config.llm.generation_kwargs.model:
                config.llm.generation_kwargs.model = 'gpt-4o-mini'
    return config

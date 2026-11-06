"""Migration and local inference tests. No credentials, email, or model downloads."""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from zotero_arxiv_daily.configuration import migrate_legacy_config
from zotero_arxiv_daily.local_llm import LocalLLM
from zotero_arxiv_daily.protocol import _request_llm


@pytest.fixture
def legacy(monkeypatch):
    for name in ('USE_LLM_API', 'DEBUG', 'SEND_EMPTY', 'MAX_PAPER_NUM',
                 'MODEL_NAME', 'OPENAI_API_KEY', 'OPENAI_API_BASE', 'ZOTERO_IGNORE'):
        monkeypatch.delenv(name, raising=False)
    for name, value in {'ZOTERO_ID': '123', 'ZOTERO_KEY': 'dummy-key',
                        'SENDER': 'sender@example.invalid', 'RECEIVER': 'receiver@example.invalid',
                        'SENDER_PASSWORD': 'dummy-password', 'SMTP_SERVER': 'smtp.example.invalid',
                        'SMTP_PORT': '465', 'ARXIV_QUERY': 'cs.AI+cs.CV+cs.LG',
                        'LANGUAGE': 'Chinese'}.items():
        monkeypatch.setenv(name, value)
    root = Path(__file__).resolve().parents[1] / 'config'
    return OmegaConf.merge(OmegaConf.load(root / 'base.yaml'), OmegaConf.load(root / 'custom.yaml'))


def test_legacy_defaults_use_free_local_chinese_model(legacy):
    cfg = migrate_legacy_config(legacy)
    assert cfg.executor.source == ['arxiv']
    assert cfg.source.arxiv.category == ['cs.AI', 'cs.CV', 'cs.LG']
    assert cfg.llm.backend == 'local'
    assert cfg.llm.language == 'Chinese'
    assert cfg.email.smtp_server == 'smtp.example.invalid'
    assert cfg.email.smtp_port == 465
    assert cfg.email.receiver == 'receiver@example.invalid'
    assert cfg.executor.send_empty is False
    assert cfg.executor.max_paper_num == 100
    assert not OmegaConf.missing_keys(cfg)


def test_empty_optional_environment_uses_typed_defaults(legacy, monkeypatch):
    for name in ('USE_LLM_API', 'SEND_EMPTY', 'MAX_PAPER_NUM', 'DEBUG'):
        monkeypatch.setenv(name, '')
    assert migrate_legacy_config(legacy).llm.backend == 'local'


def test_legacy_api_mode(legacy, monkeypatch):
    monkeypatch.setenv('USE_LLM_API', 'true')
    monkeypatch.setenv('OPENAI_API_KEY', 'dummy-api-key')
    monkeypatch.setenv('MODEL_NAME', 'example-model')
    cfg = migrate_legacy_config(legacy)
    assert cfg.llm.backend == 'api'
    assert cfg.llm.api.base_url is None
    assert cfg.llm.generation_kwargs.model == 'example-model'


def test_api_mode_requires_key(legacy, monkeypatch):
    monkeypatch.setenv('USE_LLM_API', 'true')
    with pytest.raises(ValueError, match='requires OPENAI_API_KEY'):
        migrate_legacy_config(legacy)


@pytest.mark.parametrize('name,value', [('ARXIV_QUERY', 'cat:cs.AI OR secret'),
                                       ('MAX_PAPER_NUM', 'secret'), ('SMTP_PORT', 'secret'),
                                       ('USE_LLM_API', 'secret')])
def test_invalid_config_does_not_echo_values(legacy, monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError) as error:
        migrate_legacy_config(legacy)
    assert value not in str(error.value)


def test_custom_configuration_is_not_overwritten(monkeypatch):
    monkeypatch.setenv('ARXIV_QUERY', 'cs.AI')
    cfg = OmegaConf.create({'executor': {'source': ['biorxiv']},
                            'llm': {'api_mode': 'response', 'language': 'French'}})
    before = OmegaConf.to_container(cfg)
    assert OmegaConf.to_container(migrate_legacy_config(cfg)) == before


def test_local_adapter_lazy_loads_once_and_limits_context(monkeypatch):
    calls = []
    class Model:
        def tokenize(self, text, add_bos=False):
            return list(text)
        def detokenize(self, tokens):
            return bytes(tokens)
        def create_chat_completion(self, **kwargs):
            calls.append(kwargs)
            return {'choices': [{'message': {'content': 'summary'}}]}
    loads = []
    def load(**kwargs):
        loads.append(kwargs)
        return Model()
    monkeypatch.setitem(sys.modules, 'llama_cpp', SimpleNamespace(Llama=SimpleNamespace(from_pretrained=load)))
    client = LocalLLM()
    assert not loads
    messages = [{'role': 'system', 'content': 'system'}, {'role': 'user', 'content': 'a' * 10000}]
    for _ in range(2):
        assert _request_llm(client, {'backend': 'local', 'generation_kwargs': {'max_tokens': 16384}}, messages) == 'summary'
    assert len(loads) == 1
    assert loads[0]['n_ctx'] == 5000
    assert loads[0]['repo_id'] == 'Qwen/Qwen2.5-3B-Instruct-GGUF'
    assert calls[0]['max_tokens'] == 768
    assert sum(len(m['content']) for m in calls[0]['messages']) <= 3976
    assert len(messages[1]['content']) == 10000


def test_legacy_ignore_retains_negation(legacy, monkeypatch):
    from zotero_arxiv_daily.executor import Executor
    from zotero_arxiv_daily.protocol import CorpusPaper
    from datetime import datetime
    monkeypatch.setenv('ZOTERO_IGNORE', 'archive/*\n!archive/keep')
    executor = Executor.__new__(Executor)
    executor.config = migrate_legacy_config(legacy)
    executor.include_path_patterns = executor.ignore_path_patterns = None
    papers = [CorpusPaper(path, '', datetime.now(), [path]) for path in ['archive/drop', 'archive/keep', 'active']]
    assert [p.title for p in executor.filter_corpus(papers)] == ['archive/keep', 'active']


def test_local_failure_is_not_silently_sent_as_abstract(monkeypatch):
    from zotero_arxiv_daily.protocol import Paper
    paper = Paper('arxiv', 'title', [], 'abstract', 'https://example.invalid')
    def fail(*args):
        raise RuntimeError('model unavailable')
    monkeypatch.setattr(paper, '_generate_tldr_with_llm', fail)
    with pytest.raises(RuntimeError, match='model unavailable'):
        paper.generate_tldr(LocalLLM(), {'backend': 'local'})


def test_workflow_config_override_preserves_literal_bytes(tmp_path):
    import os
    import subprocess
    import yaml
    root = Path(__file__).resolve().parents[1]
    (tmp_path / 'config').mkdir()
    target = tmp_path / 'config/custom.yaml'
    for workflow in ('main.yml', 'test.yml'):
        data = yaml.safe_load((root / '.github/workflows' / workflow).read_text())
        step = data['jobs']['calculate-and-send']['steps'][-1]
        script = step['run'].split('          uv run')[0] if '          uv run' in step['run'] else step['run'].split('uv run')[0]
        assert 'cat config' not in script
        for override in ('', ' \t\n', 'example: "literal\\nvalue"'):
            target.write_text('original\n')
            result = subprocess.run(['bash', '-c', script], cwd=tmp_path,
                                    env={**os.environ, 'CUSTOM_CONFIG': override}, capture_output=True, check=True)
            assert target.read_text() == (override + '\n' if override.strip() else 'original\n')
            assert result.stdout == b''


def test_executor_local_mode_does_not_construct_openai(legacy, monkeypatch):
    import zotero_arxiv_daily.executor as executor_module
    def unexpected_api(**kwargs):
        raise AssertionError('API client must not be constructed in local mode')
    monkeypatch.setattr(executor_module, 'OpenAI', unexpected_api)
    executor = executor_module.Executor(migrate_legacy_config(legacy))
    assert isinstance(executor.openai_client, LocalLLM)
    assert executor.openai_client._model is None


def test_main_failure_does_not_log_provider_secrets(config, monkeypatch, capsys):
    from zotero_arxiv_daily import main as main_module
    class BrokenExecutor:
        def __init__(self, cfg):
            raise RuntimeError('private-credential-do-not-log')
    monkeypatch.setattr(main_module, 'Executor', BrokenExecutor)
    with pytest.raises(SystemExit) as error:
        main_module.main.__wrapped__(config)
    assert error.value.code == 1
    captured = capsys.readouterr()
    assert 'private-credential-do-not-log' not in captured.out + captured.err
    assert 'Run failed (RuntimeError)' in captured.out

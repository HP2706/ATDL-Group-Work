"""All computed evaluation metrics reach scalar history, tables and durable summaries."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

from video3d.config import load_config
from video3d.logging import RunLogger, numeric_metrics


def test_numeric_metrics_flattens_and_excludes_missing_or_metadata() -> None:
    assert numeric_metrics({'viou':.2, 'chamfer':None, 'fscore':float('nan'), 'k':1,
                            'schema_version':3, 'empty_pred':False,
                            'mesh_surface':{'aligned':{'fscore':.7}}, 'sample_id':'object'}) == {
        'viou':.2, 'empty_pred':False, 'mesh_surface/aligned/fscore':.7}


def test_all_metrics_logged_without_reusing_training_optimizer_step(tmp_path: Path) -> None:
    logger = RunLogger(load_config('c3'))
    logger.run = MagicMock()
    logger.run.summary = {}
    rows = [{'sample_id':'a', 'viou':.2, 'chamfer':.1},
            {'sample_id':'b', 'viou':.4, 'chamfer':None}]
    logger.log_evaluation(rows[0], 1)
    logger.run.log.assert_called_once_with({'eval/sample_index':1,'eval/viou':.2,'eval/chamfer':.1},step=None)
    (tmp_path/'evaluation.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
    summary = logger.finish_evaluation(rows, tmp_path, 'evaluation', 2)
    assert summary['eval/complete']
    assert abs(summary['eval/mean/viou']-.3) < 1e-8
    assert summary['eval/count/chamfer'] == 1
    assert logger.run.summary == summary
    table = logger.run.log.call_args.args[0]['eval/results']
    assert set(table.columns) == {'sample_id','viou','chamfer'}
    assert len(table.data) == 2
    logger.run.log_artifact.assert_called_once()
    assert json.loads((tmp_path/'evaluation.summary.json').read_text()) == summary


def test_partial_evaluation_is_explicit_and_saved_without_wandb(tmp_path: Path) -> None:
    logger = RunLogger(load_config('c3'))
    summary = logger.finish_evaluation([{'sample_id':'a','viou':.2}], tmp_path, 'partial', 250)
    assert not summary['eval/complete'] and summary['eval/objects'] == 1
    assert (tmp_path/'partial.summary.json').exists()


def test_large_metric_set_keeps_artifact_metadata_bounded(tmp_path: Path) -> None:
    logger = RunLogger(load_config('c3'))
    logger.run = MagicMock()
    logger.run.summary = {}
    row = {'sample_id':'object', **{f'metric-{index}':float(index) for index in range(150)}}
    summary = logger.finish_evaluation([row], tmp_path, 'large', 1)
    artifact = logger.run.log_artifact.call_args.args[0]
    assert len(artifact.metadata) < 100
    assert artifact.metadata['metric_count'] == 150
    assert len([key for key in summary if key.startswith('eval/mean/')]) == 150

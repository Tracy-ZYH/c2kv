"""Explicit evaluation capacity overrides without rewriting training geometry."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

CAPACITY_FIELDS = frozenset({'max_workspace_tokens', 'max_sequence_tokens'})


def resolve_eval_packing(profile, path=None):
    training = deepcopy(profile['packing_contract'])
    effective = deepcopy(training)
    receipt = {'schema': 'a-event-native-runtime-packing-v1',
               'training_packing': training, 'effective_packing': effective,
               'override_source': None, 'override_sha256': None}
    if path is None:
        return receipt
    path = Path(path)
    content = path.read_bytes()
    value = json.loads(content)
    if set(value) != {'schema', 'capacity'} or value['schema'] != 'a-event-native-eval-capacity-v1':
        raise ValueError('Expected an explicit event-native evaluation capacity contract')
    capacity = value['capacity']
    if not isinstance(capacity, dict) or set(capacity) != CAPACITY_FIELDS:
        raise ValueError('Only workspace and sequence capacities may be overridden')
    if any(type(v) is not int or v <= 0 for v in capacity.values()):
        raise ValueError('Evaluation capacities must be positive integers')
    context = profile['model_geometry']['max_position_embeddings']
    if capacity['max_sequence_tokens'] > context:
        raise ValueError('Evaluation sequence capacity exceeds model context')
    if capacity['max_workspace_tokens'] + training['max_target_tokens'] > capacity['max_sequence_tokens']:
        raise ValueError('Workspace and target reservation exceed evaluation sequence capacity')
    effective.update(capacity)
    receipt.update(override_source=str(path.resolve()),
                   override_sha256=hashlib.sha256(content).hexdigest())
    return receipt

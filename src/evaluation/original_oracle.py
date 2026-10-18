"""Read-only, independent execution of hash-checked ILVR source for a bounded witness.

This module reconstructs pristine source bytes and executes separate source classes;
it never calls the restored generation adapter or its model class. It does not load
Reference, git, or another checkout at runtime. The current shared Transformers
support files are byte-identical to 13a67ed (recorded in original-provenance.json).
"""
from __future__ import annotations

import ast
import hashlib
from pathlib import Path
import sys
import types

ROOT = Path(__file__).resolve().parents[2]
EVAL_SOURCE = Path(__file__).parent/'original_source/original_eval.py.txt'
_SOURCE_CLASS = None


def _checked(source, expected):
    if hashlib.sha256(source.encode()).hexdigest()!=expected:
        raise ValueError('original source snapshot hash mismatch')
    return source


def source_model_class():
    global _SOURCE_CLASS
    if _SOURCE_CLASS is not None:
        return _SOURCE_CLASS
    from transformers.generation.utils import GenerationMixin as CurrentMixin
    generation=ROOT/'transformers/src/transformers/generation/_ilvr_original_utils.py'
    text=generation.read_text().replace('from .utils import GenerationMixin as _CurrentGenerationMixin\n','').replace(
        'class GenerationMixin(_CurrentGenerationMixin):','class GenerationMixin:')
    _checked(text,'acc203e30739d40abd36f96b93c8110369a9742717f4db050075387e3a241863')
    module=types.ModuleType('transformers.generation._ilvr_source_oracle')
    module.__file__=str(generation)
    module.__package__='transformers.generation'
    sys.modules[module.__name__]=module
    exec(compile(text,str(generation), 'exec'),module.__dict__)
    # Original PreTrainedModel inherited GenerationMixin. Current PreTrainedModel
    # still does. This local diamond preserves all original method owners without
    # changing any global class or substituting current generation algorithms.
    class SourceGenerationMixin(module.GenerationMixin,CurrentMixin):
        pass
    modeling=ROOT/'transformers/src/transformers/models/qwen2_5_vl/_ilvr_original_modeling.py'
    text=modeling.read_text().replace('from ...generation._ilvr_original_utils import GenerationMixin',
        'from ...generation import GenerationMixin')
    _checked(text,'7961e7841bb5aae550e42ccb13e2de8e4802057a0e8878baf5fe224f97e37002')
    tree=ast.parse(text)
    # The only binding adaptation is this private module's GenerationMixin import.
    tree.body=[node for node in tree.body if not (isinstance(node,ast.ImportFrom)
        and node.level==3 and node.module=='generation')]
    original=types.ModuleType('transformers.models.qwen2_5_vl._ilvr_model_source_oracle')
    original.__file__=str(modeling)
    original.__package__='transformers.models.qwen2_5_vl'
    original.GenerationMixin=SourceGenerationMixin
    sys.modules[original.__name__]=original
    exec(compile(tree,str(modeling),'exec'),original.__dict__)
    cls=original.Qwen2_5_VLForConditionalGeneration
    for name in ('generate','_sample','_update_model_kwargs_for_generation','_get_initial_cache_position'):
        if getattr(cls,name).__module__ != module.__name__:
            raise AssertionError(f'wrong source-oracle dispatch for {name}')
    _SOURCE_CLASS=cls
    return cls


def source_evaluation_functions(*, dataset_root=''):
    import logging
    import os
    import re
    import time
    import torch
    from PIL import Image
    from typing import Any, Dict, List, Union
    from transformers import LogitsProcessor, LogitsProcessorList
    text=_checked(EVAL_SOURCE.read_text(),'481a300d2ecc4c29b6a2a772a9d7cae4d3a339a1a3b1b6fc7e3dd7cb2d8bf904')
    names={'LatentTemplateLogitsProcessor','load_model','_resolve_image_paths','open_images',
        '_clean_answer_span','extract_final_answer','_strip_special_tokens',
        'extract_assistant_content','normalize_for_match','run_one_example'}
    tree=ast.parse(text)
    tree.body=[node for node in tree.body if isinstance(node,(ast.ClassDef,ast.FunctionDef)) and node.name in names]
    namespace=dict(torch=torch,os=os,re=re,time=time,logging=logging,Image=Image,
        Any=Any,Dict=Dict,List=List,Union=Union,LogitsProcessorList=LogitsProcessorList,
        Qwen2_5_VLForConditionalGeneration=source_model_class(),BASE_DATASET_DIR=str(dataset_root),
        _yes_set={'yes','true','a'},_no_set={'no','false','b'})
    utility_source=EVAL_SOURCE.with_name('utils_deepseed.py.txt')
    utility_text=_checked(utility_source.read_text(),'d45dc3339fe1ad02836f1eeb7fa1f5171d32ec499460fe51e922d172e9e0cf9f')
    utility_tree=ast.parse(utility_text)
    utility_tree.body=[node for node in utility_tree.body if isinstance(node,ast.ClassDef) and node.name=='LatentTemplateLogitsProcessor']
    namespace['LogitsProcessor']=LogitsProcessor
    exec(compile(utility_tree,str(utility_source),'exec'),namespace)
    exec(compile(tree,str(EVAL_SOURCE),'exec'),namespace)
    return namespace

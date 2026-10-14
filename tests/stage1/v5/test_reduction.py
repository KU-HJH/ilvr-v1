"""Mutation witnesses: configured GAS tail and empty original blocks."""
import importlib.util
from types import SimpleNamespace

import pytest
import torch
from accelerate.utils import DistributedType
from transformers import Trainer, TrainingArguments

# Shared with Task7: four ranks; empty rank and microbatch, unequal block counts.
FOUR_RANK_WINDOWS = (
    ((2, 1, 0, 0), (1, 0, 1, 0), (0, 0, 0, 0), (1, 2, 0, 1)),
    ((1, 0), (1, 1), (0, 0), (0, 2)),
)


def api():
    assert importlib.util.find_spec('stage1.v5_reduction'), 'exact window reduction missing'
    from stage1.v5_reduction import V5Window, v5_auxiliary_scale
    return V5Window, v5_auxiliary_scale


def test_scale_keeps_configured_gas_on_partial_tail_and_empty_blocks():
    Window, scale = api()
    assert scale(Window(10, 4, 4, 4)) == 1.6
    assert scale(Window(5, 2, 4, 4)) == 3.2
    assert scale(Window(0, 2, 4, 4)) == 0.
    for bad in ((-1, 1, 4, 4), (1, 0, 4, 4), (1, 5, 4, 4), (1, 1, 0, 4)):
        with pytest.raises(ValueError): Window(*bad)


@pytest.mark.parametrize('counts', FOUR_RANK_WINDOWS)
def test_real_parent_dispatch_matches_independent_global_gradient(counts, tmp_path):
    """Actual Trainer dispatch/scaling; fake DS boundary records and executes backward.

    CPU cannot create a GPU ZeRO engine. Task7 consumes these literal windows
    for that missing distributed engine witness, rather than treating this as DS.
    """
    Window, scale = api()
    grads, kwargs_seen = [], []
    total = sum(map(sum, counts))
    for rank, microbatches in enumerate(counts):
        parameter = torch.nn.Parameter(torch.tensor(2.))
        model = torch.nn.Linear(1, 1)
        model.weight = parameter
        trainer = Trainer.__new__(Trainer)
        trainer.args = TrainingArguments(str(tmp_path), use_cpu=True, gradient_accumulation_steps=4, report_to=[])
        trainer.state = SimpleNamespace(global_step=0)
        trainer.optimizer = torch.optim.AdamW([parameter], lr=.01)
        trainer.model_accepts_loss_kwargs = False
        trainer.compute_loss_func = None
        trainer.use_apex = False
        trainer.use_cpu_amp = False
        trainer._prepare_inputs = lambda x: x
        trainer.accelerator = SimpleNamespace(distributed_type=DistributedType.DEEPSPEED,
            backward=lambda loss, **kw: (kwargs_seen.append(kw), loss.backward()))
        window = Window(total, len(microbatches), 4, 4)
        trainer.compute_loss = lambda _m, x, **kw: .1 * parameter.square() + scale(window) * x['blocks'] * parameter.pow(3)
        batches, _ = Trainer.get_batch_samples(trainer, iter({'blocks': n} for n in microbatches), 4)
        assert len(batches) == len(microbatches)
        for batch in batches: Trainer.training_step(trainer, model, batch)
        grads.append(parameter.grad.clone())
    # Independent global objective: native configured sixteen denominator,
    # auxiliary actual original-step denominator; empty blocks remain counted.
    p = torch.tensor(2., requires_grad=True)
    objective = .1 * len(counts) * len(counts[0]) / 16 * p.square() + p.pow(3)
    objective.backward()
    torch.testing.assert_close(torch.stack(grads).mean(), p.grad)
    assert kwargs_seen == [{'scale_wrt_gas': False}] * (4 * len(counts[0]))


def _parent_resume_boundary_code():
    """Execute actual parent statements, not a reimplemented scheduling helper.

    No loop/model/optimizer/backward is run. Parent resume assignments include
    the nested ignore_data_skip branch and its GAS AugAssign. Boundary includes
    the actual policy-gated skipped-source correction.
    """
    import ast
    import inspect
    import textwrap
    tree = ast.parse(textwrap.dedent(inspect.getsource(Trainer._inner_training_loop)))
    resume = []
    boundary = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        target_names = {t.id for t in node.targets if isinstance(t, ast.Name)}
        if 'epochs_trained' in target_names and 'self.state.global_step' in ast.unparse(node):
            resume.append(node)
        if 'boundary_step' in target_names and ast.unparse(node.value) == 'step + 1':
            for parent in ast.walk(tree):
                body = getattr(parent, 'body', None)
                if isinstance(body, list) and node in body:
                    start = body.index(node)
                    for statement in body[start:]:
                        boundary.append(statement)
                        if isinstance(statement, ast.Assign) and any(
                                isinstance(t, ast.Name) and t.id == 'do_sync_step' for t in statement.targets):
                            break
                    break
    for node in ast.walk(tree):
        if (isinstance(node, ast.If) and ast.unparse(node.test) == 'not args.ignore_data_skip'
                and any(isinstance(child, ast.AugAssign) and isinstance(child.target, ast.Name)
                        and child.target.id == 'steps_trained_in_current_epoch' for child in node.body)):
            resume.append(node)
            break
    assert len(resume) == 2 and boundary
    def compile_nodes(nodes):
        return compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),
                       '<actual Trainer resume/boundary statements>', 'exec')
    prefetch_setup = []
    request = []
    for parent in ast.walk(tree):
        body = getattr(parent, 'body', None)
        if not isinstance(body, list): continue
        for index,node in enumerate(body):
            if (isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id=='remainder' for t in node.targets)
                    and 'num_examples' in ast.unparse(node.value)):
                for statement in body[index:]:
                    if isinstance(statement, ast.For): break
                    prefetch_setup.append(statement)
                break
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id=='num_batches' for t in node.targets):
            request.append(node)
    assert prefetch_setup and len(request)==1
    return compile_nodes(resume), compile_nodes(boundary), compile_nodes(prefetch_setup), compile_nodes(request)


@pytest.mark.parametrize('num_examples',[10,37])
@pytest.mark.parametrize('saved_step,want_epoch,want_skip', [(0,0,0),(1,0,4),(2,0,8),(3,1,0),(6,2,0)])
def test_v5_parent_resume_keeps_source_positions_and_syncs_partial_tail(tmp_path,saved_step,want_epoch,want_skip,num_examples):
    from torch.utils.data import DataLoader
    from stage1.v5_trainer import CustomTrainerStage1V5
    from stage1.v5_metadata import V5BatchMetadata,V5Block
    trainer = CustomTrainerStage1V5.__new__(CustomTrainerStage1V5)
    trainer.args = TrainingArguments(str(tmp_path),use_cpu=True,per_device_train_batch_size=1,
        gradient_accumulation_steps=4,max_steps=6,report_to=[])
    from transformers.trainer_callback import TrainerState
    checkpoint = tmp_path/f'checkpoint-{saved_step}'
    checkpoint.mkdir()
    state_path=checkpoint/'trainer_state.json'
    TrainerState(global_step=saved_step).save_to_json(str(state_path))
    trainer.state = TrainerState.load_from_json(str(state_path))
    trainer.accelerator = SimpleNamespace(device=torch.device('cpu'),num_processes=1,
        reduce=lambda value,**kw:value)
    loader = DataLoader(list(range(num_examples)),batch_size=1,sampler=list(range(10)))
    values = trainer.set_initial_training_values(trainer.args,loader,16)
    resume, boundary, prefetch_setup, request = _parent_resume_boundary_code()
    env = dict(self=trainer,args=trainer.args,num_update_steps_per_epoch=values[1])
    exec(resume,env)
    assert (env['epochs_trained'],env['steps_trained_in_current_epoch']) == (want_epoch,want_skip)
    observed, updates = [], []
    for epoch in range(env['epochs_trained'],values[0]):
        skipped = env['steps_trained_in_current_epoch'] if epoch == env['epochs_trained'] else 0
        sources = iter({'source':i,'stage1_v5_metadata':V5BatchMetadata((V5Block(0,0,1,2,(),(),()),))}
                       for i in range(skipped,10))
        local_step = -1
        env.update(num_examples=values[2],steps_in_epoch=10)
        exec(prefetch_setup,env)
        for update_step in range(env['total_updates']):
            env['update_step']=update_step
            exec(request,env)
            batches,_ = trainer.get_batch_samples(sources,env['num_batches'])
            for batch in batches:
                local_step += 1
                env.update(step=local_step,steps_skipped=skipped,steps_in_epoch=10)
                exec(boundary,env)
                observed.append((epoch,batch['source'],env['do_sync_step']))
                assert batch['stage1_v5_window'].configured_gas == 4
                if env['do_sync_step']:
                    updates.append((epoch,batch['source']))
    expected = [(epoch,i,i in (3,7,9)) for epoch in range(want_epoch,2)
                for i in range(want_skip if epoch == want_epoch else 0,10)]
    print(f'V5_PARENT_RESUME observed saved={saved_step} sources_sync={observed} updates={updates}')
    assert observed == expected
    assert values[0] == 2 and values[1] == 3
    assert len(updates) == 6-saved_step
    if saved_step == 1:
        assert observed[:6] == [(0,4,False),(0,5,False),(0,6,False),(0,7,True),(0,8,False),(0,9,True)]
    if saved_step == 2:
        assert observed[:2] == [(0,8,False),(0,9,True)]
    print(f'V5_PARENT_RESUME saved={saved_step} epoch={want_epoch} skip={want_skip} sources_sync={observed} updates={updates}')


def test_standard_parent_retains_historical_floor_epoch_budget(tmp_path):
    from torch.utils.data import DataLoader
    trainer = Trainer.__new__(Trainer)
    args = TrainingArguments(str(tmp_path),use_cpu=True,gradient_accumulation_steps=4,max_steps=6,report_to=[])
    assert trainer.set_initial_training_values(args,DataLoader(range(10),batch_size=1),16)[1] == 2
